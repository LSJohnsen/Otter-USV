import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OTTER_API_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if OTTER_API_DIR not in sys.path:
    sys.path.insert(0, OTTER_API_DIR)

SAVE_DIR = os.path.join(CURRENT_DIR, "ppo_saves")   
os.makedirs(SAVE_DIR, exist_ok=True)

import Otter_api
from DRL_control import Otter_simulator_DRL
from lib.plotTimeSeries import *
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import pandas as pd
from gymnasium.spaces import Box
import torch
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from torch import nn
from lib.Performance_metrics import PerformanceMetrics
from logs.IO import log_to_csv, log_params as io_log_params


# Use cpu for PPO
device = torch.device("cpu")
print(f"Using device: {device}")

simulator_environments = 8                                                                              # Number of simulation environments -> change depending on cpu capacity ~2-16
wave_function = False                                                                                   # adds a simple eastward wave function
use_target_coordinates = False                                                                          # To use coordinates as a target or to use a linear path
use_moving_target = True                                                                                # To use moving target instead of target list (path following)
target_list = [[0, 10000]]                                                                              # List of targets to use if use_target_coordinates is set to True
end_when_last_target_reached = True                                                                     # Ends the simulation when the final target is reached
moving_target_start = [0, -10]                                                                         # Start point of the moving target if use_moving_target is set to True
moving_target_increase = [-0.5, 0.0]                                                                     # Movement of the moving target each second                                                                                  # How many meters target should move each simulation before truncation
target_radius = 0.2                                                                                     # Radius from center of target that counts as target reached
verbose = True                                                                                          # Enable verbose printing
store_force_file = False                                                                                # Store the simulated control forces in a .csv file
circular_target = True                                                                                  # Make the moving target a circle in the simulation
animate_path = False
training_timesteps = 1000000                                                                            # Set timesteps (10mil-50mil+ depending on straight/circle)
log_results = False                                                                                     # log sim to csv, false when training

start_north = -20 #not used?                                                                                       # Target north position from reference point
start_east = -20 #not used?                                                                             # Target east position from reference point
randomize_position = True                                                                               # Used to randomize usv start position for better training
randomize_path = True                                                                                   # Randomizes paths to circular/straight line/stationary
v_north = 0                                                                                             # Moving target speed north (m/s)
v_east = -1.5                                                                                           # Moving target speed east (m/s)
radius = 40 # SIM LOGIC LINE~500 X/Y START (FIX)                                                        # If not tracking a circular motion 
max_target_delta = 250                                                                                  # Max distance target moves before truncation
max_episode_time = 266.66
v_circle = 1.5                                                                                          # Angular velocity (m/s)
side_length = 50                                                                                        # Square tracking side length
side_target_speed = 1                                                                                   # Speed of square target

numDataPoints = 830                                                                                     # number of 3D data points
FPS = 60                                                                                                # frames per second (animated GIF)
filename = '3D_animation.gif'                                                                           # data file for animated GIF
browser = 'chrome'

CHECKPOINT_MODEL = os.path.join(SAVE_DIR, "ppo_otter_checkpoint_rand.zip")                                   # checkpoint model path
CHECKPOINT_VECNORM = os.path.join(SAVE_DIR, "ppo_otter_checkpoint_vecnormalize_rand.pkl")
FINAL_MODEL = os.path.join(SAVE_DIR, "ppo_otter_model.zip")
FINAL_VECNORM = os.path.join(SAVE_DIR, "vecnormalize.pkl")

otter = Otter_api.otter()

simulator = Otter_simulator_DRL.OtterSimDRL(target_list,
                                            use_target_coordinates,
                                            target_radius,
                                            use_moving_target,
                                            moving_target_start,
                                            moving_target_increase,
                                            end_when_last_target_reached,
                                            verbose,
                                            store_force_file,
                                            circular_target,
                                            radius)

print("initialized otter api and simulator")

otter.controls = ["Left propeller shaft speed (rad/s)", "Right propeller shaft speed (rad/s)"]           # values needed for the plotting
otter.dimU = len(otter.controls)


class OtterEnv(gym.Env):
    def __init__(self, simulator, otter):
        super().__init__()                                                                               # call gym.env constructor

        self.simulator = simulator
        self.otter = otter
        self.path_modes = ["circle", "line", "stationary"] 
        self.path_mode = "circle"

        self.sampletime = 0.1  # iteration updates
        self.episode_duration = 400000  # no. simulation samples (truncates at distances, just ensure not too small)
        self.sim_duration = int(self.episode_duration / self.sampletime)  # sim duration
        self.current_step = 0
        self.target_arc_length = 0
        self.prev_tau_X = 0

        self.Umax = 6 * 0.5144
        self.max_force = 200

        self.simData = []
        self.targetData = []
        self.simTime = []
        self.distanceHistory = []
        self.headingErrorHistory = []
        self.yawHistory = []
        self.last_sim_data = None
        self.last_target_data = None
        self.last_sim_time = None
        self.last_yaw_history = None
        
        # training metrics
        self.IAE_distance_History = []
        self.IAE_heading_History = []

        # final evaluation metrics
        self.metrics = PerformanceMetrics()
        self.last_IAE_distance = 0.0
        self.last_IAE_heading = 0.0
        self.last_ISU = 0.0
        self.last_ISU_normalized = 0.0
        self.last_IAU = 0.0
        self.cum_distance = 0.0              # distance  avg distance
        self.reached_target_time = 0.0       # time of first intercept
        self.reached_flag = False            # has target been intercepted

        self.last_avg_distance = 0.0         # stored at end of episode (for logging)
        self.last_reached_target_time = 0.0  # stored at end of episode (for logging)

        self.initial_target = list(self.simulator.moving_target)
        self.has_plotted = False

        # min/max distance to target, angle to target, surge/sway velocity
        self.observation_space = Box(low=np.array([-1,
                                                   -np.pi,
                                                   -1,
                                                   -1],
                                                  dtype=np.float32),
                                     high=np.array([1,
                                                    np.pi,
                                                    1,
                                                    1],
                                                   dtype=np.float32))

        # min/max forces in surge/yaw normalized
        self.action_space = Box(low=np.array([-1,
                                              -1],
                                             dtype=np.float32),
                                high=np.array([1,
                                               1],
                                              dtype=np.float32))

    def step(self, action):

        self.current_step += 1
        truncated_count = 0

        # Chooses action which are passed to the simulator at current sampletime
        tau_X, tau_N = action
        eta, nu, target, distance_to_target, heading_error, u_actual = self.simulator.simulate_step(
            self.sampletime,
            self.otter,
            tau_X,
            tau_N
        )
        self.metrics.update(
            distance_to_target=distance_to_target,
            heading_error=heading_error,
            u1=u_actual[0],
            u2=u_actual[1],
            dt=self.sampletime,
        )
   

        commands = np.array([tau_X, tau_N])
        actuals = u_actual

        # Stack all usv states into sequence of arrays
        full_state = np.hstack([eta,
                                nu,
                                commands,
                                actuals])

        self.simData.append(full_state)
        self.targetData.append(np.array(self.simulator.moving_target, dtype=float))
        self.simTime.append(self.current_step * self.sampletime)
        self.yawHistory.append(eta[5])
        self.distanceHistory.append(distance_to_target)
        self.headingErrorHistory.append(heading_error)

        self.cum_distance += distance_to_target * self.sampletime
        if (not self.reached_flag) and (distance_to_target < self.simulator.surge_setpoint):
            self.reached_flag = True
            self.reached_target_time = self.current_step * self.sampletime

        # Normalize target distances (if distance > ... 1) 
        if self.simulator.circular_target:
            r = float(getattr(self.simulator, "radius", radius))
            normalized_distance = np.clip(distance_to_target / r, -1.0, 1.0)
        else:
            normalized_distance = np.tanh(distance_to_target / max_target_delta)

        
        obs = np.array([normalized_distance,
                        heading_error,
                        np.clip(nu[0] / self.Umax, -1.0, 1.0),  # surge velocity normalized
                        np.clip(nu[1] / self.Umax, -1.0, 1.0),  # sway velocity normalized
                        ], dtype=np.float32)

        # Determine target deltas for episode termination
        episode_time = self.current_step * self.sampletime
        truncated = episode_time >= max_episode_time
        truncated_count += 1
        if truncated_count % 100 == 0:
            print(f"[{self.current_step}] Target at {self.simulator.moving_target}, "
                  f"Initial at {self.initial_target}, "
                  f"Δ={np.linalg.norm(np.array(self.simulator.moving_target) - np.array(self.initial_target)):.2f}")

        if truncated:
            self.last_sim_data = self.simData.copy()
            self.last_target_data = self.targetData.copy()
            self.last_sim_time = self.simTime.copy()
            self.last_yaw_history = self.yawHistory.copy()

            if self.current_step > 0:
                self.last_avg_distance       = self.cum_distance / self.current_step
            else:
                self.last_avg_distance       = 0.0
                self.last_reached_target_time = self.reached_target_time
            param_dict = {
                "Control_method": "DRL",
                "IAE_distance": self.last_IAE_distance,
                "IAE_heading":  self.last_IAE_heading,
                "ISU":          self.last_ISU,
                "ISU_normalized": self.last_ISU_normalized,
                "IAU":          self.last_IAU,
                "avg_distance_to_target": self.last_avg_distance,
                "reached_target_time":    self.last_reached_target_time,
            }
            
            
            if log_results:
                io_log_params(param_dict, filename="parameters_DRL.txt", verbose=True)
                log_to_csv(
                    simTime=self.last_sim_time,
                    simData=self.last_sim_data,
                    targetData=self.last_target_data,
                    filename="sim_log_DRL.csv",
                    verbose=True)

        if not hasattr(self, 'last_distance'):
            self.last_distance = distance_to_target

        # Reward handling
        reward = 0
        terminated = truncated
        info = {}

        #distance_to_target, u = nu[0], v = nu[1], r = nu[5]
        intercept_tolerance = 0.5                                   # should be set to UOWC width? 
        w = np.exp(-(distance_to_target / intercept_tolerance)**2)  # exp scaling for smoother movement close to target

        reward += 3 * (self.last_distance - distance_to_target)     # reward reducing distance
        reward -= 0.1 * self.last_distance                          # penalize distance to target
    

        is_moving = 1.0 if self.simulator.use_moving_target else 0.0

        # heading reward when further away from target
        reward += (1.0 - w) * 0.5 * np.cos(heading_error)

        # for station-keeping
        reward -= w * (1.0 - is_moving) * (1.0*abs(nu[0]) + 1.0*abs(nu[1]) + 0.8*abs(nu[5]))

        # reduce oscillation
        reward -= w * is_moving * (0.6*abs(nu[3]) + 0.6*abs(nu[5]))

        # smooth actuation close to target
        reward -= w * 0.02 * (abs(tau_X) + abs(tau_N))

        self.last_distance = distance_to_target
        '''
        if self.simulator.circular_target:
            reward -= 0.1 * abs(nu[5])
        else:
            reward -= 0.1 * distance_to_target
            reward += 0.5 * np.cos(heading_error)
            reward -= 0.2 * abs(nu[1])
            reward -= 0.1 * abs(nu[5])

        self.last_distance = distance_to_target


        if distance_to_target < self.simulator.surge_setpoint:
            reward += 5.0 # test different here (oscillating at circular)
        '''

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        # pick the “live” data if it exists, otherwise the last-episode
        simData_list = self.simData if len(self.simData) > 1 else self.last_sim_data
        targetData_list = self.targetData if len(self.targetData) > 1 else self.last_target_data
        simTime_list = self.simTime if len(self.simTime) > 1 else self.last_sim_time
        yaw_list = self.yawHistory if len(self.yawHistory) > 1 else self.last_yaw_history

        # convert to arrays
        simData = np.array(simData_list)
        targetData = np.array(targetData_list)
        simTime = np.array(simTime_list)
        yawHistory = np.array(yaw_list)

        if simData.size and simData.shape[0] > 1:
            plotPosTar2(simTime, simData, 1, targetData)
            plotVehicleStates(simTime, simData, 2)
            plotControls(simTime, simData, self.otter, 3)
            #plotSpeed(simTime, simData, 5)
            plotSurge(simTime, simData, 6)
            plotYaw(simTime, simData, 7) 
            plt.show()

    def reset(self, seed=None, options=None):


        #  store last metrics 
        if self.current_step > 0:
            self.last_IAE_distance, self.last_IAE_heading = self.metrics.get_IAE()
            self.last_ISU = self.metrics.get_ISU()
            self.last_ISU_normalized = self.metrics.get_ISU_normalized()
            self.last_IAU = self.metrics.get_IAU()

        else:
            self.last_IAE_distance = 0.0
            self.last_IAE_heading = 0.0
            self.last_ISU = 0.0
            self.last_ISU_normalized = 0.0
            self.last_IAU = 0.0

        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        #  reset episode 
        self.current_step = 0
        self.target_arc_length = 0.0
        self.metrics.reset()
        self.cum_distance = 0.0
        self.reached_target_time = 0.0
        self.reached_flag = False

        # reset moving target & simulator time
        
        if randomize_path == True:
            self.path_mode = np.random.choice(self.path_modes)

        if self.path_mode == "circle":
            self.simulator.circular_target = True
            self.simulator.use_moving_target = True
            # randomize radius
            self.simulator.radius = float(self.np_random.uniform(20.0, 60.0))
            # reset any circular phase
            self.simulator.asd = 0.0

        elif self.path_mode == "line":
            self.simulator.circular_target = False
            self.simulator.use_moving_target = True
            velocity = float(self.np_random.uniform(-0.5, 1))
            heading = float(self.np_random.uniform(-np.pi, np.pi))
            self.simulator.moving_target_increase = np.array([
                velocity * np.cos(heading),
                velocity * np.sin(heading)
            ], dtype=float)

        elif self.path_mode == "stationary":
            self.simulator.circular_target = False
            self.simulator.use_moving_target = False
            self.simulator.moving_target_increase = np.array([0.0, 0.0], dtype=float)

        self.simulator.moving_target = self.simulator.moving_target_start.copy()
        self.initial_target = list(self.simulator.moving_target)

        #  choose initial USV state relative to target (should randomize during training)
        rng = self.np_random 
        
        target_pos = np.array(self.simulator.moving_target, dtype=float)
        target_x, target_y = target_pos


        if randomize_position:
            if self.simulator.circular_target:
                r_min = 5.0
                r_max = float(getattr(self.simulator, "radius", radius))
            else:
                r_min, r_max = 5.0, 50.0

            alpha = rng.uniform(-np.pi, np.pi)      # randomize direction from target
            r = rng.uniform(r_min, r_max)           # randomize distance to target

            x = target_x + r * np.cos(alpha)        # start x/y based on target position 
            y = target_y + r * np.sin(alpha)

            
            yaw = rng.uniform(-np.pi, np.pi)        # randomize heading

            self.r0 = float(r)                      # store for log
            self.alpha0 = float(alpha)

        else:                                       # In testing to validate against other controls
            x = target_x
            y = target_y + 10.0  
            self.r0 = 0.0
            self.alpha0 = 0.0

        eta_initial = [x, y, 0.0, 0.0, 0.0, yaw]
        self.simulator.initial_state(eta_initial)

        #  reset simulator
        distance_to_target, heading_error, nu = self.simulator.reset_simulation()

        # clear histories 
        self.simData = []
        self.targetData = []
        self.simTime = []
        self.yawHistory = []
        self.distanceHistory = []
        self.headingErrorHistory = []

        eta0 = self.simulator.eta.copy()
        nu0  = self.simulator.nu.copy()
        u0   = self.simulator.u_actual.copy()
        commands0 = np.array([0.0, 0.0])   # no control before first step

        full_state0 = np.hstack([eta0, nu0, commands0, u0])
        self.simData.append(full_state0)
        self.targetData.append(np.array(self.simulator.moving_target, dtype=float))
        self.simTime.append(0.0)
        self.yawHistory.append(eta0[5])
        self.distanceHistory.append(distance_to_target)
        self.headingErrorHistory.append(heading_error)

        # initial observation 
        if self.simulator.circular_target:
            normalized_distance = np.clip(distance_to_target / self.simulator.circle_radius,
                                        -1.0, 1.0)
        else:
            normalized_distance = np.tanh(distance_to_target / max_target_delta)  

        obs = np.array([
            normalized_distance,
            heading_error,
            np.clip(nu[0] / self.Umax, -1.0, 1.0),
            np.clip(nu[1] / self.Umax, -1.0, 1.0),
        ], dtype=np.float32)

        info = {}
        return obs, info


    def seed(self, seed=None):
        pass


def make_env():
    def _init():
        otter = Otter_api.otter()

        otter.controls = ["Left propeller shaft speed (rad/s)", "Right propeller shaft speed (rad/s)"]   # Some values needed for the plotting
        otter.dimU = len(otter.controls)

        simulator = Otter_simulator_DRL.OtterSimDRL(
            target_list,
            use_target_coordinates,
            target_radius,
            use_moving_target,
            moving_target_start,
            moving_target_increase,
            end_when_last_target_reached,
            verbose,
            store_force_file,
            circular_target
        )
        return OtterEnv(simulator=simulator, otter=otter)
    return _init


# logs the distances and headings during training for IAE plotting
class CallBackLog(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.IAE_distance_history = []
        self.IAE_heading_history = []

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals["dones"]):
            if done:
                IAE_distance = self.training_env.get_attr("last_IAE_distance", i)[0]
                IAE_heading = self.training_env.get_attr("last_IAE_heading", i)[0]  

                self.IAE_distance_history.append(IAE_distance)
                self.IAE_heading_history.append(IAE_heading)

                if self.verbose and len(self.IAE_distance_history) % 10 == 0:
                    print(f"Episode {len(self.IAE_distance_history)} - "
                          f"IAE Distance: {IAE_distance:.2f}, Heading: {IAE_heading:.2f}")

        return True

    def return_log(self):
        return self.IAE_distance_history, self.IAE_heading_history


if __name__ == "__main__":

    print(f"Using device: {device}")

    n_envs = 8

    mode = int(input("\n\nChoose '1' for model training or '2' to run saved model: "))
    if mode == 1:

        env = SubprocVecEnv([make_env() for _ in range(n_envs)])

        # If a previous checkpoint exists, resume from it, otherwise start fresh
        if os.path.exists(CHECKPOINT_MODEL) and os.path.exists(CHECKPOINT_VECNORM):
            print("\nFound checkpoint. Loading model and VecNormalize to continue\n")
            env = VecNormalize.load(CHECKPOINT_VECNORM, env)
            model = PPO.load(CHECKPOINT_MODEL, env=env, device="cpu")
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            model = PPO(
                MlpPolicy,
                env,
                verbose=1,
                device="cpu",
                normalize_advantage=True,
                gae_lambda=0.98,
                learning_rate=0.001,
                clip_range=0.2,
                n_steps=4096,
                ent_coef=0.01,
                target_kl=None,
                policy_kwargs=dict(activation_fn=nn.Tanh),
            )

        print("\nTraining model: \n")
        IAE_callback = CallBackLog(verbose=1)

        try:
            model.learn(
                total_timesteps=training_timesteps,
                callback=IAE_callback,
            )
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Saving checkpoint before exiting...")
            model.save(CHECKPOINT_MODEL)
            if isinstance(env, VecNormalize):
                env.save(CHECKPOINT_VECNORM)
            print("Checkpoint saved. Exiting.")
            sys.exit(0)

        # Final save
        model.save(FINAL_MODEL)
        if isinstance(env, VecNormalize):
            env.save(FINAL_VECNORM)


    elif mode == 2:
        print("Loading previously saved model for evaluation")

        # Use single env 
        eval_env = DummyVecEnv([make_env()])

        # Load the VecNormalize as used during training
        eval_env = VecNormalize.load(CHECKPOINT_VECNORM, eval_env)

        # set to evaluation mode
        eval_env.training = False          #  no updates to running stats
        eval_env.norm_reward = False       #  don't normalize rewards during eval

        # Load the trained policy
        model = PPO.load(CHECKPOINT_MODEL, env=eval_env, device="cpu")


    else:
        print("Chosen option is not valid.")

    if mode == 2:
        IAE_callback = CallBackLog(verbose=1)
        IAE_distance, IAE_heading = IAE_callback.return_log()
        #print(f"Final IAE Distance: {IAE_distance[-1]}, Final IAE Heading: {IAE_heading[-1]}")
        plt.figure()
        plt.plot(IAE_distance, label="IAE Distance", linewidth=0.8)
        plt.xlabel("Episode")
        plt.ylabel("IAE")
        plt.title("IAE Distance per Episode During Training")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(IAE_heading, label="IAE Heading", linewidth=0.8)
        plt.xlabel("Episode")
        plt.ylabel("IAE")
        plt.title("IAE Heading per Episode During Training")
        plt.legend()
        plt.grid(True)
        plt.show()

    eval_env = DummyVecEnv([make_env()])
    if mode == 2:
        eval_env = VecNormalize.load(CHECKPOINT_VECNORM, eval_env)

    iae_distances = []
    iae_headings = []
    episode_actions = []
    episode_count = 0
    logged_once = False

    obs = eval_env.reset()

    # monte carlo logs for eval
    mc_log = {"IAE_distance": [], 
                "IAE_heading": [],
                "avg_distance": [],
                "intercept_time": [],
                "success": [],
                "r0": [],
                "alpha0": []
                }
    
    # for MC use 540000 for 200 samples (saving will be slow)
    for i in range(10000):

        env_single = eval_env.envs[0]

        last_target = env_single.simulator.moving_target.copy()
        action, _states = model.predict(obs)
        
        episode_actions.append(action)
        obs, rewards, dones, infos = eval_env.step(action)

        if dones[0]:
            
            mc_log["IAE_distance"].append(env_single.last_IAE_distance)
            mc_log["IAE_heading"].append(env_single.last_IAE_heading)
            mc_log["avg_distance"].append(env_single.last_avg_distance)
            mc_log["intercept_time"].append(env_single.last_reached_target_time)
            mc_log["success"].append(int(env_single.reached_flag))
            mc_log["r0"].append(env_single.r0)
            mc_log["alpha0"].append(env_single.alpha0)

            iae_distances.append(env_single.last_IAE_distance)
            iae_headings.append(env_single.last_IAE_heading)
            
            if episode_count == 0:
                actions_arr = np.vstack(episode_actions)
                plt.figure()
                plt.plot(actions_arr[:, 0], linewidth=1, label="Normalized Surge command (τ_X)")
                plt.xlabel("Time step")
                plt.ylabel("Normalized Surge command (τ_X)")
                plt.title("Surge Control Action")
                plt.legend()
                plt.grid(True)
                plt.show()

                plt.figure()
                plt.plot(actions_arr[:, 1], linewidth=1, label="Normalized Yaw command (τ_X)")
                plt.xlabel("Time step")
                plt.ylabel("Normalized Yaw command (τ_N)")
                plt.title("Yaw Control Action")
                plt.legend()
                plt.grid(True)
                plt.show()
            
                if len(env_single.simData) > 0:
                    simData_arr    = np.array(env_single.simData)
                    targetData_arr = np.array(env_single.targetData)
                    simTime_arr    = np.array(env_single.simTime)
                

                if episode_count == 0:
                    actions_arr = np.vstack(episode_actions)
                    ...

                # increase to compare more plots
                if episode_count < 1:
                    env_single.render()

                episode_count += 1
                episode_actions = []
                obs = eval_env.reset()

    print(f"Average eval IAE distance: {np.mean(iae_distances):.2f}")
    print(f"Average eval IAE heading:  {np.mean(iae_headings):.2f}")

    # save mc for plotting 
    df = pd.DataFrame(mc_log)
    df.to_csv("monte_carlo_results.csv", index=False)
