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
from gymnasium.spaces import Box
import torch
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from torch import nn

# Use cpu for PPO
device = torch.device("cpu")
print(f"Using device: {device}")

simulator_environments = 8                                                                              # Number of simulation environments -> change depending on cpu capacity ~2-16
wave_function = False                                                                                   # adds a simple eastward wave function
use_target_coordinates = False                                                                          # To use coordinates as a target or to use a linear path
use_moving_target = True                                                                                # To use moving target instead of target list (path following)
target_list = [[0, 10000]]                                                                              # List of targets to use if use_target_coordinates is set to True
end_when_last_target_reached = True                                                                     # Ends the simulation when the final target is reached
moving_target_start = [0, 0]                                                                            # Start point of the moving target if use_moving_target is set to True
moving_target_increase = [1.5, 0.0]                                                                     # Movement of the moving target each second
max_target_delta = 60                                                                                   # How many meters target should move each simulation before truncation
target_radius = 0.2                                                                                     # Radius from center of target that counts as target reached
verbose = True                                                                                          # Enable verbose printing
store_force_file = False                                                                                # Store the simulated control forces in a .csv file
circular_target = True                                                                                  # Make the moving target a circle in the simulation
animate_path = False
training_timesteps = 30000000                                                                           # Set timesteps (10mil-50mil+ depending on straight/circle)

start_north = -20                                                                                       # Target north position from reference point
start_east = -20                                                                                        # Target east position from reference point
v_north = 0                                                                                             # Moving target speed north (m/s)
v_east = -1.5                                                                                           # Moving target speed east (m/s)
radius = 20                                                                                             # If tracking a circular motion
if circular_target:                                                                                     # Full circle length
    max_target_delta = 2 * radius * np.pi
v_circle = 1.5                                                                                          # Angular velocity (m/s)
side_length = 50                                                                                        # Square tracking side length
side_target_speed = 1                                                                                   # Speed of square target

numDataPoints = 830                                                                                     # number of 3D data points
FPS = 60                                                                                                # frames per second (animated GIF)
filename = '3D_animation.gif'                                                                           # data file for animated GIF
browser = 'chrome'

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
                                            )

print("initialized otter api and simulator")

otter.controls = ["Left propeller shaft speed (rad/s)", "Right propeller shaft speed (rad/s)"]           # values needed for the plotting
otter.dimU = len(otter.controls)


class OtterEnv(gym.Env):
    def __init__(self, simulator, otter):
        super().__init__()                                                                               # call gym.env constructor

        self.simulator = simulator
        self.otter = otter

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
        self.IAE_distance_History = []
        self.IAE_heading_History = []
        self.last_sim_data = None
        self.last_target_data = None
        self.last_sim_time = None
        self.last_yaw_history = None

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

        commands = np.array([tau_X, tau_N])
        actuals = u_actual

        # Stack all usv states into sequence of arrays
        full_state = np.hstack([eta,
                                nu,
                                commands,
                                actuals])

        self.simData.append(full_state)
        self.targetData.append(target)
        self.simTime.append(self.current_step * self.sampletime)
        self.yawHistory.append(eta[5])
        self.distanceHistory.append(distance_to_target)
        self.headingErrorHistory.append(heading_error)

        # Normalize target distances
        if circular_target:
            normalized_distance = np.clip(distance_to_target / radius, -1.0, 1.0)
        else:
            normalized_distance = np.tanh(distance_to_target / max_target_delta)

        obs = np.array([normalized_distance,
                        heading_error,
                        np.clip(nu[0] / self.Umax, -1.0, 1.0),  # surge velocity normalized
                        np.clip(nu[1] / self.Umax, -1.0, 1.0),  # sway velocity normalized
                        ], dtype=np.float32)

        # Determine target deltas for episode termination
        if circular_target:
            self.target_arc_length += v_circle * self.sampletime
            target_delta = self.target_arc_length
        else:
            target_delta = np.linalg.norm(np.array(self.simulator.moving_target) - np.array(self.initial_target))

        truncated = target_delta >= max_target_delta or self.current_step > self.sim_duration
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

        if not hasattr(self, 'last_distance'):
            self.last_distance = distance_to_target

        # Reward handling
        reward = 0
        terminated = truncated
        info = {}

        reward += 3 * (self.last_distance - distance_to_target)

        if circular_target:
            self.last_distance = distance_to_target
            reward -= 0.1 * abs(nu[5])  # yaw-rate penalty
        else:
            reward -= 0.1 * distance_to_target
            reward += 0.5 * np.cos(heading_error)
            reward -= 0.2 * abs(nu[1])  # sway-rate penalty
            reward -= 0.1 * abs(nu[5])  # yaw-rate penalty
            self.last_distance = distance_to_target

        if distance_to_target < self.simulator.surge_setpoint:
            reward += 1.0

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
            plotPosTar(simTime, simData, 1, targetData, headings=yawHistory)
            plotVehicleStates(simTime, simData)
            plotControls(simTime, simData, self.otter, 2)
            plotSpeed(simTime, simData, 5)
            plt.show()

    def reset(self, seed=None, options=None):

        if len(self.distanceHistory) > 0:
            IAE_distance = np.sum(np.abs(self.distanceHistory)) * self.sampletime
            IAE_heading = np.sum(np.abs(self.headingErrorHistory)) * self.sampletime

            # Store it in new attributes
            self.last_IAE_distance = IAE_distance
            self.last_IAE_heading = IAE_heading
        else:
            self.last_IAE_distance = 0.0
            self.last_IAE_heading = 0.0

        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        self.simulator.moving_target = list(moving_target_start)
        self.initial_target = list(self.simulator.moving_target)

        self.start_target_x = self.simulator.moving_target[0]
        self.current_step = 0
        self.target_arc_length = 0.0
        self.simData = []
        self.targetData = []
        self.simTime = []
        self.yawHistory = []
        self.distanceHistory = []
        self.headingErrorHistory = []

        # Target position
        target_pos = np.array(self.simulator.moving_target)
        target_x, target_y = target_pos

        # limit initial north distance and yaw angle to target after reset
        yaw = np.arctan2(self.simulator.moving_target_increase[1],
                         self.simulator.moving_target_increase[0])  # gets the correct initial angle

        x = target_x
        y = target_y + 10  # 10 meter east offset

        eta_initial = [x, y, 0, 0, 0, yaw]
        self.simulator.initial_state(eta_initial)  # sets initial state on the target location and angle

        distance_to_target, heading_error, nu = self.simulator.reset_simulation()

        if circular_target:
            normalized_distance = np.clip(distance_to_target / radius, -1.0, 1.0)
        else:
            normalized_distance = np.tanh(distance_to_target / max_target_delta)

        obs = np.array([
            normalized_distance,
            heading_error,
            np.clip(nu[0] / self.Umax, -1.0, 1.0),
            np.clip(nu[1] / self.Umax, -1.0, 1.0)
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

                if self.verbose and len(self.IAE_distance_history) % 100 == 0:
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

        checkpoint_model = os.path.join(SAVE_DIR, "ppo_otter_checkpoint.zip")
        checkpoint_vecnorm = os.path.join(SAVE_DIR, "ppo_otter_checkpoint_vecnormalize.pkl")
        final_model = os.path.join(SAVE_DIR, "ppo_otter_model.zip")
        final_vecnorm = os.path.join(SAVE_DIR, "vecnormalize.pkl")

        # If a previous checkpoint exists, resume from it, otherwise start fresh
        if os.path.exists(checkpoint_model) and os.path.exists(checkpoint_vecnorm):
            print("\nFound checkpoint. Loading model and VecNormalize to continue (remember to add options for inpt)\n")
            env = VecNormalize.load(checkpoint_vecnorm, env)
            model = PPO.load(checkpoint_model, env=env, device="cpu")
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
            # ctrl+c save and exit
            print("\nKeyboardInterrupt detected. Saving checkpoint before exiting...")
            model.save(checkpoint_model)
            if isinstance(env, VecNormalize):
                env.save(checkpoint_vecnorm)
            print("Checkpoint saved. Exiting.")
            sys.exit(0)

        # If training finishes normally, save a final model
        model.save("ppo_otter_model.zip")
        if isinstance(env, VecNormalize):
            env.save("vecnormalize.pkl")

    elif mode == 2:
        print("Loading previously saved model")
        eval_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
        eval_env = VecNormalize.load("vecnormalize.pkl", eval_env)

        eval_env.training = True  # set True if training
        eval_env.norm_reward = False

        model = PPO.load("circle_updated.zip", env=eval_env, device="cpu")

        if eval_env.training:
            IAE_callback = CallBackLog(verbose=1)
            model.learn(total_timesteps=training_timesteps, callback=IAE_callback)

            model.save("circle_updated.zip")
            eval_env.save("circle_normalize_updated.pkl")

    else:
        print("Chosen option is not valid.")

    if mode == 2:
        IAE_distance, IAE_heading = IAE_callback.return_log()
        print(f"Final IAE Distance: {IAE_distance[-1]}, Final IAE Heading: {IAE_heading[-1]}")
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
        eval_env = VecNormalize.load("circle_normalize_updated.pkl", eval_env)

    iae_distances = []
    iae_headings = []
    episode_actions = []
    episode_count = 0

    obs = eval_env.reset()
    for i in range(10000):

        env_single = eval_env.envs[0]
        last_target = env_single.simulator.moving_target.copy()
        action, _states = model.predict(obs)
        episode_actions.append(action)
        obs, rewards, dones, infos = eval_env.step(action)

        if dones[0]:

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

            if episode_count < 5:
                env_single.render()

            episode_count += 1
            episode_actions = []
            obs = eval_env.reset()

    print(f"Average eval IAE distance: {np.mean(iae_distances):.2f}")
    print(f"Average eval IAE heading:  {np.mean(iae_headings):.2f}")
