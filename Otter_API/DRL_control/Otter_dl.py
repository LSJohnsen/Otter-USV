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
from collections import deque
from torch import nn
from lib.Performance_metrics import PerformanceMetrics
from logs.IO import log_to_csv, log_params as io_log_params
import csv
import time

# Use cpu since bottleneck is simulation dynamics, not grapical
device = torch.device("cpu")
print(f"Using device: {device}")

simulator_environments = 8                                                                              # Number of simulation environments -> change depending on cpu capacity ~2-16
wave_function = False                                                                                   # adds a simple eastward wave function
use_target_coordinates = False                                                                          # To use coordinates as a target or to use a linear path
use_moving_target = True                                                                                # To use moving target instead of target list (path following)
target_list = [[0, 10000]]                                                                              # List of targets to use if use_target_coordinates is set to True
end_when_last_target_reached = True                                                                     # Ends the simulation when the final target is reached
moving_target_start = [0, -10]                                                                          # Start point of the moving target if use_moving_target is set to True
moving_target_increase = [-0.5, 0.0]                                                                    # Movement of the moving target each second                                                                                  # How many meters target should move each simulation before truncation
target_radius = 0.2                                                                                     # Radius from center of target that counts as target reached
verbose = True                                                                                          # Enable verbose printing
store_force_file = False                                                                                # Store the simulated control forces in a .csv file
circular_target = True                                                                                  # Make the moving target a circle in the simulation
animate_path = False
training_timesteps = 2000000                                                                            # Set timesteps (10mil-50mil+ depending on straight/circle)
log_results = False                                                                                     # log sim to csv, false when training

start_north = -20 #not used?                                                                            # Target north position from reference point
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

CHECKPOINT_MODEL = os.path.join(SAVE_DIR, "ppo_otter_checkpoint_station.zip")                                   # checkpoint model path
CHECKPOINT_VECNORM = os.path.join(SAVE_DIR, "ppo_otter_checkpoint_vecnormalize_station.pkl")
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

# used to log IAE over several training sessions
def append_iae_training_progress(csv_path: str, iae_callback):
    iae_dist, iae_head = iae_callback.return_log()
    n = min(len(iae_dist), len(iae_head))
    if n == 0:
        print(f"IAE CSV: No episodes logged yet; nothing to write to {csv_path}")
        return

    # last episode number in file
    start_episode = 1
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r") as f:
                rows = list(csv.reader(f))
                if len(rows) > 1: 
                    last_row = rows[-1]
                    start_episode = int(last_row[1]) + 1
        except Exception:
            start_episode = 1

    # append new
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)

        if not file_exists:
            w.writerow(["unix_time", "episode", "IAE_distance", "IAE_heading"])

        for i in range(n):
            w.writerow([
                int(time.time()),
                start_episode + i,
                float(iae_dist[i]),
                float(iae_head[i])
            ])

    print(f"IAE CSV: Appended {n} episodes to {csv_path} (starting from {start_episode})")

class OtterEnv(gym.Env):
    def __init__(self, simulator, otter):
        super().__init__()                                                                               # call gym.env constructor

        self.simulator = simulator
        self.otter = otter
        self.path_modes = ["circle", "line", "stationary"] 
        self.path_mode = "circle"

        # Overwrite when training one at a time. station->line->all three
        self.path_modes = ["stationary"] 
        self.path_mode = "stationary" 

        self.sampletime = 0.1  # iteration updates
        self.episode_duration = 400000  # no. simulation samples (truncates at distances, just ensure not too small)
        self.sim_duration = int(self.episode_duration / self.sampletime)  # sim duration
        self.current_step = 0
        self.target_arc_length = 0
        self.prev_tau_X = 0

        # callback function for finished learning
        self.hold_radius = 0.2          # meters
        self.hold_time_required = 10.0  # seconds
        self.hold_time = 0.0            # accumulate within radius
        self.last_hold_success = False  # stored at end of episode

        #used for observation/action
        self.Umax = 6 * 0.5144
        self.Rmax = 2                   # Just a chosen relative value for normalization 2rad/s
        self.max_force = 150            
        self.last_distance = 0

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

        # normalized values
        self.observation_space = Box(low=np.array([-1,                   # distance to target
                                                   -1,                   # heading error
                                                   -1,                   # surge velocity
                                                   -1,                   # sway velocity
                                                   -1,                   # yaw rate
                                                   -1],                  # target error rate of change (see third order traj ref) (d_dot) 
                                                  dtype=np.float32),
                                     high=np.array([1,
                                                    1,
                                                    1,
                                                    1,
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
        prev_distance = self.last_distance

        # Chooses action which are passed to the simulator at current sampletime
        tau_X, tau_N = action
        eta, nu, target, distance_to_target, heading_error, u_actual = self.simulator.simulate_step(
            self.sampletime,
            self.otter,
            tau_X,
            tau_N
        )   

        #closing speed
        d_dot = (prev_distance - distance_to_target) / self.sampletime

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
        
        # update for training end if enough episodes
        if distance_to_target <= self.hold_radius:
            self.hold_time += self.sampletime
        else:
            self.hold_time = 0.


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


        obs = np.array([
            normalized_distance,                         # distance normalized
            heading_error / np.pi,                       # heading normalized
            np.clip(nu[0] / self.Umax, -1.0, 1.0),       # surge velocity normalized
            np.clip(nu[1] / self.Umax, -1.0, 1.0),       # sway velocity normalized
            np.clip(nu[5] / self.Rmax, -1.0, 1.0),       # yaw rate normalized
            np.clip(d_dot / self.Umax, -1.0, 1.0)        # closing speed normalized
        ], dtype=np.float32)

        # Determine target deltas for episode termination 
        episode_time = self.current_step * self.sampletime

        ''' 
        currently reset at start of steps, can be removed or use later? useless for time based ml?
        '''

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


        '''
        Reward handling 
            d_dot = (prev_distance - distance_to_target) / self.sampletime  
            u = nu[0]
            v = nu[1]
            r = nu[5]
            e = heading_error  # already wrapped
            d = distance_to_target
        '''

        reward = 0.0
        terminated = truncated
        info = {}

        d, u, v, r = distance_to_target, nu[0], nu[1], nu[5] 
        # d_dot = (prev_distance - d) / self.sampletime

        # change based on actual UOWC system performance 
        d_opt = 0.1
        d_acc = 1.0

        in_range = np.clip((d_acc - d) / d_acc, 0.0, 1.0)   # 1 when d=0, 0 when d>=d_acc
        outside_range = 1.0 - in_range                      # scaling by range
        w_far = outside_range**2                            # weightstrong when far
        w_close = in_range**2                               # weight strong when close

        # Approach shaping when outside closing range
        reward += 1.0 * outside_range * (prev_distance - d)

        # Prefer being inside acceptable range
        reward += 0.5 * in_range

        # Prefer optimal distance - penalize deviation from d_opt
        reward -= 2.0 * ((d - d_opt) / d_acc)**2

        # Prevent overshoot near target 
        reward -= 0.6 * w_close * abs(d_dot)

        # slow and stable when close
        reward -= 0.2  * w_close * abs(u)
        reward -= 0.05 * w_close * abs(v)   # keep sway penalty smaller if circular tracking matters
        reward -= 0.15 * w_close * abs(r)

        # weak heading guidance when far away
        reward += 0.05 * outside_range * np.cos(heading_error)

        # Continuous hold reward when in range
        t_short = self.hold_time_required / 5.0
        t_long  = self.hold_time_required

        hold_ratio_short = np.clip((self.hold_time - t_short) / max(t_short, 1e-6), 0.0, 1.0)
        hold_ratio_long  = np.clip(self.hold_time / max(t_long,  1e-6), 0.0, 1.0)

        reward += in_range * (0.2 * hold_ratio_short + 0.8 * hold_ratio_long)

        reward *= 0.01 #reduce size for smaller diff, remove if turning on normalization
            
        self.last_distance = float(distance_to_target)
        

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        # pick the “live” data if it exists, otherwise the last-episode
        simData_list = self.simData if len(self.simData) > 1 else self.last_sim_data
        targetData_list = self.targetData if len(self.targetData) > 1 else self.last_target_data
        simTime_list = self.simTime if len(self.simTime) > 1 else self.last_sim_time
        yaw_list = self.yawHistory if len(self.yawHistory) > 1 else self.last_yaw_history

        # convert
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

        # check if hold_time reaches required 10s
        self.last_hold_success = (self.hold_time >= self.hold_time_required)

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
                r_min, r_max = -15.0, 15.0          

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
        self.last_distance = distance_to_target

        # clear histories 
        self.simData = []
        self.targetData = []
        self.simTime = []
        self.yawHistory = []
        self.distanceHistory = []
        self.headingErrorHistory = []
        # reset hold_time
        self.hold_time = 0.0

        eta0 = self.simulator.eta.copy()
        nu0  = self.simulator.nu.copy()
        u0   = self.simulator.u_actual.copy()
        commands0 = np.array([0.0, 0.0])   # no control before first step
        d_dot = 0

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
            normalized_distance,                         # distance normalized
            heading_error / np.pi,                       # heading normalized
            np.clip(nu[0] / self.Umax, -1.0, 1.0),       # surge velocity normalized
            np.clip(nu[1] / self.Umax, -1.0, 1.0),       # sway velocity normalized
            np.clip(nu[5] / self.Rmax, -1.0, 1.0),       # yaw rate normalized
            np.clip(d_dot / self.Umax, -1.0, 1.0)        # closing speed normalized
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

# stop if maintaining objective (above target for 10s straight) over threshold% of window_size-episodes
class StopOnSuccessRate(BaseCallback):
    def __init__(self, window_size=200, threshold=0.99, verbose=1):
        super().__init__(verbose)
        self.window_size = window_size
        self.threshold = threshold
        self.history = deque(maxlen=window_size)
        self.completed = False

    def _on_step(self) -> bool:

        for i, done in enumerate(self.locals["dones"]):
            if done:
                # maintained objective check
                success = self.training_env.get_attr("last_hold_success", i)[0]
                self.history.append(1 if success else 0)

                # dont eval before 200ep completed
                if len(self.history) == self.window_size:
                    rate = sum(self.history) / self.window_size

                    if self.verbose:
                        print(f"[StopCheck] success rate last {self.window_size}: {rate:.3f}")

                    if rate >= self.threshold:
                        self.completed = True
                        print("Success-rate criterion met. Stopping training.")
                        return False  # stops learn()

        return True
    
if __name__ == "__main__":

    print(f"Using device: {device}")

    n_envs = 8

    print("\n\n\n\n##################################################################")
    print("\n ENSURE CORRECT PATHS FOR CHECKPOINTS TO PREVENT OVERRIDING FIELES\n")
    print("##################################################################")
    mode = int(input("\n\nChoose '1' for model training or '2' to run saved model: "))
    if mode == 1:

        env = SubprocVecEnv([make_env() for _ in range(n_envs)])

        # If a previous checkpoint exists, resume from it, otherwise start fresh
        if os.path.exists(CHECKPOINT_MODEL) and os.path.exists(CHECKPOINT_VECNORM):
            print("\nFound checkpoint. Loading model and VecNormalize to continue\n")
            env = VecNormalize.load(CHECKPOINT_VECNORM, env)
            model = PPO.load(CHECKPOINT_MODEL, env=env, device="cpu")
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0) # changed to not normalized rewards (better with current design?)
            ''' original from previous, new uses relu, slower learning etc.
            model = PPO(
                MlpPolicy,
                env,
                verbose=1,
                device="cpu",
                normalize_advantage=True,
                gae_lambda=0.98,
                learning_rate=0.0002,
                clip_range=0.2,
                n_steps=4096,
                ent_coef=0.01,
                target_kl=None,
                policy_kwargs=dict(activation_fn=nn.Tanh),
            )
            '''
            model = PPO(
                "MlpPolicy",
                env,
                
                # stability settings    
                learning_rate=0.0004,   # 1e-4 stable but prob too slow
                clip_range=0.2,       # 0.2 default 
                target_kl=0.02,       # prevent large policy changes

                # GAE & rollout settings
                gae_lambda=0.98,      # advantage estimation 
                n_steps=4096,         # larger>more stable
                batch_size=512,       # Good balance for large rollouts
                n_epochs=10,          # Standard PPO setting


                # Exploration control
                ent_coef=0.005,       # larger more = exploration less stable
                vf_coef=0.1,        # critic doesn't drown actor=

                normalize_advantage=True,


                # Network architecture
                policy_kwargs=dict(
                    activation_fn=nn.ReLU,  # standard, fastest (action is standard tanh still)
                    net_arch=dict(
                        pi=[64, 64],        # actor
                        vf=[128, 128]       # critic
                    )
                ),

                verbose=1,
                device="cpu"  
            )

        print("\nTraining model: \n")
        IAE_callback = CallBackLog(verbose=1)
        stop_callback = StopOnSuccessRate(window_size=200, threshold=0.99, verbose=1) # threshold = % of window size required to be complete, modify for relevance
        interrupted = False

        try:
            model.learn(
                total_timesteps=training_timesteps,
                callback=[IAE_callback, stop_callback],
            )
        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted. Will save checkpoint.")
            append_iae_training_progress(os.path.join(SAVE_DIR, "iae_training_progress.csv"), IAE_callback)
        finally:
            # always save checkpoint
            print("\nCompleted total timesteps: saving checkpoint")
            model.save(CHECKPOINT_MODEL)
            if isinstance(env, VecNormalize):
                env.save(CHECKPOINT_VECNORM)


            # save to final if goal reached
            if stop_callback.completed and not interrupted:
                print("\n Goal reached, saving final model")
                model.save(FINAL_MODEL)
                if isinstance(env, VecNormalize):
                    env.save(FINAL_VECNORM)

        if interrupted:
            sys.exit(0)


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

    

try:
    if "IAE_callback" in globals() and mode == 1:
        append_iae_training_progress(os.path.join(SAVE_DIR, "iae_training_progress.csv"), IAE_callback)
except Exception as e:
    print(f"IAE CSV: Failed to write IAE progress: {e}")