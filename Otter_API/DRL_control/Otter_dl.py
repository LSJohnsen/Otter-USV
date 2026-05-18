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
from stable_baselines3.common.vec_env import VecMonitor 
from stable_baselines3.common.vec_env import VecNormalize
from collections import deque
from torch import nn
from lib.Performance_metrics import PerformanceMetrics
from logs.IO import log_to_csv, log_params as io_log_params
import csv
import time
from DRL_control.reward_callback_plot import append_reward_training_progress


# generally use cpu since bottleneck is simulation dynamics, not grapical
device = torch.device("cpu")
print(f"Using device: {device}")

simulator_environments = 8                                                                              # Number of simulation environments -> change depending on cpu capacity ~2-16
wave_function = False                                                                                   # adds a simple eastward wave function
use_target_coordinates = False                                                                          # To use coordinates as a target or to use a linear path
use_moving_target = False                                                                               # To use moving target instead of target list (path following)
target_list = [[0, 10000]]                                                                              # List of targets to use if use_target_coordinates is set to True
end_when_last_target_reached = True                                                                     # Ends the simulation when the final target is reached
moving_target_start = [0, 0]                                                                            # Start point of the moving target if use_moving_target is set to True
moving_target_increase = [-0.5, 0.0]                                                                    # Movement of the moving target each second                                                                                  # How many meters target should move each simulation before truncation
target_radius = 0.2                                                                                     # Radius from center of target that counts as target reached
verbose = True                                                                                          # Enable verbose printing
store_force_file = False                                                                                # Store the simulated control forces in a .csv file
circular_target = True                                                                                  # Make the moving target a circle in the simulation
animate_path = False
training_timesteps = 100000000                                                                          # Set timesteps (or just change success criteria)
log_results = False                                                                                     # log sim to csv, false when training

# Finished-controller experiment settings
EXPERIMENT_LOG_DIR = os.path.join(CURRENT_DIR, "experiment_logs")
# Convention: positions/velocities are [North, East], yaw=0 points North.
# The USV starts at [0, 0] NE pointing North for both cases.
FINISHED_CONTROLLER_EXPERIMENTS = [
    {
        "name": "DRL_stationary_target_20N_10E",
        "path_mode": "stationary",
        "target_start": [20.0, 10.0],
        "target_velocity": [0.0, 0.0],
        "initial_eta": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "max_time": 150.0,
    },
    {
        "name": "DRL_straight_target_30N_0E_vE_1p5",
        "path_mode": "line",
        "target_start": [30.0, 0.0],
        "target_velocity": [0.0, 1.5],
        "initial_eta": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "max_time": 100.0,
    },
]

randomize_position = True                                                                               # Used to randomize usv start position for better training
randomize_path = True                                                                                   # Randomizes paths to circular/straight line/stationary
v_north = 0                                                                                             # Moving target speed north (m/s)
v_east = -1.5                                                                                           # Moving target speed east (m/s)
radius = 40 # SIM LOGIC LINE~500 X/Y START (FIX)                                                        # If not tracking a circular motion 
max_target_delta = 250                                                                                  # Max distance target moves before truncation
max_episode_time = 100.0
v_circle = 1.5                                                                                          # Angular velocity (m/s)
side_length = 50                                                                                        # Square tracking side length
side_target_speed = 1                                                                                   # Speed of square target


USE_RELATIVE_VELOCITY_OBS = True                                                                       # True  = new 10-observation setup with x_rel_dot and y_rel_dot

'''
Probability of paths: [stationary, straight line, circular, zigzag, waypoints]
waypoints is a stationary target that updates with a new location once target is reached
'''
path_probabilities = [0.0, 0.0, 0.0, 1.0, 0.0]                                                                  

# Disturbances (randomized directions and amplitudes, see simulator)
wave_disturbance = True
wind_disturbance = True

# Stationary target start-distance randomization [m]
stationary_start_min_dist = 10.0     # closest initial distance 
stationary_start_max_dist = 30.0     # farthest initial distance 

numDataPoints = 830                                                                                     # number of 3D data points
FPS = 60                                                                                                # frames per second (animated GIF)
filename = '3D_animation.gif'                                                                           # data file for animated GIF
browser = 'chrome'

if USE_RELATIVE_VELOCITY_OBS:
    CHECKPOINT_MODEL = os.path.join(SAVE_DIR, "otter_10obs_station_dist.zip")
    CHECKPOINT_VECNORM = os.path.join(SAVE_DIR, "otter_10obs_station_dist.pkl")
    FINAL_MODEL = os.path.join(SAVE_DIR, "ppo_otter_10obs_final.zip")
    FINAL_VECNORM = os.path.join(SAVE_DIR, "ppo_otter_10obs_final_vecnorm.pkl")
else:
    CHECKPOINT_MODEL = os.path.join(SAVE_DIR, "ppo_otter_straight_dist.zip")
    CHECKPOINT_VECNORM = os.path.join(SAVE_DIR, "ppo_otte_straight_dist_vecnorm.pkl")
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
                                            radius,
                                            use_waves=wave_disturbance,
                                            use_wind=wind_disturbance)


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

def wrap_to_pi(angle): 
    return (angle + np.pi) % (2 * np.pi) - np.pi

class OtterEnv(gym.Env):
    def __init__(self, simulator, otter, experiment_config=None):
        super().__init__()                                                                               # call gym.env constructor

        self.simulator = simulator
        self.otter = otter
        self.experiment_config = experiment_config


        # Overwrite when training one at a time. station->line->all three
        self.path_modes = ["stationary", "line", "circle", "zigzag", "waypoint"] # line, stationary, circle


        self.sampletime = 0.2  # iteration updates
        self.episode_duration = 400000  # no. simulation samples (truncates at distances, just ensure not too small)
        self.sim_duration = int(self.episode_duration / self.sampletime)  # sim duration
        self.current_step = 0
        self.target_arc_length = 0
        self.prev_tau_X = 0
        self.prev_action = np.zeros(2, dtype=float)

        # callback function for finished learning
        self.hold_radius = 0.2          # meters
        self.hold_time_required = 30.0  # seconds
        self.hold_time = 0.0            # accumulate within radius
        self.max_hold_time = 0.0        # longest period
        self.last_hold_success = False  # stored at end of episode
        self.objective_achieved = False
        self.min_distance_to_target = np.inf

        self.last_max_hold_time = 0.0
        self.last_objective_achieved = False
        self.last_min_distance_to_target = np.inf

        # used for observation/action/rewards
        self.Umax = 6 * 0.5144
        self.Rmax = 2                   # Just a chosen relative value for normalization 2rad/s
        self.tauX_max = 150
        self.tauN_max = 110         
        self.max_rad = 0.0
        self.u_applied = np.zeros(2)
        self.tau_act = 1.0     
        self.last_distance = 0
        self.last_alloc_scale = 1.0
        
        # target references for heading rewards
        self.prev_target_pos = None         
        self.target_heading_ref = 0.0
        self.stationary_heading_ref = 0.0

        # control action reward shaping
        self.prev_cmd = np.zeros(2, dtype=float)
        self.prev_applied = np.zeros(2, dtype=float)


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

        self.fixed_experiment_mode = False
        self.experiment_max_time = max_episode_time
        
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

        # with relative x,y,r
        # Observation space
        if USE_RELATIVE_VELOCITY_OBS:
            self.observation_space = Box(
                low=np.array([
                    -1,  # x_rel
                    -1,  # y_rel
                    -1,  # yaw_rel
                    -1,  # surge vel
                    -1,  # sway vel
                    -1,  # yaw vel
                    -1,  # euclidean dist
                    -1,  # euclidean dist rate
                    -1,  # x_rel_dot
                    -1,  # y_rel_dot
                ], dtype=np.float32),
                high=np.array([
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
                ], dtype=np.float32)
            )
        else:
            self.observation_space = Box(
                low=np.array([
                    -1,  # x_rel
                    -1,  # y_rel
                    -1,  # yaw_rel
                    -1,  # surge vel
                    -1,  # sway vel
                    -1,  # yaw vel
                    -1,  # euclidean dist
                    -1,  # euclidean dist rate
                ], dtype=np.float32),
                high=np.array([
                    1, 1, 1, 1, 1, 1, 1, 1
                ], dtype=np.float32)
            )

        # min/max forces in surge/yaw normalized
        if use_moving_target:
            self.action_space = Box(
                low=np.array([0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32)
                )
        else:
            self.action_space = Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32)
                )
            
    def step(self, action):
        self.current_step += 1
        truncated_count = 0
        prev_distance = self.last_distance                                      # previous Euclidean distance for d_dot

        # PPO normalized action [-1, 1]
        action = np.asarray(action, dtype=float)
        action = np.clip(action, -1.0, 1.0)

        # Desired normalized surge/yaw command
        surge_des = float(action[0])
        yaw_des   = float(action[1])

        # Match live PMARMAN architecture:
        # action[0] -> X command
        # action[1] -> N command
        tau_cmd = np.array([
            surge_des * self.tauX_max,
            yaw_des   * self.tauN_max
        ], dtype=float)

        alloc_scale = 1.0

        # Actuator lag on generalized command
        alpha_t = self.sampletime / (self.tau_act + self.sampletime)
        self.u_applied = (1.0 - alpha_t) * self.u_applied + alpha_t * tau_cmd

        tau_X, tau_N = self.u_applied

        # Store allocation scale for reward/debug
        self.last_alloc_scale = alloc_scale

        # simulate with applied input
        eta, nu, target, distance_to_target, heading_error, u_actual = self.simulator.simulate_step(
            self.sampletime,
            self.otter,
            tau_X,
            tau_N
        )                                                                       # one simulation step

        self.metrics.update(
            distance_to_target=distance_to_target,
            heading_error=heading_error,
            u1=u_actual[0],
            u2=u_actual[1],
            dt=self.sampletime,
        )                                                                       # update evaluation metrics

        # for state/logging
        # Convert DRL generalized command to left/right shaft speed
        n1_cmd, n2_cmd = self.otter.otter_control.controlAllocation(tau_X, tau_N)

        # Store the same format as PID/NMPC:
        # commands = desired shaft speeds [rad/s]
        # actuals  = actual shaft speeds [rad/s]
        commands = np.array([n1_cmd, n2_cmd], dtype=float)
        actuals = u_actual
        full_state = np.hstack([eta, nu, commands, actuals])

        # target state
        target_pos = np.array(self.simulator.moving_target, dtype=float)        # target position in world frame
        target_delta = target_pos - self.prev_target_pos                        # target position change
        target_speed = np.linalg.norm(target_delta) / self.sampletime           # target speed magnitude

        # update target heading if target actually moved
        if target_speed > 1e-4:
            self.target_heading_ref = float(np.arctan2(target_delta[1], target_delta[0]))  # target motion heading

        self.prev_target_pos = target_pos.copy()                                # store target position for next step

        # USV position in world frame
        usv_pos = np.array([eta[0], eta[1]], dtype=float)                       # vessel x,y position

        # relative position from USV to target
        rel_pos = target_pos - usv_pos                                          # vector from USV to target
        x_rel = rel_pos[0]                                                      # relative x
        y_rel = rel_pos[1]                                                      # relative y

        # Euclidean distance
        e_d = np.linalg.norm(rel_pos)                                           # distance derived from x_rel, y_rel

        # LOS heading from USV to target
        psi_los = float(np.arctan2(y_rel, x_rel))                               # angle from USV to target

        # relative yaw error
        yaw_rel = wrap_to_pi(psi_los - eta[5])                                  # yaw error

        # distance rate
        d_dot = (e_d - prev_distance) / self.sampletime                         # positive = moving away, negative = closing

        # normalize new observation variables
        pos_scale = max_target_delta                                            # scale for x_rel, y_rel, and e_d
        x_rel_norm = np.clip(x_rel / pos_scale, -1.0, 1.0)                      # normalized relative x
        y_rel_norm = np.clip(y_rel / pos_scale, -1.0, 1.0)                      # normalized relative y
        e_d_norm = np.clip(e_d / pos_scale, -1.0, 1.0)                          # normalized Euclidean distance
        d_dot_norm = np.clip(d_dot / self.Umax, -1.0, 1.0)                      # normalized distance rate

        # Relative x/y rate for optional 10-observation setup
        x_rel_dot = (x_rel - self.prev_x_rel) / self.sampletime
        y_rel_dot = (y_rel - self.prev_y_rel) / self.sampletime

        x_rel_dot_norm = np.clip(x_rel_dot / self.Umax, -1.0, 1.0)
        y_rel_dot_norm = np.clip(y_rel_dot / self.Umax, -1.0, 1.0)

        #hold radius depending on path mode
        if self.path_mode == "stationary":
            success_radius = self.hold_radius
        else:
            success_radius = 0.5

        #  hold logic
        if distance_to_target <= success_radius:
            self.hold_time += self.sampletime                                           # accumulate hold time
        else:
            self.hold_time = 0.0                                                        # reset hold time if outside success radius

        self.max_hold_time = max(self.max_hold_time, self.hold_time)                    # log max hold time

        # success condition
        success = self.hold_time >= self.hold_time_required

        self.min_distance_to_target = min(
            self.min_distance_to_target,
            float(distance_to_target)
        )

        if success:
            self.objective_achieved = True

        # logging
        self.simData.append(full_state)
        self.targetData.append(target_pos.copy())
        self.simTime.append(self.current_step * self.sampletime)
        self.yawHistory.append(eta[5])
        self.distanceHistory.append(distance_to_target)
        self.headingErrorHistory.append(heading_error)

        self.cum_distance += distance_to_target * self.sampletime
        if (not self.reached_flag) and (distance_to_target < self.simulator.surge_setpoint):
            self.reached_flag = True
            self.reached_target_time = self.current_step * self.sampletime

        # updated observation - later versions should have target vx/vy for more predictive control
        if USE_RELATIVE_VELOCITY_OBS:
            obs = np.array([
                x_rel_norm,                                                         # relative x-position
                y_rel_norm,                                                         # relative y-position
                yaw_rel / np.pi,                                                    # normalized yaw error
                np.clip(nu[0] / self.Umax, -1.0, 1.0),                              # surge velocity
                np.clip(nu[1] / self.Umax, -1.0, 1.0),                              # sway velocity
                np.clip(nu[5] / self.Rmax, -1.0, 1.0),                              # yaw rate
                e_d_norm,                                                           # Euclidean distance
                d_dot_norm,                                                         # Euclidean distance rate
                x_rel_dot_norm,                                                     # relative x-rate
                y_rel_dot_norm,                                                     # relative y-rate
            ], dtype=np.float32)
        else:
            obs = np.array([
                x_rel_norm,                                                         # relative x-position
                y_rel_norm,                                                         # relative y-position
                yaw_rel / np.pi,                                                    # normalized yaw error
                np.clip(nu[0] / self.Umax, -1.0, 1.0),                              # surge velocity
                np.clip(nu[1] / self.Umax, -1.0, 1.0),                              # sway velocity
                np.clip(nu[5] / self.Rmax, -1.0, 1.0),                              # yaw rate
                e_d_norm,                                                           # Euclidean distance
                d_dot_norm,                                                         # Euclidean distance rate
            ], dtype=np.float32)

        # random noise for bad sensor data 
        obs[0] += np.random.normal(0, 0.002)  # x_rel norm
        obs[1] += np.random.normal(0, 0.002)  # y_rel norm
        obs[2] += np.random.normal(0, 0.005)  # yaw_rel norm
        obs[3] += np.random.normal(0, 0.02)   # u norm
        obs[4] += np.random.normal(0, 0.02)   # v norm
        obs[5] += np.random.normal(0, 0.01)   # r norm
        obs = np.clip(obs, -1.0, 1.0)

        
        self.prev_x_rel = float(x_rel)
        self.prev_y_rel = float(y_rel)
        ''' 
        currently reset at start of steps, can be removed or use later? useless for time based ml?
        '''
        
                
        episode_time = self.current_step * self.sampletime

        if self.fixed_experiment_mode:
            truncated = episode_time >= self.experiment_max_time
            terminated = False
        else:
            truncated = episode_time >= max_episode_time
            terminated = success

        if self.fixed_experiment_mode:
            info = {"is_success": bool(self.objective_achieved)}
        else:
            info = {"is_success": bool(success)}

        # redundant atm
        truncated_count += 1                                                                
        if truncated_count % 100 == 0:
            print(f"[{self.current_step}] Target at {self.simulator.moving_target}, "
                  f"Initial at {self.initial_target}, "
                  f"Δ={np.linalg.norm(np.array(self.simulator.moving_target) - np.array(self.initial_target)):.2f}")
        ###
        if terminated or truncated:
            self.last_sim_data = self.simData.copy()
            self.last_target_data = self.targetData.copy()
            self.last_sim_time = self.simTime.copy()
            self.last_yaw_history = self.yawHistory.copy()

            if self.current_step > 0:
                self.last_avg_distance = self.cum_distance / self.current_step
            else:
                self.last_avg_distance = 0.0

            self.last_reached_target_time = self.reached_target_time

            # Update final metrics before logging
            self.last_IAE_distance, self.last_IAE_heading = self.metrics.get_IAE()
            self.last_ISU = self.metrics.get_ISU()
            self.last_ISU_normalized = self.metrics.get_ISU_normalized()
            self.last_IAU = self.metrics.get_IAU()

            self.last_max_hold_time = float(self.max_hold_time)
            self.last_objective_achieved = bool(self.objective_achieved)
            self.last_min_distance_to_target = float(self.min_distance_to_target)

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

        #save for rendering before reset 
        if success or truncated:
            self.last_simData = np.array(self.simData, dtype=float)
            self.last_targetData = np.array(self.targetData, dtype=float)
            self.last_simTime = np.array(self.simTime, dtype=float)                     
                
        # states
        d, u, v, r = distance_to_target, nu[0], nu[1], nu[5]
        psi = eta[5]

        # tuning parameters
        sigma_p = 1.5                                                       # width of position reward
        C_p = 1.0                                                           # amplitude of position reward

        sigma_psi = 0.6                                                     # width of heading reward  
        C_psi = 0.5                                                         # amplitude of heading reward

        sigma_u = 0.5                                                       # width of surge reward
        C_u = 0.5                                                           # amplitude surge reward

        sigma_v = 0.5                                                       # width of velocity reward
        C_v = 0.5                                                           # amplitude of velocity reward

        K_d = 3.0                                                           # slope of tanh transition distance rate
        C_d_dot = 1.0                                                       # amplitude of distance-rate reward

        d_acc = 1.0                                                         # acceptable distance to target

        C_r = 1.2                                                           # yaw rate penalty scale when close to target and correct orientation

        # distance derivative - d_dot
        d_dot = (d - self.last_distance) / self.sampletime  

        # USV velocity in world frame
        vx_usv = u * np.cos(psi) - v * np.sin(psi)
        vy_usv = u * np.sin(psi) + v * np.cos(psi)

        
        # target velocity in world frame
        if self.simulator.use_moving_target:
            vx_t = target_speed * np.cos(self.target_heading_ref)
            vy_t = target_speed * np.sin(self.target_heading_ref)
        else:
            vx_t = 0.0
            vy_t = 0.0

        # gaus pos reward
        d_opt = 0.1                                                         # desired distance to target / hover distance
        r_pos = C_p * np.exp(-((d - d_opt) ** 2) / (2 * sigma_p ** 2))      # reward for being near desired distance
        
        # what's considered within acceptable range 
        in_range = np.clip((d_acc - d) / d_acc, 0.0, 1.0)

        # distance rate reward
        r_d_dot = -C_d_dot * np.tanh(K_d * d_dot)            # reward/penalty for closing/moving away 

        # gaus heading reward

        e_track = wrap_to_pi(self.target_heading_ref - psi)                                  # target trajectory heading error
        heading_scale = np.clip(1.0 - abs(e_track) / np.pi, 0.0, 1.0)                        # not penalize yaw rate if heading error is large

        if self.simulator.use_moving_target:
            psi_los = float(np.arctan2(target_pos[1] - eta[1], target_pos[0] - eta[0]))      # LOS heading to target
            e_los = wrap_to_pi(psi_los - psi)                                                # LOS error
            
            

            r_heading = (
                C_psi * (1.0 - in_range) * np.exp(-(e_los ** 2) / (2 * sigma_psi ** 2)) +    # LOS alignment far away
                C_psi * in_range * np.exp(-(e_track ** 2) / (2 * sigma_psi ** 2))            # trajectory alignment close in
            )
           
        else:
            e_hold = wrap_to_pi(self.stationary_heading_ref - psi)                           # fixed heading reference for stationary hold
            r_heading = C_psi * np.exp(-(e_hold ** 2) / (2 * sigma_psi ** 2))                # reward for maintaining heading
        
  
        r_heading2 = -in_range * heading_scale * C_r * abs(r)                                # penalize spinning close to target

        # surge reward
        u_far = 1.0                                     # surge speed when far away

        if self.simulator.use_moving_target:
            u_close = target_speed                      # match target speed when close
        else:
            u_close = 0.0                                # stationary target -> stop near target

        u_d = u_far if d > sigma_p else u_close          # desired surge speed depending on distance

        alpha_u = 0.1                                                                           # offset for small negative reward far from desired speed
        r_surge = (C_u + alpha_u) * np.exp(-((u - u_d) ** 2) / (2 * sigma_u ** 2)) - alpha_u    # surge-speed reward

        
        # relative velocity reward
        e_vx = vx_usv - vx_t
        e_vy = vy_usv - vy_t
        e_v = np.sqrt(e_vx**2 + e_vy**2)

        r_vel = in_range * C_v * np.exp(-(e_v**2) / (2 * sigma_v**2))       # reward for maintaining same total velocity as target in world when close
        
        # time penalty
        C_t = 0.01                                                         # small constant penalty per step
        r_time = C_t                                                        # faster convergence

        # action penalty
        scale = np.array([self.tauX_max, self.tauN_max], dtype=float)       # normalization for actuator commands
        delta_cmd = (tau_cmd - self.prev_cmd) / scale                       # normalized command change
        C_a_x = 0.5      # surge command smoothing
        C_a_n = 5.0      # yaw command smoothing

        r_action = (
            C_a_x * delta_cmd[0]**2 +
            C_a_n * delta_cmd[1]**2
)



        self.prev_cmd[:] = tau_cmd                                          # update curr cmd          

        # Continuous hold reward when in range
        t_short = self.hold_time_required / 5.0                             # small reward for holding 2 seconds
        t_long  = self.hold_time_required                                      

        # increasing reward for staying on targe
        hold_ratio_short = np.clip(self.hold_time / max(t_short, 1e-6), 0.0, 1.0)**2   # increasing reward for staying on targe - testing scaling to prioritize holding longer
        hold_ratio_long  = np.clip(self.hold_time / max(t_long,  1e-6), 0.0, 1.0)      # reward for completing long hold

        r_hold = 5 * in_range * (0.2 * hold_ratio_short + 0.8 * hold_ratio_long)                    # hold time reward


        # final
        reward += r_pos                                                         # reward for correct distance
        reward += r_d_dot                                                       # reward for decreasing distance
        reward += r_heading                                                     # reward for correct heading
        reward += r_heading2                                                    # pure yaw rate penalty close to target
        reward += r_surge                                                       # reward for correct surge speed
        reward += r_vel                                                         # penalty for not keeping same velocity as target
        reward -= r_time                                                        # penalty for slow convergence
        reward -= r_action                                                      # penalty for aggressive actuation   
        reward += r_hold                                                        # reward for hovering
        if success:
            reward += 100.0                                                     # bonus for completing the hold objective

        
        self.episode_reward_breakdown["r_pos"] += float(r_pos)
        self.episode_reward_breakdown["r_d_dot"] += float(r_d_dot)
        self.episode_reward_breakdown["r_heading"] += float(r_heading)
        self.episode_reward_breakdown["r_heading2"] += float(r_heading2)
        self.episode_reward_breakdown["r_surge"] += float(r_surge)
        self.episode_reward_breakdown["r_vel"] += float(r_vel)
        self.episode_reward_breakdown["r_time"] += float(r_time)
        self.episode_reward_breakdown["r_action"] += float(r_action)
        self.episode_reward_breakdown["r_hold"] += float(r_hold)

        if success:
            self.episode_reward_breakdown["success_bonus"] += 100.0

        reward *= 0.1
        self.episode_reward_breakdown["total_reward"] += float(reward)


        if terminated or truncated:
            if self.fixed_experiment_mode:
                info["is_success"] = bool(self.objective_achieved)
            else:
                info["is_success"] = bool(success)

            info["max_hold_time"] = float(self.max_hold_time)
            info["min_distance"] = float(self.min_distance_to_target)
            info["objective_achieved"] = bool(self.objective_achieved)

            info["IAE_distance"] = self.last_IAE_distance
            info["IAE_heading"] = self.last_IAE_heading
            info["avg_distance"] = self.last_avg_distance
            info["intercept_time"] = self.last_reached_target_time
            info["r0"] = self.r0
            info["alpha0"] = self.alpha0
            info["reward_breakdown"] = self.episode_reward_breakdown.copy()

        self.last_distance = float(e_d)

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        # Prefer current live data if it has enough samples
        # otherwise fall back to the last completed episode
        simData_list = self.simData if self.simData is not None and len(self.simData) > 1 else getattr(self, "last_simData", None)
        targetData_list = self.targetData if self.targetData is not None and len(self.targetData) > 1 else getattr(self, "last_targetData", None)
        simTime_list = self.simTime if self.simTime is not None and len(self.simTime) > 1 else getattr(self, "last_simTime", None)

        if simData_list is None or simTime_list is None:
            print("render skipped: no data available")
            return

        simData = np.asarray(simData_list)
        targetData = None if targetData_list is None else np.asarray(targetData_list)
        simTime = np.asarray(simTime_list)

        if simData.ndim < 2 or simData.shape[0] <= 1:
            print(f"render skipped: invalid simData shape {simData.shape}")
            return

        if simTime.ndim != 1 or simTime.shape[0] != simData.shape[0]:
            print(f"render skipped: simTime shape {simTime.shape} does not match simData {simData.shape}")
            return

        if targetData is not None:
            if targetData.ndim < 2 or targetData.shape[0] != simData.shape[0]:
                print(f"render skipped: targetData shape {targetData.shape} does not match simData {simData.shape}")
                return

        plotPosTar2(simTime, simData, 1, targetData)
        plotVehicleStates(simTime, simData, 2)
        plotControls(simTime, simData, self.otter, 3)
        plotSurge(simTime, simData, 6)
        plotYaw(simTime, simData, 7)
        plt.show()

    def reset(self, seed=None, options=None):

        self.last_hold_success = (self.hold_time >= self.hold_time_required)

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
        # new random force/direction profile for disturbances
        self.simulator.randomize_disturbances(self.np_random)


        self.episode_reward_breakdown = {
            "r_pos": 0.0,
            "r_d_dot": 0.0,
            "r_heading": 0.0,
            "r_heading2": 0.0,
            "r_surge": 0.0,
            "r_vel": 0.0,
            "r_time": 0.0,
            "r_action": 0.0,
            "r_hold": 0.0,
            "r_alloc": 0.0,
            "success_bonus": 0.0,
            "total_reward": 0.0,
        }

        self.current_step = 0
        self.target_arc_length = 0.0
        self.metrics.reset()
        self.cum_distance = 0.0
        self.reached_target_time = 0.0
        self.max_hold_time = 0.0
        self.objective_achieved = False
        self.reached_flag = False
        self.min_distance_to_target = np.inf

       
       
        # choose path mode
        if self.experiment_config is not None:
            # Deterministic experiment mode for finished-controller evaluation.
            self.path_mode = self.experiment_config["path_mode"]
        elif randomize_path:
            self.path_mode = self.np_random.choice(self.path_modes, p=path_probabilities)
        else:
            self.path_mode = "stationary"

        if self.experiment_config is not None:
            self.simulator.moving_target_start = np.array(
                self.experiment_config["target_start"],
                dtype=float,
            )

        if self.path_mode == "circle":
            self.simulator.circular_target = True
            self.simulator.use_moving_target = True
            self.simulator.zigzag_target = False
            self.simulator.waypoint_target = False

            self.simulator.radius = float(self.np_random.uniform(20.0, 60.0))
            self.simulator.target_radius = self.simulator.radius
            self.simulator.asd = 0.0

        elif self.path_mode == "line":
            self.simulator.circular_target = False
            self.simulator.use_moving_target = True
            self.simulator.zigzag_target = False
            self.simulator.waypoint_target = False

            if self.experiment_config is not None:
                self.simulator.moving_target_increase = np.array(
                    self.experiment_config.get("target_velocity", [0.0, 0.0]),
                    dtype=float,
                )
            else:
                velocity = float(self.np_random.uniform(0.5, 1.5))
                heading = float(self.np_random.uniform(-np.pi, np.pi))

                self.simulator.moving_target_increase = np.array([
                    velocity * np.cos(heading),
                    velocity * np.sin(heading)
                ], dtype=float)

        elif self.path_mode == "zigzag":
            self.simulator.circular_target = False
            self.simulator.use_moving_target = True
            self.simulator.zigzag_target = True
            self.simulator.waypoint_target = False

            velocity = float(self.np_random.uniform(0.5, 1.2))
            heading = float(self.np_random.uniform(-np.pi, np.pi))

            self.simulator.zigzag_velocity = velocity
            self.simulator.zigzag_base_heading = heading
            self.simulator.zigzag_angle = float(self.np_random.uniform(np.deg2rad(20), np.deg2rad(45)))

            # Switch direction after traveling roughly 15–20 meters
            zigzag_side_length = float(self.np_random.uniform(15.0, 20.0))
            dt_phys = 0.02

            self.simulator.zigzag_period_steps = int(
                zigzag_side_length / (velocity * dt_phys)
            )

            self.simulator.zigzag_step_counter = 0
            self.simulator.zigzag_direction = 1.0

            zigzag_heading = heading + self.simulator.zigzag_direction * self.simulator.zigzag_angle

            self.simulator.moving_target_increase = np.array([
                velocity * np.cos(zigzag_heading),
                velocity * np.sin(zigzag_heading)
            ], dtype=float)

        elif self.path_mode == "waypoint":
            self.simulator.circular_target = False
            self.simulator.use_moving_target = True
            self.simulator.zigzag_target = False
            self.simulator.waypoint_target = True

            self.simulator.waypoint_velocity = float(self.np_random.uniform(0.5, 1.2))
            self.simulator.waypoint_acceptance_radius = float(self.np_random.uniform(2.0, 5.0))
            self.simulator.waypoint_area_radius = float(self.np_random.uniform(30.0, 80.0))

            waypoint_angle = float(self.np_random.uniform(-np.pi, np.pi))
            waypoint_radius = float(self.np_random.uniform(10.0, self.simulator.waypoint_area_radius))

            self.simulator.current_waypoint = (
                self.simulator.moving_target_start
                + np.array([
                    waypoint_radius * np.cos(waypoint_angle),
                    waypoint_radius * np.sin(waypoint_angle)
                ], dtype=float)
            )

            direction = self.simulator.current_waypoint - self.simulator.moving_target_start
            direction_norm = np.linalg.norm(direction)

            if direction_norm > 1e-8:
                direction_unit = direction / direction_norm
            else:
                direction_unit = np.array([1.0, 0.0], dtype=float)

            self.simulator.moving_target_increase = (
                self.simulator.waypoint_velocity * direction_unit
            )

        elif self.path_mode == "stationary":
            self.simulator.circular_target = False
            self.simulator.use_moving_target = False
            self.simulator.zigzag_target = False
            self.simulator.waypoint_target = False

            self.simulator.moving_target_increase = np.array([0.0, 0.0], dtype=float)

        else:
            raise ValueError(f"Unknown path_mode: {self.path_mode}")

        self.simulator.moving_target = np.array(self.simulator.moving_target_start, dtype=float).copy()
        self.initial_target = list(self.simulator.moving_target)

        rng = self.np_random
        target_pos = np.array(self.simulator.moving_target, dtype=float)
        target_x, target_y = target_pos

        # choose initial USV pose
        if self.experiment_config is not None:
            eta_cfg = self.experiment_config.get(
                "initial_eta",
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
            x = float(eta_cfg[0])
            y = float(eta_cfg[1])
            yaw = float(eta_cfg[5])
            self.r0 = float(np.linalg.norm(target_pos - np.array([x, y], dtype=float)))
            self.alpha0 = float(np.arctan2(y - target_y, x - target_x))

        elif randomize_position:
            if self.path_mode == "stationary":
                # Randomize stationary tracking start distance
                r_min = stationary_start_min_dist
                r_max = stationary_start_max_dist

            elif self.simulator.circular_target:
                r_min = 5.0
                r_max = float(getattr(self.simulator, "radius", radius))

            else:
                r_min, r_max = 5.0, 25.0

            alpha = rng.uniform(-np.pi, np.pi)
            r = rng.uniform(r_min, r_max)

            x = target_x + r * np.cos(alpha)
            y = target_y + r * np.sin(alpha)

            # Optional: random yaw, harder exploration
            yaw = rng.uniform(-np.pi, np.pi)

            # Alternative: start facing the stationary target, easier training
            # yaw = wrap_to_pi(np.arctan2(target_y - y, target_x - x))

            self.r0 = float(r)
            self.alpha0 = float(alpha)

        else:
            x = target_x
            y = target_y + 10.0
            yaw = rng.uniform(-np.pi, np.pi)  
            self.r0 = 0.0
            self.alpha0 = 0.0

        eta_initial = [x, y, 0.0, 0.0, 0.0, yaw]
        self.simulator.initial_state(eta_initial)

        self.u_applied[:] = 0.0
        self.prev_applied[:] = 0.0
        self.prev_cmd[:] = 0.0
        self.prev_action[:] = 0.0
        self.last_alloc_scale = 1.0

        # stationary heading reference based on initial LOS
        usv_pos_init = np.array([x, y], dtype=float)
        rel_init = target_pos - usv_pos_init
        self.stationary_heading_ref = float(np.arctan2(rel_init[1], rel_init[0]))

        # target-heading tracking
        self.prev_target_pos = target_pos.copy()
        self.target_heading_ref = self.stationary_heading_ref

        # reset simulator
        distance_to_target, heading_error, nu = self.simulator.reset_simulation()

        # clear histories
        self.simData = []
        self.targetData = []
        self.simTime = []
        self.yawHistory = []
        self.distanceHistory = []
        self.headingErrorHistory = []
        self.hold_time = 0.0
        self.max_hold_time = 0.0

        eta0 = self.simulator.eta.copy()
        nu0 = self.simulator.nu.copy()
        u0 = self.simulator.u_actual.copy()
        commands0 = np.array([0.0, 0.0], dtype=float)

        full_state0 = np.hstack([eta0, nu0, commands0, u0])
        self.simData.append(full_state0)
        self.targetData.append(target_pos.copy())
        self.simTime.append(0.0)
        self.yawHistory.append(eta0[5])
        self.distanceHistory.append(distance_to_target)
        self.headingErrorHistory.append(heading_error)

        # build initial observation
        usv_pos = np.array([eta0[0], eta0[1]], dtype=float)
        rel_pos = target_pos - usv_pos
        x_rel = rel_pos[0]
        y_rel = rel_pos[1]

        e_d = np.linalg.norm(rel_pos)
        psi_los = float(np.arctan2(y_rel, x_rel))
        yaw_rel = wrap_to_pi(psi_los - eta0[5])
        d_dot = 0.0

        pos_scale = max_target_delta
        x_rel_norm = np.clip(x_rel / pos_scale, -1.0, 1.0)
        y_rel_norm = np.clip(y_rel / pos_scale, -1.0, 1.0)
        e_d_norm = np.clip(e_d / pos_scale, -1.0, 1.0)
        d_dot_norm = np.clip(d_dot / self.Umax, -1.0, 1.0)

        self.last_distance = e_d

        self.prev_x_rel = float(x_rel)
        self.prev_y_rel = float(y_rel)

        x_rel_dot_norm = 0.0
        y_rel_dot_norm = 0.0

        if USE_RELATIVE_VELOCITY_OBS:
            obs = np.array([
                x_rel_norm,
                y_rel_norm,
                yaw_rel / np.pi,
                np.clip(nu0[0] / self.Umax, -1.0, 1.0),
                np.clip(nu0[1] / self.Umax, -1.0, 1.0),
                np.clip(nu0[5] / self.Rmax, -1.0, 1.0),
                e_d_norm,
                d_dot_norm,
                x_rel_dot_norm,
                y_rel_dot_norm,
            ], dtype=np.float32)
        else:
            obs = np.array([
                x_rel_norm,
                y_rel_norm,
                yaw_rel / np.pi,
                np.clip(nu0[0] / self.Umax, -1.0, 1.0),
                np.clip(nu0[1] / self.Umax, -1.0, 1.0),
                np.clip(nu0[5] / self.Rmax, -1.0, 1.0),
                e_d_norm,
                d_dot_norm,
            ], dtype=np.float32)

        info = {}


        return obs, info


    def seed(self, seed=None):
        pass

    def project_normalized_surge_yaw(self, surge_cmd, yaw_cmd):
        """
        takes normalized desired surge/yaw commands in [-1, 1]
        surge_cmd =  1 means both thrusters forward
        yaw_cmd   =  1 means left forward, right reverse/less forward

        constraint is:
            left_cmd  = surge + yaw
            right_cmd = surge - yaw

        Both left/right must stay within [-1, 1].
        """

        surge_cmd = float(np.clip(surge_cmd, -1.0, 1.0))
        yaw_cmd   = float(np.clip(yaw_cmd,   -1.0, 1.0))

        left_cmd  = surge_cmd + yaw_cmd
        right_cmd = surge_cmd - yaw_cmd

        max_abs = max(abs(left_cmd), abs(right_cmd), 1.0)

        left_cmd  /= max_abs
        right_cmd /= max_abs

        surge_feasible = 0.5 * (left_cmd + right_cmd)
        yaw_feasible   = 0.5 * (left_cmd - right_cmd)

        alloc_scale = 1.0 / max_abs

        return surge_feasible, yaw_feasible, alloc_scale

def make_env(experiment_config=None):
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
            circular_target,
            radius,
            use_waves=wave_disturbance,
            use_wind=wind_disturbance,
        )
        return OtterEnv(simulator=simulator, otter=otter, experiment_config=experiment_config)
    return _init

class RewardCallback(BaseCallback):
    def __init__(self, n_envs, print_every=10, verbose=1):
        super().__init__(verbose)
        self.print_every = print_every

        self.n_envs = n_envs                                  # number of parallel envs
        self.current_rewards = np.zeros(n_envs)              # running reward per env
        self.episode_rewards = []                            # completed episode rewards
        self.episode_lengths = []                            # episode lengths
        self.current_lengths = np.zeros(n_envs)

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]                     # reward per env step
        dones = self.locals["dones"]                         # done flags per env

        self.current_rewards += rewards                      # accumulate rewards
        self.current_lengths += 1                            # count steps

        for i in range(self.n_envs):
            if dones[i]:                                     # episode finished in env i
                ep_r = self.current_rewards[i]
                ep_l = self.current_lengths[i]

                self.episode_rewards.append(ep_r)
                self.episode_lengths.append(ep_l)

                if self.verbose and len(self.episode_rewards) % self.print_every == 0:
                    print(f"Episode {len(self.episode_rewards)} - Reward: {ep_r:.2f}, Length: {ep_l}")

                # reset that env's counters
                self.current_rewards[i] = 0.0
                self.current_lengths[i] = 0.0

        return True

    def return_log(self):
        return self.episode_rewards, self.episode_lengths
    
# logs the distances and headings during training for IAE plotting
class CallBackLog(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.IAE_distance_history = []
        self.IAE_heading_history = []
        self.IAU_history = []

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals["dones"]):
            if done:
                IAE_distance = self.training_env.get_attr("last_IAE_distance", i)[0]
                IAE_heading = self.training_env.get_attr("last_IAE_heading", i)[0]
                IAU = self.training_env.get_attr("last_IAU", i)[0]

                self.IAE_distance_history.append(IAE_distance)
                self.IAE_heading_history.append(IAE_heading)
                self.IAU_history.append(IAU)

                if self.verbose and len(self.IAE_distance_history) % 10 == 0:
                    print(
                        f"Episode {len(self.IAE_distance_history)} - "
                        f"IAE Distance: {IAE_distance:.2f}, "
                        f"IAE Heading: {IAE_heading:.2f}, "
                        f"IAU: {IAU:.2f}"
                    )

        return True

    def return_log(self):
        return self.IAE_distance_history, self.IAE_heading_history, self.IAU_history

# stop if maintaining objective (above target for 10s straight) over threshold% of window_size-episodes
class StopOnSuccessRate(BaseCallback):
    def __init__(self, window_size=200, threshold=0.90, verbose=1):
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

def log_one_episode_reward_breakdown(model, eval_env):
    obs = eval_env.reset()
    done = False
    truncated = False
    episode_reward = 0.0

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)

        episode_reward += float(rewards[0])
        done = bool(dones[0])

        # Gymnasium vec envs put truncation into done
        info = infos[0]

        if done:
            breakdown = info.get("reward_breakdown", None)

            print("\nEpisode finished")
            print(f"Success: {info.get('is_success', False)}")
            print(f"Total episode reward returned by env: {episode_reward:.3f}")

            if breakdown is None:
                print("No reward breakdown found in info.")
                return

            print("\nReward breakdown:")
            for k, v in breakdown.items():
                print(f"{k:>15}: {v:.3f}")

            return


def select_experiment_paths():
    """Use final model/normalizer when available, otherwise fall back to checkpoint."""
    model_path = FINAL_MODEL if os.path.exists(FINAL_MODEL) else CHECKPOINT_MODEL
    vecnorm_path = FINAL_VECNORM if os.path.exists(FINAL_VECNORM) else CHECKPOINT_VECNORM

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(f"VecNormalize file not found: {vecnorm_path}")

    return model_path, vecnorm_path


def run_single_finished_controller_experiment(case, model_path, vecnorm_path):
    """Run one deterministic finished-controller experiment and save its log."""
    os.makedirs(EXPERIMENT_LOG_DIR, exist_ok=True)

    eval_env = DummyVecEnv([make_env(experiment_config=case)])
    eval_env = VecNormalize.load(vecnorm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(model_path, env=eval_env, device="cpu")

    # Get the real env before reset, then force fixed-duration experiment mode
    env_single = eval_env.envs[0]
    env_single.fixed_experiment_mode = True
    env_single.experiment_max_time = float(case.get("max_time", max_episode_time))

    obs = eval_env.reset()

    # DummyVecEnv reset may recreate/reset internal state, so set this again
    env_single = eval_env.envs[0]
    env_single.fixed_experiment_mode = True
    env_single.experiment_max_time = float(case.get("max_time", max_episode_time))

    max_steps = int(env_single.experiment_max_time / env_single.sampletime)

    episode_reward = 0.0
    info = {}

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)

        episode_reward += float(rewards[0])
        info = infos[0]

        if bool(dones[0]):
            break

    # -------------------------------------------------
    # Prefer terminal buffers, because DummyVecEnv may
    # auto-reset current buffers after done.
    # -------------------------------------------------
    def terminal_or_current(attr_names, current_value):
        if isinstance(attr_names, str):
            attr_names = [attr_names]

        for attr_name in attr_names:
            terminal_value = getattr(env_single, attr_name, None)

            if terminal_value is None:
                continue

            if hasattr(terminal_value, "__len__") and len(terminal_value) == 0:
                continue

            return np.asarray(terminal_value, dtype=float)

        return np.asarray(current_value, dtype=float)

    sim_time = terminal_or_current(
        ["last_simTime", "last_sim_time"],
        env_single.simTime,
    )

    sim_data = terminal_or_current(
        ["last_simData", "last_sim_data"],
        env_single.simData,
    )

    target_data = terminal_or_current(
        ["last_targetData", "last_target_data"],
        env_single.targetData,
    )

    # -------------------------------------------------
    # Make sure arrays are valid and aligned
    # -------------------------------------------------
    sim_time = np.asarray(sim_time, dtype=float).reshape(-1)
    sim_data = np.asarray(sim_data, dtype=float)
    target_data = np.asarray(target_data, dtype=float)

    if sim_data.ndim != 2:
        raise ValueError(f"sim_data has invalid shape: {sim_data.shape}")

    if target_data.ndim != 2:
        raise ValueError(f"target_data has invalid shape: {target_data.shape}")

    n = min(len(sim_time), sim_data.shape[0], target_data.shape[0])

    sim_time = sim_time[:n]
    sim_data = sim_data[:n, :]
    target_data = target_data[:n, :]

    if n < 2:
        raise ValueError(
            f"Not enough samples for experiment {case['name']}. "
            f"n={n}, sim_time={sim_time.shape}, sim_data={sim_data.shape}, "
            f"target_data={target_data.shape}"
        )

    # -------------------------------------------------
    # Save CSV log
    # -------------------------------------------------
    csv_path = os.path.join(EXPERIMENT_LOG_DIR, f"{case['name']}.csv")

    log_to_csv(
        simTime=sim_time,
        simData=sim_data,
        targetData=target_data,
        filename=csv_path,
        verbose=True,
    )

    # -------------------------------------------------
    # Compute IAE directly from the saved trajectory.
    #
    # sim_data[:, 0] = USV north
    # sim_data[:, 1] = USV east
    # sim_data[:, 5] = yaw/psi
    # target_data[:, 0] = target north
    # target_data[:, 1] = target east
    # -------------------------------------------------
    usv_north = sim_data[:, 0]
    usv_east = sim_data[:, 1]
    psi = sim_data[:, 5]

    target_north = target_data[:, 0]
    target_east = target_data[:, 1]

    north_error = target_north - usv_north
    east_error = target_east - usv_east

    distance_error = np.sqrt(north_error**2 + east_error**2)

    los_heading = np.arctan2(east_error, north_error)
    heading_error = np.array(
        [wrap_to_pi(los_heading[i] - psi[i]) for i in range(len(psi))],
        dtype=float,
    )

    valid = (
        np.isfinite(sim_time)
        & np.isfinite(distance_error)
        & np.isfinite(heading_error)
    )

    sim_time_metric = sim_time[valid]
    distance_metric = distance_error[valid]
    heading_metric = heading_error[valid]

    if len(sim_time_metric) < 2:
        raise ValueError(
            f"Not enough valid metric samples for experiment {case['name']}."
        )

    duration = max(sim_time_metric[-1] - sim_time_metric[0], 1e-9)

    iae_distance = float(np.trapezoid(np.abs(distance_metric), sim_time_metric))
    iae_heading = float(np.trapezoid(np.abs(heading_metric), sim_time_metric))
    avg_distance = float(
    np.trapezoid(distance_metric, sim_time_metric)
    / max(sim_time_metric[-1] - sim_time_metric[0], 1e-9)
)

    # Use stored terminal actuator metrics if available.
    iau = float(getattr(env_single, "last_IAU", np.nan))
    isu = float(getattr(env_single, "last_ISU", np.nan))
    isu_normalized = float(getattr(env_single, "last_ISU_normalized", np.nan))

    # If terminal values are missing, fall back to current metrics.
    if not np.isfinite(iau):
        iau = float(env_single.metrics.get_IAU())

    if not np.isfinite(isu):
        isu = float(env_single.metrics.get_ISU())

    if not np.isfinite(isu_normalized):
        isu_normalized = float(env_single.metrics.get_ISU_normalized())

    success = bool(
        info.get("is_success", False)
        or getattr(env_single, "objective_achieved", False)
        or getattr(env_single, "max_hold_time", 0.0) >= env_single.hold_time_required
    )

    max_hold_time = float(
        getattr(env_single, "max_hold_time", info.get("max_hold_time", np.nan))
    )

    intercept_time = float(
        getattr(env_single, "reached_target_time", info.get("intercept_time", np.nan))
    )

    param_dict = {
        "Control_method": "DRL",
        "experiment": case["name"],
        "model_path": model_path,
        "vecnormalize_path": vecnorm_path,

        "target_start_N": float(case["target_start"][0]),
        "target_start_E": float(case["target_start"][1]),
        "target_velocity_N": float(case["target_velocity"][0]),
        "target_velocity_E": float(case["target_velocity"][1]),

        "initial_north": float(case["initial_eta"][0]),
        "initial_east": float(case["initial_eta"][1]),
        "initial_yaw": float(case["initial_eta"][5]),

        "episode_reward": float(episode_reward),
        "success": int(success),

        "IAE_distance": iae_distance,
        "IAE_heading": iae_heading,
        "IAU": iau,
        "ISU": isu,
        "ISU_normalized": isu_normalized,

        "avg_distance": avg_distance,
        "intercept_time": intercept_time,
        "max_hold_time": max_hold_time,
        "duration": float(duration),
        "n_samples": int(n),
    }

    param_path = os.path.join(EXPERIMENT_LOG_DIR, f"{case['name']}_parameters.txt")
    io_log_params(param_dict, filename=param_path, verbose=True)

    print(f"\nFinished experiment: {case['name']}")
    print(f"  log: {csv_path}")
    print(f"  reward: {episode_reward:.3f}")
    print(f"  success: {success}")
    print(f"  duration: {duration:.2f}")
    print(f"  samples: {n}")
    print(f"  avg_distance: {avg_distance}")
    print(f"  IAE_distance: {iae_distance}")
    print(f"  IAE_heading: {iae_heading}")
    print(f"  IAU: {iau}")

    eval_env.close()

    return param_dict


def run_finished_controller_experiments():
    """Run the two fixed cases used for controller comparison."""
    model_path, vecnorm_path = select_experiment_paths()

    print("\nRunning finished-controller DRL experiments")
    print(f"Model: {model_path}")
    print(f"VecNormalize: {vecnorm_path}")
    print(f"Logs: {EXPERIMENT_LOG_DIR}\n")

    results = []
    for case in FINISHED_CONTROLLER_EXPERIMENTS:
        results.append(run_single_finished_controller_experiment(case, model_path, vecnorm_path))

    summary_path = os.path.join(EXPERIMENT_LOG_DIR, "finished_controller_experiment_summary.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"\nSaved experiment summary: {summary_path}")

if __name__ == "__main__":

    print(f"Using device: {device}")

    n_envs = 8

    print("\n\n\n\n##################################################################")
    print("\n ENSURE CORRECT PATHS FOR CHECKPOINTS TO PREVENT OVERRIDING FIELES\n")
    print("##################################################################")
    mode = int(input("\n\nChoose '1' for model training or '2' to run saved model, or '3' to complete & log experiment: "))
    if mode == 1:

        env = SubprocVecEnv([make_env() for _ in range(n_envs)])
        env = VecMonitor(env)   

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
                gae_lambda=0.95,      # advantage estimation 
                n_steps=4096,         # larger>more stable
                batch_size=256,       # Good balance for large rollouts
                n_epochs=10,          # Standard PPO setting


                # Exploration control
                ent_coef=0.005,         # larger more = exploration less stable
                vf_coef=0.2,            # critic doesn't drown actor=

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
        reward_callback = RewardCallback(n_envs=n_envs, print_every=10, verbose=1)
        stop_callback = StopOnSuccessRate(window_size=200, threshold=0.90, verbose=1) # threshold = % of window size required to be success 
        interrupted = False

        try:
            model.learn(
                total_timesteps=training_timesteps,
                callback=[reward_callback, IAE_callback, stop_callback],
            )

        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted. Will save checkpoint.")

        finally:
            print("\nTraining ended. Saving checkpoint and logs.")

            # Save checkpoint model
            model.save(CHECKPOINT_MODEL)

            # Save VecNormalize statistics
            if isinstance(env, VecNormalize):
                env.save(CHECKPOINT_VECNORM)

            # Save CSV logs for BOTH interrupt and StopOnSuccess
            append_reward_training_progress(
                os.path.join(SAVE_DIR, "reward_training_progress.csv"),
                reward_callback
            )

            append_iae_training_progress(
                os.path.join(SAVE_DIR, "iae_training_progress.csv"),
                IAE_callback
            )

            rewards, lengths = reward_callback.return_log()

            print(f"Collected {len(rewards)} completed episodes")

            # Save final model if success criterion was reached
            if stop_callback.completed and not interrupted:
                print("\nSuccess-rate criterion reached. Saving final model.")

                model.save(FINAL_MODEL)

                if isinstance(env, VecNormalize):
                    env.save(FINAL_VECNORM)

            # Optional plot for both interrupted and completed training
            if len(rewards) > 0:
                plt.figure()
                plt.plot(rewards, linewidth=0.8, label="Episode reward")
                plt.xlabel("Episode")
                plt.ylabel("Reward")
                plt.title("Episode Reward During Training")
                plt.legend()
                plt.grid(True)
                plt.show()

            if interrupted:
                print("\nInterrupted training saved successfully.")

            elif stop_callback.completed:
                print("\nStopOnSuccess training saved successfully.")

            else:
                print("\nTraining reached total_timesteps and was saved successfully.")

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
        log_one_episode_reward_breakdown(model, eval_env)

    elif mode == 3:
        run_finished_controller_experiments()
        sys.exit(0)

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
                "max_hold_time": [],
                "r0": [],
                "alpha0": []
                }
    
    # for MC use 540000 for 200 samples (saving will be slow)
    for i in range(100000):

        env_single = eval_env.envs[0]

        last_target = env_single.simulator.moving_target.copy()
        action, _states = model.predict(obs, deterministic=True)
        
        episode_actions.append(action)
        obs, rewards, dones, infos = eval_env.step(action)

        if dones[0]:
            ep_info = infos[0]
            mc_log["IAE_distance"].append(ep_info.get("IAE_distance", np.nan))
            mc_log["IAE_heading"].append(ep_info.get("IAE_heading", np.nan))
            mc_log["avg_distance"].append(ep_info.get("avg_distance", np.nan))
            mc_log["intercept_time"].append(ep_info.get("intercept_time", np.nan))
            mc_log["success"].append(int(ep_info.get("is_success", False)))
            mc_log["max_hold_time"].append(ep_info.get("max_hold_time", np.nan))
            mc_log["r0"].append(ep_info.get("r0", np.nan))
            mc_log["alpha0"].append(ep_info.get("alpha0", np.nan))

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
                    

                # increase to compare more plots
                if episode_count < 3:
                    env_single.render()

                episode_count += 1
                episode_actions = []
                obs = eval_env.reset()

    print(f"Average eval IAE distance: {np.mean(iae_distances):.2f}")
    print(f"Average eval IAE heading:  {np.mean(iae_headings):.2f}")

    success_rate = np.mean(mc_log["success"])
    print(f"Evaluation success rate: {success_rate:.2%}")

    # save mc for plotting 
    df = pd.DataFrame(mc_log)
    df.to_csv("monte_carlo_results.csv", index=False)

    rewards, lengths = reward_callback.return_log()               # get stored episode rewards and lengths

    plt.figure()
    plt.plot(rewards, linewidth=0.8, label="Episode reward")      # raw episode reward
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Reward During Training")
    plt.legend()
    plt.grid(True)
    plt.show()
    
