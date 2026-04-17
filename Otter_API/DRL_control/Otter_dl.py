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
use_moving_target = True                                                                                # To use moving target instead of target list (path following)
target_list = [[0, 10000]]                                                                              # List of targets to use if use_target_coordinates is set to True
end_when_last_target_reached = True                                                                     # Ends the simulation when the final target is reached
moving_target_start = [0, 0]                                                                            # Start point of the moving target if use_moving_target is set to True
moving_target_increase = [-0.5, 0.0]                                                                    # Movement of the moving target each second                                                                                  # How many meters target should move each simulation before truncation
target_radius = 0.2                                                                                     # Radius from center of target that counts as target reached
verbose = True                                                                                          # Enable verbose printing
store_force_file = False                                                                                # Store the simulated control forces in a .csv file
circular_target = True                                                                                  # Make the moving target a circle in the simulation
animate_path = False
training_timesteps = 100000000                                                                          # Set timesteps (10mil-50mil+ depending on straight/circle)
log_results = False                                                                                     # log sim to csv, false when training

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
path_probabilities = [0, 1, 0]                                                                          # probability of [stationary, straight line, circle] target movement

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

def wrap_to_pi(angle): 
    return (angle + np.pi) % (2 * np.pi) - np.pi

class OtterEnv(gym.Env):
    def __init__(self, simulator, otter):
        super().__init__()                                                                               # call gym.env constructor

        self.simulator = simulator
        self.otter = otter


        # Overwrite when training one at a time. station->line->all three
        self.path_modes = ["stationary", "line", "circle"] # line, stationary, circle


        self.sampletime = 0.1  # iteration updates
        self.episode_duration = 400000  # no. simulation samples (truncates at distances, just ensure not too small)
        self.sim_duration = int(self.episode_duration / self.sampletime)  # sim duration
        self.current_step = 0
        self.target_arc_length = 0
        self.prev_tau_X = 0
        self.prev_action = np.zeros(2, dtype=float)

        # callback function for finished learning
        self.hold_radius = 0.2          # meters
        self.hold_time_required = 10.0  # seconds
        self.hold_time = 0.0            # accumulate within radius
        self.max_hold_time = 0.0        # longest period
        self.last_hold_success = False  # stored at end of episode

        # used for observation/action/rewards
        self.Umax = 6 * 0.5144
        self.Rmax = 2                   # Just a chosen relative value for normalization 2rad/s
        self.tauX_max = 150
        self.tauN_max = 110         
        self.max_rad = 0.0
        self.u_applied = np.zeros(2)
        self.tau_act = 1.0     
        self.last_distance = 0
        
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
        '''
        # normalized values
        self.observation_space = Box(low=np.array([-1,                   # distance to target
                                                   -1,                   # heading error
                                                   -1,                   # surge velocity
                                                   -1,                   # sway velocity
                                                   -1,                   # yaw rate
                                                   -1],                  # target error rate of change (d_dot) 
                                                  dtype=np.float32),
                                     high=np.array([1,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    1],
                                                   dtype=np.float32))
        '''
        # with relative x,y,r
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
                1,  # x_rel
                1,  # y_rel
                1,  # yaw_rel
                1,  # surge vel
                1,  # sway vel
                1,  # yaw vel
                1,  # euclidean dist
                1,  # euclidean dist rate
            ], dtype=np.float32)
)

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
        prev_distance = self.last_distance                                      # previous Euclidean distance for d_dot

        # raw action command
        tau_cmd = np.array(action, dtype=float)                                 # PPO output action

        # actuator lag
        alpha_t = self.sampletime / (self.tau_act + self.sampletime)            # first-order actuator lag factor
        self.u_applied = (1 - alpha_t) * self.u_applied + alpha_t * tau_cmd     # lagged actuator command

        # clamp applied tau to physical limits
        self.u_applied = np.clip(
            self.u_applied,
            [-self.tauX_max, -self.tauN_max],
            [ self.tauX_max,  self.tauN_max],
        )                                                                       # saturate to actuator limits

        tau_X, tau_N = self.u_applied                                           # applied control inputs

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
        commands = self.u_applied                                               # actual command sent to plant
        actuals = u_actual                                                      # measured/actual thruster outputs
        full_state = np.hstack([eta, nu, commands, actuals])                    # log full state vector

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

        # update hold logic
        if distance_to_target <= self.hold_radius:
            self.hold_time += self.sampletime                                   # accumulate hold time
        else:
            self.hold_time = 0.0                                                # reset hold time if outside hold radius

        self.max_hold_time = max(self.max_hold_time, self.hold_time)            # log max hold time
        success = self.hold_time >= self.hold_time_required                     # success if held long enough

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

        # updated observation
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

        '''
        obs = np.array([
            normalized_distance,                         # distance normalized
            heading_error / np.pi,                       # heading normalized
            np.clip(nu[0] / self.Umax, -1.0, 1.0),       # surge velocity normalized
            np.clip(nu[1] / self.Umax, -1.0, 1.0),       # sway velocity normalized
            np.clip(nu[5] / self.Rmax, -1.0, 1.0),       # yaw rate normalized
            np.clip(d_dot / self.Umax, -1.0, 1.0)        # closing speed normalized
        ], dtype=np.float32)
        

        # Determine target deltas for episode termination 
        e
        '''


        ''' 
        currently reset at start of steps, can be removed or use later? useless for time based ml?
        '''

        
        episode_time = self.current_step * self.sampletime
        truncated = episode_time >= max_episode_time
        terminated = success
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
        C_t = 0.001                                                         # small constant penalty per step
        r_time = C_t                                                        # faster convergence

        # action penalty
        scale = np.array([self.tauX_max, self.tauN_max], dtype=float)       # normalization for actuator commands
        delta_cmd = (tau_cmd - self.prev_cmd) / scale                       # normalized command change
        C_a = 0.01                                                          # amplitude of actuator penalty
        r_action = C_a * (abs(delta_cmd[0]) + abs(delta_cmd[1]))            # penalty for aggressive actuator changes

        self.prev_cmd[:] = tau_cmd                                          # update curr cmd          

        # Continuous hold reward when in range
        t_short = self.hold_time_required / 5.0                             # small reward for holding 2 seconds
        t_long  = self.hold_time_required                                      

        # increasing reward for staying on targe
        hold_ratio_short = np.clip(self.hold_time / max(t_short, 1e-6), 0.0, 1.0)**2   # increasing reward for staying on targe -.. testing scaling to prioritize holding longer
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
        reward += r_hold                                                        # reward for hovering above target
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
            info["is_success"] = bool(success)
            info["max_hold_time"] = self.max_hold_time
            info["IAE_distance"] = self.last_IAE_distance
            info["IAE_heading"] = self.last_IAE_heading
            info["avg_distance"] = self.last_avg_distance
            info["intercept_time"] = self.last_reached_target_time
            info["r0"] = self.r0
            info["alpha0"] = self.alpha0
            info["reward_breakdown"] = self.episode_reward_breakdown.copy()



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
            "success_bonus": 0.0,
            "total_reward": 0.0,
        }

        self.current_step = 0
        self.target_arc_length = 0.0
        self.metrics.reset()
        self.cum_distance = 0.0
        self.reached_target_time = 0.0
        self.max_hold_time = 0.0
        self.reached_flag = False

        # choose path mode
        if randomize_path:
            self.path_mode = self.np_random.choice(self.path_modes, p=path_probabilities)
        else:
            self.path_mode = "stationary"

        if self.path_mode == "circle":
            self.simulator.circular_target = True
            self.simulator.use_moving_target = True
            self.simulator.radius = float(self.np_random.uniform(20.0, 60.0))
            self.simulator.asd = 0.0

        elif self.path_mode == "line":
            self.simulator.circular_target = False
            self.simulator.use_moving_target = True
            velocity = float(self.np_random.uniform(0.2, 0.5))
            heading = float(self.np_random.uniform(-np.pi/2, np.pi/2))
            #heading = float(2*np.pi)
            self.simulator.moving_target_increase = np.array([
                velocity * np.cos(heading),
                velocity * np.sin(heading)
            ], dtype=float)

        elif self.path_mode == "stationary":
            self.simulator.circular_target = False
            self.simulator.use_moving_target = False
            self.simulator.moving_target_increase = np.array([0.0, 0.0], dtype=float)

        else:
            raise ValueError(f"Unknown path_mode: {self.path_mode}")

        self.simulator.moving_target = self.simulator.moving_target_start.copy()
        self.initial_target = list(self.simulator.moving_target)

        rng = self.np_random
        target_pos = np.array(self.simulator.moving_target, dtype=float)
        target_x, target_y = target_pos

        # choose initial USV pose
        if randomize_position:
            if self.simulator.circular_target:
                r_min = 5.0
                r_max = float(getattr(self.simulator, "radius", radius))
            else:
                r_min, r_max = 5.0, 25.0

            alpha = rng.uniform(-np.pi, np.pi)
            r = rng.uniform(r_min, r_max)

            x = target_x + r * np.cos(alpha)
            y = target_y + r * np.sin(alpha)
            yaw = rng.uniform(-np.pi, np.pi)

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

        self.prev_action[:] = 0.0
        self.prev_cmd[:] = 0.0

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

        '''
        obs = np.array([
            normalized_distance,                         # distance normalized
            heading_error / np.pi,                       # heading normalized
            np.clip(nu[0] / self.Umax, -1.0, 1.0),       # surge velocity normalized
            np.clip(nu[1] / self.Umax, -1.0, 1.0),       # sway velocity normalized
            np.clip(nu[5] / self.Rmax, -1.0, 1.0),       # yaw rate normalized
            np.clip(d_dot / self.Umax, -1.0, 1.0)        # closing speed normalized
        ], dtype=np.float32)
        '''

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
    def __init__(self, window_size=200, threshold=0.95, verbose=1):
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

if __name__ == "__main__":

    print(f"Using device: {device}")

    n_envs = 8

    print("\n\n\n\n##################################################################")
    print("\n ENSURE CORRECT PATHS FOR CHECKPOINTS TO PREVENT OVERRIDING FIELES\n")
    print("##################################################################")
    mode = int(input("\n\nChoose '1' for model training or '2' to run saved model: "))
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
        stop_callback = StopOnSuccessRate(window_size=200, threshold=0.95, verbose=1) # threshold = % of window size required to be success 
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
            print("\nCompleted total timesteps: saving checkpoint")
            model.save(CHECKPOINT_MODEL)
            if isinstance(env, VecNormalize):
                env.save(CHECKPOINT_VECNORM)

            # always save training logs
            append_reward_training_progress(
                os.path.join(SAVE_DIR, "reward_training_progress.csv"),
                reward_callback
            )
            append_iae_training_progress(
                os.path.join(SAVE_DIR, "iae_training_progress.csv"),
                IAE_callback
            )

            if stop_callback.completed and not interrupted:
                print("\n Goal reached, saving final model")
                model.save(FINAL_MODEL)
                if isinstance(env, VecNormalize):
                    env.save(FINAL_VECNORM)

            if interrupted:
                rewards, lengths = reward_callback.return_log()

                print(f"Collected {len(rewards)} completed episodes")

                if len(rewards) > 0:
                    plt.figure()
                    plt.plot(rewards, linewidth=0.8, label="Episode reward")
                    plt.xlabel("Episode")
                    plt.ylabel("Reward")
                    plt.title("Episode Reward During Training")
                    plt.legend()
                    plt.grid(True)
                    plt.show()

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
    

try:
    if "IAE_callback" in globals() and mode == 1:
        append_iae_training_progress(os.path.join(SAVE_DIR, "iae_training_progress.csv"), IAE_callback)
except Exception as e:
    print(f"IAE CSV: Failed to write IAE progress: {e}")