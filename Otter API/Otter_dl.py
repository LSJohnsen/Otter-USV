import Otter_api
import lib.PID_Controller_test_v2 as PID_Controller_test_v2
import Otter_simulator_DRL
import lib.Live_guidance as Live_guidance
import lib.Live_plotter as Live_plotter
from lib.plotTimeSeries import *
import matplotlib.pyplot as plt
import threading
import atexit
import time 
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import torch
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") # Stable-baselines3 PPO runs faster on CPU!
print(f"Using device: {device}")    



use_target_coordinates = False                                                                          # To use coordinates as a target or to use a linear path
use_moving_target = True                                                                                # To use moving target instead of target list (path following)
target_list = [[0, 10000]]                                                                              # List of targets to use if use_target_coordinates is set to True
end_when_last_target_reached = True                                                                     # Ends the simulation when the final target is reached
moving_target_start = [0, 0]                                                                            # Start point of the moving target if use_moving_target is set to True
moving_target_increase = [1.5, 0.0]                                                                     # Movement of the moving target each second
max_target_delta = 50                                                                                   # How many meters target should move each simulation before truncation
target_radius = 0.5                                                                                     # Radius from center of target that counts as target reached, change this depending on the complete size of the run. Very low values causes instabillity
verbose = True                                                                                          # Enable verbose printing
store_force_file = False                                                                                # Store the simulated control forces in a .csv file
circular_target = False                                                                                 # Make the moving target a circle in the simulation
animate_path = False                                                                        


start_north = -20                                                                                       # Target north position from referance point
start_east = -20                                                                                        # Target east position from referance point
v_north = 0                                                                                             # Moving target speed north (m/s)
v_east = -1.5                                                                                           # Moving target speed east (m/s)
radius = 40                                                                                             # If tracking a circular motion
v_circle = 1.5                                                                                          # Angular velocity (m/s)
side_length = 50                                                                                        # Square tracking side length
side_target_speed = 1                                                                                   # Speed of square target


numDataPoints = 830                                                                                     # number of 3D data points
FPS = 60                                                                                                # frames per second (animated GIF)
filename = '3D_animation.gif'                                                                           # data file for animated GIF
browser = 'chrome'      


otter = Otter_api.otter()
simulator = Otter_simulator_DRL.OtterSimDRL(target_list, # Creates Simulator object
                                            use_target_coordinates, 
                                            target_radius, 
                                            use_moving_target, 
                                            moving_target_start, 
                                            moving_target_increase,
                                            end_when_last_target_reached, 
                                            verbose, 
                                            store_force_file, 
                                            circular_target)          
                   
print("initialized ottter api and simulator")

otter.controls = ["Left propeller shaft speed (rad/s)", "Right propeller shaft speed (rad/s)"]           # Some values needed for the plotting
otter.dimU = len(otter.controls)                                                                        


class OtterEnv(gym.Env):
    def __init__(self, simulator, otter):
        super().__init__()                                                                               # call gym.env constructor 

        self.simulator = simulator
        self.otter = otter

        self.sampletime = 0.1 #iteration update 1/hz
        self.episode_duration = 10000 # no. simulation samples
        self.sim_duration = int(self.episode_duration / self.sampletime) #sim duration
        self.current_step = 0

        self.Umax = 6 * 0.5144
        self.max_force = 200 

        self.simData = []
        self.targetData = []
        self.simTime = []
        self.initial_target = list(self.simulator.moving_target)
        self.has_plotted = False

        self.observation_space = Box(low=np.array([-1,                       # min/max distance to target, angle to target, surge/sway velocity
                                                   -np.pi,                   # normalized between 0-1, (200N, Umax)
                                                   -1, 
                                                   -1], 
                                                    dtype=np.float32),         
                                     high=np.array([1,                   
                                                    np.pi,    
                                                    1,
                                                    1],
                                                    dtype=np.float32))         
        
        # self.action_space = Box(low=np.array([-self.max_force, 
        #                                       -self.max_force/2], #min/max forces in surge/yaw
        #                                       dtype=np.float32), 
        #                         high=np.array([self.max_force, 
        #                                        self.max_force/2],
        #                                        dtype=np.float32))

        self.action_space = Box(low=np.array([-1, 
                                              -1], #min/max forces in surge/yaw normalized
                                              dtype=np.float32), 
                                high=np.array([1, 
                                               1],
                                               dtype=np.float32))
    



    def step(self, action):


        
        self.current_step += 1 
    
        #Simulate

        tau_X, tau_N = action
        

        eta, nu, target, distance_to_target, heading_error = self.simulator.simulate_step(self.sampletime,
                                                                                            self.otter,
                                                                                            tau_X,
                                                                                            tau_N)
        if self.current_step % 1000 == 0:
            with open("debug_log2.txt", "w") as f:
                f.write(f"Step: {self.current_step}, Current heading: {eta[5]} Heading error: {heading_error}, Distance to target: {distance_to_target}\n")



        #SimData vessel data:
            # eta (velocities in world frame (x, y, z, roll, pitch, yaw)), 
            # nu(velocities in body frame (u, v, w, p, q, r)), 
            # thrusters command (n1, n2), 
            # actuator output (u1, u2)

        #Store data for rendering
        self.simTime.append(self.current_step * self.sampletime)
        self.simData.append(eta[:2])
        self.targetData.append(target)
        self.yawHistory.append(eta[5]) 
        #Targetdata data on target x(north)/y(east) distance
        if circular_target == True:
            normalized_distance = np.clip(distance_to_target / radius, -1.0, 1.0)
        else:
             #Targetdata data on target x(north)/y(east) distance
            normalized_distance = np.tanh(distance_to_target / 50.0)


        usv_speed = np.linalg.norm([nu[0], nu[1]])  # total velocity
        target_speed = v_circle  
        
        obs = np.array([normalized_distance, #current distance to target normalized
                             heading_error, 
                             np.clip(nu[0] / self.Umax, -1.0, 1.0), #surge velocity
                             np.clip(nu[1] / self.Umax, -1.0, 1.0), #sway velocity
                             ], dtype = np.float32)
        

        usv_speed = np.linalg.norm([nu[0], nu[1]])  # total velocity
        target_speed = v_circle  


        target_delta = abs(self.simulator.moving_target[0] - self.start_target_x)
        
        truncated = target_delta >= max_target_delta or self.current_step > self.sim_duration



        # Reward handling
        reward = 0
        terminated = False
        info = {}
        
        if not hasattr(self, 'last_distance'):
            self.last_distance = distance_to_target
 
        reward += 3*(self.last_distance - distance_to_target)
        reward -= 0.1 * distance_to_target
        reward +=  0.5 * np.cos(heading_error)
        reward -=  0.2 * abs(nu[1])  # sway penalty
        reward -=  0.1 * abs(nu[5])  # yaw‐rate penalty
        self.last_distance = distance_to_target

        # if truncated:
        #     reward -= 50
        
        if distance_to_target < self.simulator.surge_setpoint:
            reward += 1.0


        # reward = -distance_to_target #reward/penalty for moving closer/further away from target  self.last_distance - distance_to_target
        # #reward -= 0.5 * abs(angle_to_target)

        # if usv_speed > 0.5 * target_speed: # and usv_speed < target_speed * 1.1: #reward for keeping the same velocity as circle target
        #     reward += 1 - abs(nu[0] - target_speed) / target_speed #normalizing difference to give greater reward for surge velocity

        # reward -= abs(nu[1]) * 0.2

        # if distance_to_target < 1:
        #     reward += 5
        
        # # reward /= 10
        # if self.current_step % 2000 == 0:
        #     print(f"Step: {self.current_step}, Reward: {reward}, Distance: {distance_to_target}")

        # self.last_distance = distance_to_target

        # if abs(heading_error) > np.pi * 0.2 or abs(heading_error) < -np.pi * 0.2: #Negative reward and terminate episode if yaw angle is very poor
        #     reward -= 5
        #     terminated = False
        #     #print(f"Terminated episode at step {self.current_step} due to bad yaw angle at {angle_to_target} radians")
        # else:
        #     terminated = False

        

        return obs, reward, terminated, truncated, info
    

    def render(self, mode="human"):    

        if len(self.simData) > 1:
            simData_array = np.array(self.simData)
            targetData_array = np.array(self.targetData)
            sim_time_array = np.array(self.simTime)
            heading_array = np.array(self.yawHistory)

            #reward_array = np.array(self.rewardHistory)
            #IAE_array

            plotPosTar(sim_time_array, simData_array, 1, targetData_array, headings=heading_array)
            plt.show()



    def reset(self, seed=None, options=None):
        
        super().reset(seed=seed)  # sets up random seed tracking


        if seed is not None:
            np.random.seed(seed)

        self.simulator.moving_target = list(self.initial_target)

        self.start_target_x = self.simulator.moving_target[0]
        self.current_step = 0
        self.simData = []
        self.targetData = []
        self.simTime = []
        self.yawHistory = [] 

        # Target position
        target_pos = np.array(self.simulator.moving_target)
        target_x, target_y = target_pos
        
        #limits initial north distance and yaw angle to target after reset
        distance = -1.0
        yaw = np.arctan2(self.simulator.moving_target_increase[1], 
                         self.simulator.moving_target_increase[0])  #gets the correct initial angle of the target movement
        #yaw = np.pi

        x = target_x - distance * np.cos(yaw)
        y = target_y - distance * np.sin(yaw)
        

        eta_initial = [x, y, 0, 0, 0, yaw]
        self.simulator.initial_state(eta_initial) #sets initial state on the target location and angle

       
        distance_to_target, heading_error, nu = self.simulator.reset_simulation()

        if circular_target == True:
            normalized_distance = np.clip(distance_to_target / radius, -1.0, 1.0)
        else:
             #Targetdata data on target x(north)/y(east) distance
            normalized_distance = np.tanh(distance_to_target / 50.0)

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

if __name__ == "__main__":


    print(f"Using device: {device}")    

    # Setup environment
    n_envs = 8 # run n_environments
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    # Train model

    # classstable_baselines3.ppo.PPO(policy, env, learning_rate=0.0003, 
    #                                n_steps=2048, batch_size=64, n_epochs=10, 
    #                                gamma=0.99, gae_lambda=0.95, clip_range=0.2, 
    #                                clip_range_vf=None, normalize_advantage=True, 
    #                                ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, 
    #                                use_sde=False, sde_sample_freq=-1, rollout_buffer_class=None, 
    #                                rollout_buffer_kwargs=None, target_kl=None, stats_window_size=100, 
    #                                tensorboard_log=None, policy_kwargs=None, verbose=0, seed=None, device='auto',
    #                                 _init_setup_model=True

    model = PPO(MlpPolicy, env, verbose=1, #mlppolicy built in 2 64x NNs
                device="cpu", 
                normalize_advantage=True,
                gae_lambda=0.98,
                learning_rate=0.0001,
                clip_range=0.2,
                n_steps=4096, #2048 
                ent_coef=0.01, # Entropy coefficient -> higher value less convergence
                target_kl=None) #0.03 #stops earlier if diverging too fast 
    
    mode = int(input("\n \nChoose '1' for model training or '2' to run saved model: "))
    if mode == 1:
        print("\nTraining model: \n")
        model.learn(total_timesteps=1000000)
    elif mode == 2:
        pass
    else:
        print("Chosen option is not valid.")

    model.save("ppo_otter_model")
    eval_env = DummyVecEnv([make_env()])   # Evaluation

    obs = eval_env.reset()
    for i in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, dones, infos = eval_env.step(action)
        
        if i % 1000 == 0:
            eval_env.envs[0].render()

        if dones[0]:
            eval_env.envs[0].render()
            obs = eval_env.reset()

    eval_env.close()
