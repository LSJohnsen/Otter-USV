# lib/Live_DRL_controller.py

import os
import time
import datetime
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class DummyDRLEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.observation_space = Box(
            low=np.array([-1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
        )

        self.action_space = Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
        )

    def reset(self, seed=None, options=None):
        return np.zeros(8, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(8, dtype=np.float32), 0.0, False, False, {}


class LiveDRLController:
    def __init__(
        self,
        live_guidance,
        model_path,
        vecnormalize_path,
        max_target_delta=250.0,
        Umax=6 * 0.5144,
        Rmax=2.0,
        tauX_max=150.0,
        tauN_max=110.0,
        scale_action=False,
    ):
        self.live_guidance = live_guidance

        self.max_target_delta = max_target_delta
        self.Umax = Umax
        self.Rmax = Rmax

        self.tauX_max = tauX_max
        self.tauN_max = tauN_max
        self.scale_action = scale_action

        dummy_env = DummyVecEnv([lambda: DummyDRLEnv()])
        self.vecnorm = VecNormalize.load(vecnormalize_path, dummy_env)
        self.vecnorm.training = False
        self.vecnorm.norm_reward = False

        self.model = PPO.load(model_path, device="cpu")

        self.last_distance = None
        self.log = pd.DataFrame()

    def make_observation(self):
        state = self.live_guidance.current_state()

        if state is None:
            return None

        x, y, psi, u, v, r = state

        target_pos = np.array(self.live_guidance.target_ne_pos, dtype=float)
        usv_pos = np.array([x, y], dtype=float)

        rel_pos = target_pos - usv_pos
        x_rel = rel_pos[0]
        y_rel = rel_pos[1]

        distance = float(np.linalg.norm(rel_pos))
        psi_los = float(np.arctan2(y_rel, x_rel))
        yaw_rel = wrap_to_pi(psi_los - psi)

        if self.last_distance is None:
            d_dot = 0.0
        else:
            d_dot = (distance - self.last_distance) / self.live_guidance.cycletime

        self.last_distance = distance

        obs = np.array([
            np.clip(x_rel / self.max_target_delta, -1.0, 1.0),
            np.clip(y_rel / self.max_target_delta, -1.0, 1.0),
            np.clip(yaw_rel / np.pi, -1.0, 1.0),
            np.clip(u / self.Umax, -1.0, 1.0),
            np.clip(v / self.Umax, -1.0, 1.0),
            np.clip(r / self.Rmax, -1.0, 1.0),
            np.clip(distance / self.max_target_delta, -1.0, 1.0),
            np.clip(d_dot / self.Umax, -1.0, 1.0),
        ], dtype=np.float32)

        return obs

    def predict_control(self):
        obs = self.make_observation()

        if obs is None:
            return None, None

        obs_vec = obs.reshape(1, -1)
        obs_norm = self.vecnorm.normalize_obs(obs_vec)

        action, _ = self.model.predict(obs_norm, deterministic=True)
        action = np.asarray(action).reshape(-1)

        if self.scale_action:
            tau_X = float(action[0] * self.tauX_max)
            tau_N = float(action[1] * self.tauN_max)
        else:
            tau_X = float(action[0])
            tau_N = float(action[1])

        return tau_X, tau_N

    def stationary_tracking(self, forward_offset=10.0, starboard_offset=5.0):
        lg = self.live_guidance

        lg.otter.establish_connection(lg.ip, lg.port)
        lg.otter.update_values()

        lg.referance_point = [
            lg.otter.sorted_values["lat"],
            lg.otter.sorted_values["lon"],
            0.0
        ]
        lg.otter.observer_coordinates = lg.referance_point

        psi = lg.get_initial_heading()

        start_north = (
            forward_offset * np.cos(psi)
            - starboard_offset * np.sin(psi)
        )

        start_east = (
            forward_offset * np.sin(psi)
            + starboard_offset * np.cos(psi)
        )

        lg.target_ne_pos = [start_north, start_east]

        print(f"Starting DRL stationary tracking at N={start_north:.2f}, E={start_east:.2f}")

        try:
            while True:
                start_time = time.time()

                tau_X, tau_N = self.predict_control()

                if tau_X is None:
                    print("No valid DRL state -> drift")
                    lg.otter.drift()
                    time.sleep(0.2)
                    continue

                lg.otter.controller_inputs_torque(tau_X, tau_N, lg.surge_setpoint)

                lg.otter.sorted_values["tau_X"] = tau_X
                lg.otter.sorted_values["tau_N"] = tau_N
                lg.otter.sorted_values["target_north_from_observer"] = lg.target_ne_pos[0]
                lg.otter.sorted_values["target_east_from_observer"] = lg.target_ne_pos[1]

                current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
                temp_df = pd.DataFrame([lg.otter.sorted_values], index=[current_datetime])
                self.log = pd.concat([self.log, temp_df])

                elapsed_time = time.time() - start_time
                if elapsed_time < lg.cycletime:
                    time.sleep(lg.cycletime - elapsed_time)

        except KeyboardInterrupt:
            print("DRL tracking disabled. Otter is now in drift mode.")
            lg.otter.drift()