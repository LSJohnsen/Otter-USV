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
        scale_action=True,
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

        self.min_send_interval = 0.15
        self.last_send_time = 0.0

    def make_observation(self):
        lg = self.live_guidance

        if not hasattr(lg, "otter_lock"):
            import threading
            lg.otter_lock = threading.Lock()

        with lg.otter_lock:
            state = lg.current_state()

        if state is None:
            return None

        x, y, psi, u, v, r = state

        target_pos = np.array(lg.target_ne_pos, dtype=float)
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
            d_dot = (distance - self.last_distance) / lg.cycletime

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

        print("DRL obs:", obs)
        print("DRL obs norm:", obs_norm)
        print("DRL raw action:", action)
        print(f"DRL tau_X={tau_X:.3f}, tau_N={tau_N:.3f}")

        return tau_X, tau_N

    def stationary_tracking(self, forward_offset=10.0, starboard_offset=5.0):
        lg = self.live_guidance

        lg.otter.establish_connection(lg.ip, lg.port)
        lg.otter.update_values()

        print("Available sorted_values keys:")
        print(list(lg.otter.sorted_values.keys()))

        print("Full sorted_values:")
        print(lg.otter.sorted_values)

        lat = lg.otter.sorted_values.get("lat")
        lon = lg.otter.sorted_values.get("lon")

        if lat is None or lon is None:
            print("DRL: missing GPS initialization values")
            return

        lg.referance_point = [lat, lon, 0.0]
        lg.otter.observer_coordinates = lg.referance_point

        lg.otter.update_values()

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

                if tau_X is None or tau_N is None:
                    print("No valid DRL state -> skipping control update")
                    time.sleep(0.2)
                    continue

                print(f"Sending DRL control: tau_X={tau_X:.3f}, tau_N={tau_N:.3f}")

                with lg.otter_lock:
                    lg.otter.controller_inputs_torque(tau_X, tau_N)

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

    def _initialize_tracking(self):
        lg = self.live_guidance

        lg.otter.establish_connection(lg.ip, lg.port)
        lg.otter.update_values()

        lat = lg.otter.sorted_values.get("lat")
        lon = lg.otter.sorted_values.get("lon")

        if lat is None or lon is None:
            print("DRL: missing GPS initialization values")
            return False

        lg.referance_point = [lat, lon, 0.0]
        lg.otter.observer_coordinates = lg.referance_point
        lg.otter.update_values()

        self.last_distance = None

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
        self.log = pd.DataFrame([lg.otter.sorted_values], index=[current_datetime])

        return True

    def _send_drl_control_and_log(self):
        lg = self.live_guidance

        tau_X, tau_N = self.predict_control()

        if tau_X is None or tau_N is None:
            print("No valid DRL state -> skipping control update")
            return False

        print(f"Sending DRL control: tau_X={tau_X:.3f}, tau_N={tau_N:.3f}")

        now = time.time()
        dt_send = now - self.last_send_time

        if dt_send < self.min_send_interval:
            time.sleep(self.min_send_interval - dt_send)

        self.last_send_time = time.time()

        with lg.otter_lock:
            lg.otter.controller_inputs_torque(tau_X, tau_N)

        lg.otter.sorted_values["tau_X"] = tau_X
        lg.otter.sorted_values["tau_N"] = tau_N
        lg.otter.sorted_values["target_north_from_observer"] = lg.target_ne_pos[0]
        lg.otter.sorted_values["target_east_from_observer"] = lg.target_ne_pos[1]

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
        temp_df = pd.DataFrame([lg.otter.sorted_values], index=[current_datetime])
        self.log = pd.concat([self.log, temp_df])

        return True

    def _save_log(self):
        logs_dir = "../logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_drl.csv"
        file_path = os.path.join(logs_dir, filename)
        self.log.to_csv(file_path, sep=";")
        print(f"DRL log saved to {file_path}")

    def straight_tracking(self, start_north, start_east, v_north, v_east, use_ref_model=None):
        lg = self.live_guidance

        if use_ref_model is None:
            use_ref_model = getattr(lg, "target_ref", False)

        if not self._initialize_tracking():
            return

        lg.target_ne_pos = [start_north, start_east]

        if hasattr(lg, "ref_dist"):
            lg.ref_dist = float(np.hypot(start_north, start_east))
            lg.ref_dist_dot = 0.0
            lg.ref_dist_ddot = 0.0

        if hasattr(lg, "update_target_reference"):
            lg.update_target_reference(use_ref_model)

        print(
            f"Starting DRL straight tracking. "
            f"Initial target N={start_north:.2f}, E={start_east:.2f}, "
            f"vN={v_north:.2f}, vE={v_east:.2f}"
        )

        try:
            while True:
                start_time = time.time()

                lg.target_ne_pos = [
                    lg.target_ne_pos[0] + v_north * lg.cycletime,
                    lg.target_ne_pos[1] + v_east * lg.cycletime,
                ]

                if hasattr(lg, "update_target_reference"):
                    lg.update_target_reference(use_ref_model)

                self._send_drl_control_and_log()

                elapsed_time = time.time() - start_time
                if elapsed_time < lg.cycletime:
                    time.sleep(lg.cycletime - elapsed_time)

        except KeyboardInterrupt:
            print("DRL straight tracking disabled. Otter is now in drift mode.")
            lg.otter.drift()
            self._save_log()

    def circular_tracking(self, start_north, start_east, radius, v, use_ref_model=None):
        lg = self.live_guidance

        if use_ref_model is None:
            use_ref_model = getattr(lg, "target_ref", False)

        if not self._initialize_tracking():
            return

        self.function_time = time.time()

        initial_north = start_north + radius
        initial_east = start_east
        lg.target_ne_pos = [initial_north, initial_east]

        if hasattr(lg, "ref_dist"):
            lg.ref_dist = float(np.hypot(initial_north, initial_east))
            lg.ref_dist_dot = 0.0
            lg.ref_dist_ddot = 0.0

        if hasattr(lg, "update_target_reference"):
            lg.update_target_reference(use_ref_model)

        print(
            f"Starting DRL circular tracking. "
            f"Center N={start_north:.2f}, E={start_east:.2f}, "
            f"radius={radius:.2f}, speed={v:.2f}"
        )

        try:
            while True:
                start_time = time.time()

                omega = v / radius
                theta = omega * (time.time() - self.function_time)

                lg.target_ne_pos = [
                    start_north + radius * np.cos(theta),
                    start_east + radius * np.sin(theta),
                ]

                if hasattr(lg, "update_target_reference"):
                    lg.update_target_reference(use_ref_model)

                self._send_drl_control_and_log()

                elapsed_time = time.time() - start_time
                if elapsed_time < lg.cycletime:
                    time.sleep(lg.cycletime - elapsed_time)

        except KeyboardInterrupt:
            print("DRL circular tracking disabled. Otter is now in drift mode.")
            lg.otter.drift()
            self._save_log()


    def square_tracking(self, start_north, start_east, side_length, v, use_ref_model=None):
        lg = self.live_guidance

        if use_ref_model is None:
            use_ref_model = getattr(lg, "target_ref", False)

        if not self._initialize_tracking():
            return

        corners = [
            np.array([start_north, start_east], dtype=float),
            np.array([start_north + side_length, start_east], dtype=float),
            np.array([start_north + side_length, start_east + side_length], dtype=float),
            np.array([start_north, start_east + side_length], dtype=float),
        ]

        corner_idx = 0
        lg.target_ne_pos = corners[corner_idx].tolist()

        if hasattr(lg, "update_target_reference"):
            lg.update_target_reference(use_ref_model)

        print(
            f"Starting DRL square tracking. "
            f"Start N={start_north:.2f}, E={start_east:.2f}, "
            f"side={side_length:.2f}, speed={v:.2f}"
        )

        try:
            while True:
                start_time = time.time()

                current_target = np.array(lg.target_ne_pos, dtype=float)
                next_corner = corners[corner_idx]

                delta = next_corner - current_target
                distance = float(np.linalg.norm(delta))

                step = v * lg.cycletime

                if distance <= step or distance < 1e-6:
                    lg.target_ne_pos = next_corner.tolist()
                    corner_idx = (corner_idx + 1) % len(corners)
                else:
                    direction = delta / distance
                    lg.target_ne_pos = (current_target + step * direction).tolist()

                if hasattr(lg, "update_target_reference"):
                    lg.update_target_reference(use_ref_model)

                self._send_drl_control_and_log()

                elapsed_time = time.time() - start_time
                if elapsed_time < lg.cycletime:
                    time.sleep(lg.cycletime - elapsed_time)

        except KeyboardInterrupt:
            print("DRL square tracking disabled. Otter is now in drift mode.")
            lg.otter.drift()
            self._save_log()