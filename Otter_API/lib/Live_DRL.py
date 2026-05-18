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


# False = old 8-observation model:
# [x_rel, y_rel, yaw_rel, u, v, r, distance, d_dot]
#
# True = new 10-observation model:
# [x_rel, y_rel, yaw_rel, u, v, r, distance, d_dot, x_rel_dot, y_rel_dot]
USE_RELATIVE_VELOCITY_OBS = False


class DummyDRLEnv(gym.Env):
    def __init__(self, use_relative_velocity_obs=False):
        super().__init__()

        self.use_relative_velocity_obs = use_relative_velocity_obs
        self.obs_dim = 10 if self.use_relative_velocity_obs else 8

        self.observation_space = Box(
            low=-np.ones(self.obs_dim, dtype=np.float32),
            high=np.ones(self.obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
        )

    def reset(self, seed=None, options=None):
        return np.zeros(self.obs_dim, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.obs_dim, dtype=np.float32), 0.0, False, False, {}


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
        tauN_max=116.0,
        scale_action=True,
        scale_yaw_to_training_limit=True,
        yaw_training_limit=76.0,
        yaw_actual_limit=116.0,
        use_relative_velocity_obs=USE_RELATIVE_VELOCITY_OBS,
    ):
        self.live_guidance = live_guidance

        self.max_target_delta = max_target_delta
        self.Umax = Umax
        self.Rmax = Rmax

        self.tauX_max = tauX_max
        self.tauN_max = tauN_max
        self.scale_action = scale_action

        self.scale_yaw_to_training_limit = scale_yaw_to_training_limit
        self.yaw_training_limit = yaw_training_limit
        self.yaw_actual_limit = yaw_actual_limit

        if self.yaw_actual_limit <= 0.0:
            raise ValueError("yaw_actual_limit must be positive")

        self.yaw_scale_factor = self.yaw_training_limit / self.yaw_actual_limit

        self.use_relative_velocity_obs = use_relative_velocity_obs

        dummy_env = DummyVecEnv([
            lambda: DummyDRLEnv(
                use_relative_velocity_obs=self.use_relative_velocity_obs
            )
        ])

        self.vecnorm = VecNormalize.load(vecnormalize_path, dummy_env)
        self.vecnorm.training = False
        self.vecnorm.norm_reward = False

        self.model = PPO.load(model_path, device="cpu")

        self.last_distance = None
        self.prev_x_rel = None
        self.prev_y_rel = None

        self.min_send_interval = 0.15
        self.last_send_time = 0.0

        self.function_time = None

        self._ensure_otter_lock()
        self._ensure_live_guidance_log()

    def _ensure_otter_lock(self):
        lg = self.live_guidance

        if not hasattr(lg, "otter_lock"):
            import threading
            lg.otter_lock = threading.Lock()

    def _ensure_live_guidance_log(self):
        """
        DRL should use the same log buffer as Live_guidance.

        This assumes Live_guidance.save_log() saves lg.log.
        """
        lg = self.live_guidance

        if not hasattr(lg, "log") or lg.log is None:
            lg.log = pd.DataFrame()

    def _log_to_live_guidance(self):
        """
        Append one DRL sample to the same log used by PID/NMPC.

        The row is copied from lg.otter.sorted_values, so all values that
        should appear in the CSV must be written into sorted_values before
        this method is called.
        """
        lg = self.live_guidance
        self._ensure_live_guidance_log()

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")

        row = lg.otter.sorted_values.copy()
        temp_df = pd.DataFrame([row], index=[current_datetime])

        lg.log = pd.concat([lg.log, temp_df])

    def _set_target_values_in_sorted_values(self):
        lg = self.live_guidance

        lg.otter.sorted_values["target_north_from_observer"] = float(lg.target_ne_pos[0])
        lg.otter.sorted_values["target_east_from_observer"] = float(lg.target_ne_pos[1])

    def _compute_tracking_values(self):
        """
        Compute target-tracking values in the same convention as the PID/NMPC
        logging.

        Returns
        -------
        dict or None
            Dictionary with tracking values, or None if current state is invalid.
        """
        lg = self.live_guidance
        self._ensure_otter_lock()

        with lg.otter_lock:
            state = lg.current_state()

        if state is None:
            return None

        x, y, psi, u, v, r = state

        target_north = float(lg.target_ne_pos[0])
        target_east = float(lg.target_ne_pos[1])

        north_error = target_north - float(x)
        east_error = target_east - float(y)

        distance_to_target = float(np.hypot(north_error, east_error))

        yaw_setpoint = float(np.arctan2(east_error, north_error))
        current_angle = float(psi)
        heading_error = float(wrap_to_pi(yaw_setpoint - current_angle))

        return {
            "north_error": north_error,
            "east_error": east_error,
            "distance_to_target": distance_to_target,
            "yaw_setpoint": yaw_setpoint,
            "current_angle": current_angle,
            "heading_error": heading_error,
            "yaw_setpoint_deg": float(np.degrees(yaw_setpoint)),
            "current_angle_deg": float(np.degrees(current_angle)),
            "heading_error_deg": float(np.degrees(heading_error)),
            "target_north_from_observer": target_north,
            "target_east_from_observer": target_east,
            "drl_x_rel": north_error,
            "drl_y_rel": east_error,
            "drl_yaw_rel": heading_error,
            "drl_distance_to_target": distance_to_target,
        }

    def _update_tracking_values_in_sorted_values(self):
        """
        Write tracking values to lg.otter.sorted_values so they are saved in CSV.
        """
        lg = self.live_guidance
        values = self._compute_tracking_values()

        if values is None:
            return False

        for key, value in values.items():
            lg.otter.sorted_values[key] = value

        return True

    def make_observation(self):
        lg = self.live_guidance
        self._ensure_otter_lock()

        with lg.otter_lock:
            state = lg.current_state()

        if state is None:
            return None

        x, y, psi, u, v, r = state

        target_pos = np.array(lg.target_ne_pos, dtype=float)
        usv_pos = np.array([x, y], dtype=float)

        rel_pos = target_pos - usv_pos
        x_rel = float(rel_pos[0])
        y_rel = float(rel_pos[1])

        distance = float(np.linalg.norm(rel_pos))
        psi_los = float(np.arctan2(y_rel, x_rel))
        yaw_rel = wrap_to_pi(psi_los - psi)

        dt = float(getattr(lg, "cycletime", 0.1))
        if dt <= 1e-6:
            dt = 0.1

        if self.last_distance is None:
            d_dot = 0.0
        else:
            d_dot = (distance - self.last_distance) / dt

        if self.prev_x_rel is None or self.prev_y_rel is None:
            x_rel_dot = 0.0
            y_rel_dot = 0.0
        else:
            x_rel_dot = (x_rel - self.prev_x_rel) / dt
            y_rel_dot = (y_rel - self.prev_y_rel) / dt

        self.last_distance = distance
        self.prev_x_rel = x_rel
        self.prev_y_rel = y_rel

        x_rel_norm = np.clip(x_rel / self.max_target_delta, -1.0, 1.0)
        y_rel_norm = np.clip(y_rel / self.max_target_delta, -1.0, 1.0)
        yaw_rel_norm = np.clip(yaw_rel / np.pi, -1.0, 1.0)

        u_norm = np.clip(u / self.Umax, -1.0, 1.0)
        v_norm = np.clip(v / self.Umax, -1.0, 1.0)
        r_norm = np.clip(r / self.Rmax, -1.0, 1.0)

        distance_norm = np.clip(distance / self.max_target_delta, -1.0, 1.0)
        d_dot_norm = np.clip(d_dot / self.Umax, -1.0, 1.0)

        if self.use_relative_velocity_obs:
            x_rel_dot_norm = np.clip(x_rel_dot / self.Umax, -1.0, 1.0)
            y_rel_dot_norm = np.clip(y_rel_dot / self.Umax, -1.0, 1.0)

            obs = np.array([
                x_rel_norm,
                y_rel_norm,
                yaw_rel_norm,
                u_norm,
                v_norm,
                r_norm,
                distance_norm,
                d_dot_norm,
                x_rel_dot_norm,
                y_rel_dot_norm,
            ], dtype=np.float32)

        else:
            obs = np.array([
                x_rel_norm,
                y_rel_norm,
                yaw_rel_norm,
                u_norm,
                v_norm,
                r_norm,
                distance_norm,
                d_dot_norm,
            ], dtype=np.float32)

        # Also make sure these are available for logging.
        self._update_tracking_values_in_sorted_values()

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

        if self.scale_yaw_to_training_limit:
            tau_N = tau_N * self.yaw_scale_factor

        print("DRL obs:", obs)
        print("DRL obs norm:", obs_norm)
        print("DRL raw action:", action)
        print(f"DRL tau_X={tau_X:.3f}, tau_N={tau_N:.3f}")

        return tau_X, tau_N

    def _initialize_tracking(self):
        lg = self.live_guidance
        self._ensure_otter_lock()
        self._ensure_live_guidance_log()

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
        self.prev_x_rel = None
        self.prev_y_rel = None
        self.last_send_time = 0.0

        return True

    def _send_drl_control_and_log(self):
        lg = self.live_guidance
        self._ensure_otter_lock()

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

        self._set_target_values_in_sorted_values()
        self._update_tracking_values_in_sorted_values()
        self._log_to_live_guidance()

        return True

    def stationary_tracking(self, forward_offset=10.0, starboard_offset=5.0):
        lg = self.live_guidance

        if not self._initialize_tracking():
            return

        print("Available sorted_values keys:")
        print(list(lg.otter.sorted_values.keys()))

        print("Full sorted_values:")
        print(lg.otter.sorted_values)

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
        self._set_target_values_in_sorted_values()
        self._update_tracking_values_in_sorted_values()
        self._log_to_live_guidance()

        print(
            f"Starting DRL stationary tracking at "
            f"N={start_north:.2f}, E={start_east:.2f}"
        )

        try:
            while True:
                start_time = time.time()

                self._send_drl_control_and_log()

                elapsed_time = time.time() - start_time
                if elapsed_time < lg.cycletime:
                    time.sleep(lg.cycletime - elapsed_time)

        except KeyboardInterrupt:
            print("DRL stationary tracking disabled. Otter is now in drift mode.")
            lg.otter.drift()

    def straight_tracking(
        self,
        forward_offset=10.0,
        starboard_offset=0.0,
        v_forward=0.0,
        v_starboard=1.5,
        use_ref_model=None,
    ):
        lg = self.live_guidance

        if use_ref_model is None:
            use_ref_model = getattr(lg, "target_ref", False)

        if not self._initialize_tracking():
            return

        psi0 = lg.get_initial_heading()

        start_north = (
            forward_offset * np.cos(psi0)
            - starboard_offset * np.sin(psi0)
        )

        start_east = (
            forward_offset * np.sin(psi0)
            + starboard_offset * np.cos(psi0)
        )

        v_north = (
            v_forward * np.cos(psi0)
            - v_starboard * np.sin(psi0)
        )

        v_east = (
            v_forward * np.sin(psi0)
            + v_starboard * np.cos(psi0)
        )

        lg.target_ne_pos = [start_north, start_east]
        self._set_target_values_in_sorted_values()

        if hasattr(lg, "ref_dist"):
            lg.ref_dist = float(np.hypot(start_north, start_east))
            lg.ref_dist_dot = 0.0
            lg.ref_dist_ddot = 0.0

        if hasattr(lg, "update_target_reference"):
            lg.update_target_reference(use_ref_model)

        self._update_tracking_values_in_sorted_values()
        self._log_to_live_guidance()

        with lg.otter_lock:
            lg.otter.controller_inputs_torque(10, 0)
        time.sleep(2)

        with lg.otter_lock:
            lg.otter.controller_inputs_torque(10, 0)
        time.sleep(1)

        print(
            f"Starting DRL straight tracking from heading-relative target. "
            f"Initial target N={start_north:.2f} m, E={start_east:.2f} m, "
            f"vN={v_north:.2f} m/s, vE={v_east:.2f} m/s, "
            f"psi0={np.degrees(psi0):.1f} deg"
        )

        try:
            while True:
                start_time = time.time()

                lg.target_ne_pos = [
                    lg.target_ne_pos[0] + v_north * lg.cycletime,
                    lg.target_ne_pos[1] + v_east * lg.cycletime,
                ]

                self._set_target_values_in_sorted_values()

                if hasattr(lg, "update_target_reference"):
                    lg.update_target_reference(use_ref_model)

                self._send_drl_control_and_log()

                elapsed_time = time.time() - start_time
                if elapsed_time < lg.cycletime:
                    time.sleep(lg.cycletime - elapsed_time)

        except KeyboardInterrupt:
            print("DRL straight tracking disabled. Otter is now in drift mode.")
            lg.otter.drift()

    def circular_tracking(
        self,
        start_north,
        start_east,
        radius,
        v,
        use_ref_model=None,
    ):
        lg = self.live_guidance

        if use_ref_model is None:
            use_ref_model = getattr(lg, "target_ref", False)

        if not self._initialize_tracking():
            return

        self.function_time = time.time()

        initial_north = start_north + radius
        initial_east = start_east

        lg.target_ne_pos = [initial_north, initial_east]
        self._set_target_values_in_sorted_values()

        if hasattr(lg, "ref_dist"):
            lg.ref_dist = float(np.hypot(initial_north, initial_east))
            lg.ref_dist_dot = 0.0
            lg.ref_dist_ddot = 0.0

        if hasattr(lg, "update_target_reference"):
            lg.update_target_reference(use_ref_model)

        self._update_tracking_values_in_sorted_values()
        self._log_to_live_guidance()

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

                self._set_target_values_in_sorted_values()

                if hasattr(lg, "update_target_reference"):
                    lg.update_target_reference(use_ref_model)

                self._send_drl_control_and_log()

                elapsed_time = time.time() - start_time
                if elapsed_time < lg.cycletime:
                    time.sleep(lg.cycletime - elapsed_time)

        except KeyboardInterrupt:
            print("DRL circular tracking disabled. Otter is now in drift mode.")
            lg.otter.drift()

    def _square_target_position(
        self,
        start_north,
        start_east,
        side_length,
        v,
        elapsed_time,
    ):
        if side_length <= 0:
            return start_north, start_east

        if v <= 0:
            return start_north, start_east

        perimeter = 4.0 * side_length
        travelled = (v * elapsed_time) % perimeter

        if travelled < side_length:
            north = start_north
            east = start_east + travelled

        elif travelled < 2.0 * side_length:
            s = travelled - side_length
            north = start_north + s
            east = start_east + side_length

        elif travelled < 3.0 * side_length:
            s = travelled - 2.0 * side_length
            north = start_north + side_length
            east = start_east + side_length - s

        else:
            s = travelled - 3.0 * side_length
            north = start_north + side_length - s
            east = start_east

        return north, east

    def square_tracking(
        self,
        start_north,
        start_east,
        side_length,
        v,
        use_ref_model=None,
    ):
        lg = self.live_guidance

        if use_ref_model is None:
            use_ref_model = getattr(lg, "target_ref", False)

        if not self._initialize_tracking():
            return

        self.function_time = time.time()

        lg.target_ne_pos = [start_north, start_east]
        self._set_target_values_in_sorted_values()

        if hasattr(lg, "ref_dist"):
            lg.ref_dist = float(np.hypot(start_north, start_east))
            lg.ref_dist_dot = 0.0
            lg.ref_dist_ddot = 0.0

        if hasattr(lg, "update_target_reference"):
            lg.update_target_reference(use_ref_model)

        self._update_tracking_values_in_sorted_values()
        self._log_to_live_guidance()

        print(
            f"Starting DRL square tracking. "
            f"Start N={start_north:.2f}, E={start_east:.2f}, "
            f"side_length={side_length:.2f}, speed={v:.2f}"
        )

        try:
            while True:
                start_time = time.time()

                elapsed_square_time = time.time() - self.function_time

                target_north, target_east = self._square_target_position(
                    start_north=start_north,
                    start_east=start_east,
                    side_length=side_length,
                    v=v,
                    elapsed_time=elapsed_square_time,
                )

                lg.target_ne_pos = [target_north, target_east]
                self._set_target_values_in_sorted_values()

                if hasattr(lg, "update_target_reference"):
                    lg.update_target_reference(use_ref_model)

                self._send_drl_control_and_log()

                elapsed_time = time.time() - start_time
                if elapsed_time < lg.cycletime:
                    time.sleep(lg.cycletime - elapsed_time)

        except KeyboardInterrupt:
            print("DRL square tracking disabled. Otter is now in drift mode.")
            lg.otter.drift()