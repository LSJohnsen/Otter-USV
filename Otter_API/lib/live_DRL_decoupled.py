import os
import time
import datetime
import threading

import numpy as np
import pandas as pd
import gymnasium as gym

from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class DummyDRLEnv(gym.Env):
    """
    Dummy environment used only for loading VecNormalize.

    Must match the observation/action dimensions used during training:
        obs:    8 values
        action: 2 values
    """

    def __init__(self):
        super().__init__()

        self.observation_space = Box(
            low=np.array(
                [-1, -1, -1, -1, -1, -1, -1, -1],
                dtype=np.float32,
            ),
            high=np.array(
                [1, 1, 1, 1, 1, 1, 1, 1],
                dtype=np.float32,
            ),
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
    """
    Live PPO/DRL controller for the Otter.

    This class is designed to match the current training setup in Otter_dl.py.

    Observation:
        [
            x_rel / max_target_delta,
            y_rel / max_target_delta,
            yaw_rel / pi,
            u / Umax,
            v / Umax,
            r / Rmax,
            distance / max_target_delta,
            d_dot / Umax,
        ]

    Action:
        action[0] -> tau_X / tauX_max
        action[1] -> tau_N / tauN_max

    Live command:
        tau_X, tau_N -> otter.controller_inputs_torque(tau_X, tau_N)

    This follows the same live guidance architecture as PID/NMPC.
    """

    def __init__(
        self,
        live_guidance,
        model_path,
        vecnormalize_path,
        max_target_delta=250.0,
        Umax=6.0 * 0.5144,
        Rmax=2.0,
        tauX_max=150.0,
        tauN_max=110.0,
        scale_action=True,
        allow_reverse=True,
        tracking_only=False,
        live_tauX_limit=None,
        live_tauN_limit=None,
        min_send_interval=0.10,
        verbose=True,
    ):
        self.live_guidance = live_guidance

        # Must match training
        self.max_target_delta = float(max_target_delta)
        self.Umax = float(Umax)
        self.Rmax = float(Rmax)

        self.tauX_max = float(tauX_max)
        self.tauN_max = float(tauN_max)

        self.scale_action = bool(scale_action)

        # If tracking_only=True, reverse surge is blocked.
        # Use this for moving target/path tracking models.
        # For stationkeeping/DP models, keep allow_reverse=True and tracking_only=False.
        self.allow_reverse = bool(allow_reverse)
        self.tracking_only = bool(tracking_only)

        # Optional conservative live limits.
        # If None, use tauX_max/tauN_max.
        self.live_tauX_limit = (
            float(live_tauX_limit)
            if live_tauX_limit is not None
            else self.tauX_max
        )
        self.live_tauN_limit = (
            float(live_tauN_limit)
            if live_tauN_limit is not None
            else self.tauN_max
        )

        self.min_send_interval = float(min_send_interval)
        self.last_send_time = 0.0

        self.verbose = bool(verbose)

        # Distance memory for d_dot
        self.last_distance = None

        # Logging
        self.log = pd.DataFrame()

        # Make sure live_guidance has a lock, same pattern as NMPC/live code
        if not hasattr(self.live_guidance, "otter_lock"):
            self.live_guidance.otter_lock = threading.Lock()

        # Load VecNormalize
        dummy_env = DummyVecEnv([lambda: DummyDRLEnv()])
        self.vecnorm = VecNormalize.load(vecnormalize_path, dummy_env)
        self.vecnorm.training = False
        self.vecnorm.norm_reward = False

        # Load PPO model
        self.model = PPO.load(model_path, device="cpu")

    # ------------------------------------------------------------------
    # Observation and action
    # ------------------------------------------------------------------

    def make_observation(self):
        """
        Builds the exact same observation vector as the training environment.

        Uses live_guidance.current_state(), same as NMPC:
            state = [x, y, psi, u, v, r]

        target position:
            live_guidance.target_ne_pos = [target_north, target_east]
        """

        lg = self.live_guidance

        with lg.otter_lock:
            state = lg.current_state()

        if state is None:
            return None

        state = np.asarray(state, dtype=float)

        if state.shape[0] < 6 or not np.all(np.isfinite(state[:6])):
            print("DRL: invalid live state:", state)
            return None

        x, y, psi, u, v, r = state[:6]

        if not hasattr(lg, "target_ne_pos"):
            print("DRL: live_guidance.target_ne_pos is missing")
            return None

        target_pos = np.array(lg.target_ne_pos, dtype=float)

        if target_pos.shape[0] < 2 or not np.all(np.isfinite(target_pos[:2])):
            print("DRL: invalid target_ne_pos:", target_pos)
            return None

        target_pos = target_pos[:2]
        usv_pos = np.array([x, y], dtype=float)

        rel_pos = target_pos - usv_pos

        x_rel = float(rel_pos[0])
        y_rel = float(rel_pos[1])

        distance = float(np.linalg.norm(rel_pos))

        psi_los = float(np.arctan2(y_rel, x_rel))
        yaw_rel = float(wrap_to_pi(psi_los - psi))

        if self.last_distance is None:
            d_dot = 0.0
        else:
            dt = float(getattr(lg, "cycletime", 0.1))
            dt = max(dt, 1e-3)
            d_dot = (distance - self.last_distance) / dt

        self.last_distance = distance

        obs = np.array(
            [
                np.clip(x_rel / self.max_target_delta, -1.0, 1.0),
                np.clip(y_rel / self.max_target_delta, -1.0, 1.0),
                np.clip(yaw_rel / np.pi, -1.0, 1.0),
                np.clip(u / self.Umax, -1.0, 1.0),
                np.clip(v / self.Umax, -1.0, 1.0),
                np.clip(r / self.Rmax, -1.0, 1.0),
                np.clip(distance / self.max_target_delta, -1.0, 1.0),
                np.clip(d_dot / self.Umax, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

        if self.verbose:
            print(
                "DRL obs raw | "
                f"x_rel={x_rel:.2f}, y_rel={y_rel:.2f}, "
                f"yaw_rel={np.degrees(yaw_rel):.1f} deg, "
                f"u={u:.2f}, v={v:.2f}, r={np.degrees(r):.1f} deg/s, "
                f"distance={distance:.2f}, d_dot={d_dot:.2f}"
            )
            print("DRL obs:", obs)

        return obs

    def predict_control(self):
        """
        Runs PPO policy and converts normalized action to tau_X, tau_N.

        Matches training:
            action[0] * tauX_max -> tau_X
            action[1] * tauN_max -> tau_N
        """

        obs = self.make_observation()

        if obs is None:
            return None, None

        obs_vec = obs.reshape(1, -1)
        obs_norm = self.vecnorm.normalize_obs(obs_vec)

        action, _ = self.model.predict(obs_norm, deterministic=True)
        action = np.asarray(action, dtype=float).reshape(-1)

        if action.shape[0] < 2 or not np.all(np.isfinite(action[:2])):
            print("DRL: invalid action:", action)
            return None, None

        action = np.clip(action[:2], -1.0, 1.0)

        if self.scale_action:
            tau_X = float(action[0] * self.tauX_max)
            tau_N = float(action[1] * self.tauN_max)
        else:
            tau_X = float(action[0])
            tau_N = float(action[1])

        # Optional tracking behavior:
        # For moving-target tracking, avoid learning/using reverse as a shortcut.
        if self.tracking_only or not self.allow_reverse:
            tau_X = max(0.0, tau_X)

        # Conservative live clipping
        if self.tracking_only or not self.allow_reverse:
            tau_X = float(np.clip(tau_X, 0.0, self.live_tauX_limit))
        else:
            tau_X = float(np.clip(tau_X, -self.live_tauX_limit, self.live_tauX_limit))

        tau_N = float(np.clip(tau_N, -self.live_tauN_limit, self.live_tauN_limit))

        if self.verbose:
            print("DRL obs norm:", obs_norm)
            print("DRL raw action:", action)
            print(f"DRL tau_X={tau_X:.3f}, tau_N={tau_N:.3f}")

        return tau_X, tau_N

    # ------------------------------------------------------------------
    # Initialization and logging
    # ------------------------------------------------------------------

    def _initialize_tracking(self):
        """
        Initializes live Otter connection and local N/E observer frame.

        This follows the same pattern as live_guidance/NMPC:
            - connect
            - update values
            - set observer to current GPS position
            - update again
        """

        lg = self.live_guidance

        if not hasattr(lg, "otter_lock"):
            lg.otter_lock = threading.Lock()

        with lg.otter_lock:
            lg.otter.establish_connection(lg.ip, lg.port)
            lg.otter.update_values()

        lat = lg.otter.sorted_values.get("lat")
        lon = lg.otter.sorted_values.get("lon")

        if lat is None or lon is None:
            print("DRL: missing GPS initialization values")
            return False

        lg.referance_point = [float(lat), float(lon), 0.0]
        lg.otter.observer_coordinates = lg.referance_point

        with lg.otter_lock:
            lg.otter.update_values()

        self.last_distance = None
        self.last_send_time = 0.0

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
        self.log = pd.DataFrame([lg.otter.sorted_values], index=[current_datetime])

        print("DRL observer reference initialized:", lg.referance_point)

        return True

    def _update_reference_values(self):
        """
        Updates common reference/logging values on live_guidance and otter.sorted_values.
        This mirrors the useful parts of live_guidance.update_target_reference().
        """

        lg = self.live_guidance

        n_usv = float(lg.otter.sorted_values.get("north_from_observer", 0.0))
        e_usv = float(lg.otter.sorted_values.get("east_from_observer", 0.0))

        n_t = float(lg.target_ne_pos[0])
        e_t = float(lg.target_ne_pos[1])

        lg.north_error = n_t - n_usv
        lg.east_error = e_t - e_usv
        lg.distance_to_target = float(np.hypot(lg.north_error, lg.east_error))

        lg.yaw_setpoint = float(
            wrap_to_pi(np.arctan2(lg.east_error, lg.north_error))
        )

        lg.current_angle = float(
            wrap_to_pi(float(lg.otter.sorted_values.get("yaw_rad", 0.0)))
        )

        lg.heading_error = float(
            wrap_to_pi(lg.yaw_setpoint - lg.current_angle)
        )

        lg.otter.sorted_values["north_error"] = lg.north_error
        lg.otter.sorted_values["east_error"] = lg.east_error
        lg.otter.sorted_values["distance_to_target"] = lg.distance_to_target
        lg.otter.sorted_values["yaw_setpoint"] = lg.yaw_setpoint
        lg.otter.sorted_values["current_angle"] = lg.current_angle
        lg.otter.sorted_values["heading_error"] = lg.heading_error
        lg.otter.sorted_values["yaw_setpoint_deg"] = np.degrees(lg.yaw_setpoint)
        lg.otter.sorted_values["current_angle_deg"] = np.degrees(lg.current_angle)
        lg.otter.sorted_values["heading_error_deg"] = np.degrees(lg.heading_error)
        lg.otter.sorted_values["target_north_from_observer"] = n_t
        lg.otter.sorted_values["target_east_from_observer"] = e_t

    def _send_drl_control_and_log(self):
        """
        Predicts and sends DRL control through the same interface as PID/NMPC:
            otter.controller_inputs_torque(tau_X, tau_N)
        """

        lg = self.live_guidance

        tau_X, tau_N = self.predict_control()

        if tau_X is None or tau_N is None:
            print("No valid DRL state -> skipping control update")
            return False

        now = time.time()
        dt_send = now - self.last_send_time

        if dt_send < self.min_send_interval:
            time.sleep(self.min_send_interval - dt_send)

        self.last_send_time = time.time()

        if self.verbose:
            print(f"Sending DRL control: tau_X={tau_X:.3f}, tau_N={tau_N:.3f}")

        with lg.otter_lock:
            lg.otter.controller_inputs_torque(tau_X, tau_N)

        lg.otter.sorted_values["tau_X"] = tau_X
        lg.otter.sorted_values["tau_N"] = tau_N

        self._update_reference_values()

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
        temp_df = pd.DataFrame([lg.otter.sorted_values], index=[current_datetime])
        self.log = pd.concat([self.log, temp_df])

        return True

    def _save_log(self, logs_dir="../logs"):
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_drl.csv"
        file_path = os.path.join(logs_dir, filename)

        if len(self.log) > 0:
            self.log.to_csv(file_path, sep=";")
            print(f"DRL log saved to {file_path}")
        else:
            print("DRL log is empty; nothing saved.")

    # ------------------------------------------------------------------
    # Live path modes
    # ------------------------------------------------------------------

    def stationary_tracking(self, forward_offset=10.0, starboard_offset=5.0):
        """
        Stationary target relative to initial vessel heading.

        Same idea as live_guidance.stationary_target_tracking().
        """

        lg = self.live_guidance

        if not self._initialize_tracking():
            return

        psi = lg.get_initial_heading()

        start_north = (
            forward_offset * np.cos(psi)
            - starboard_offset * np.sin(psi)
        )

        start_east = (
            forward_offset * np.sin(psi)
            + starboard_offset * np.cos(psi)
        )

        lg.target_ne_pos = [float(start_north), float(start_east)]

        self._update_reference_values()

        print(
            f"Starting DRL stationary tracking at "
            f"N={start_north:.2f}, E={start_east:.2f}"
        )

        try:
            while True:
                start_time = time.time()

                self._send_drl_control_and_log()

                elapsed_time = time.time() - start_time
                dt_loop = float(getattr(lg, "cycletime", 0.1))

                if elapsed_time < dt_loop:
                    time.sleep(dt_loop - elapsed_time)

        except KeyboardInterrupt:
            print("DRL stationary tracking disabled. Otter is now in drift mode.")
            lg.otter.drift()
            self._save_log()

    def straight_tracking(self, start_north, start_east, v_north, v_east):
        """
        Straight moving target in N/E frame.

        This mirrors live_guidance.target_tracking().
        """

        lg = self.live_guidance

        if not self._initialize_tracking():
            return

        lg.target_ne_pos = [float(start_north), float(start_east)]

        self._update_reference_values()

        print(
            f"Starting DRL straight tracking. "
            f"Initial target N={start_north:.2f}, E={start_east:.2f}, "
            f"vN={v_north:.2f}, vE={v_east:.2f}"
        )

        try:
            while True:
                start_time = time.time()
                dt_loop = float(getattr(lg, "cycletime", 0.1))

                lg.target_ne_pos = [
                    float(lg.target_ne_pos[0] + v_north * dt_loop),
                    float(lg.target_ne_pos[1] + v_east * dt_loop),
                ]

                self._update_reference_values()
                self._send_drl_control_and_log()

                elapsed_time = time.time() - start_time

                if elapsed_time < dt_loop:
                    time.sleep(dt_loop - elapsed_time)

        except KeyboardInterrupt:
            print("DRL straight tracking disabled. Otter is now in drift mode.")
            lg.otter.drift()
            self._save_log()

    def circular_tracking(self, start_north, start_east, radius, v):
        """
        Circular moving target in N/E frame.

        This mirrors live_guidance.circular_tracking().
        """

        lg = self.live_guidance

        if not self._initialize_tracking():
            return

        if abs(radius) < 1e-6:
            print("DRL circular tracking: radius must be nonzero")
            return

        function_time = time.time()

        initial_north = start_north + radius
        initial_east = start_east

        lg.target_ne_pos = [float(initial_north), float(initial_east)]

        self._update_reference_values()

        print(
            f"Starting DRL circular tracking. "
            f"Center N={start_north:.2f}, E={start_east:.2f}, "
            f"radius={radius:.2f}, speed={v:.2f}"
        )

        try:
            while True:
                start_time = time.time()
                dt_loop = float(getattr(lg, "cycletime", 0.1))

                omega = float(v) / float(radius)
                theta = omega * (time.time() - function_time)

                lg.target_ne_pos = [
                    float(start_north + radius * np.cos(theta)),
                    float(start_east + radius * np.sin(theta)),
                ]

                self._update_reference_values()
                self._send_drl_control_and_log()

                elapsed_time = time.time() - start_time

                if elapsed_time < dt_loop:
                    time.sleep(dt_loop - elapsed_time)

        except KeyboardInterrupt:
            print("DRL circular tracking disabled. Otter is now in drift mode.")
            lg.otter.drift()
            self._save_log()

    def square_tracking(self, start_north, start_east, side_length, target_speed):
        """
        Square path target in N/E frame.

        This mirrors the live_guidance square target logic, but with cleaner
        direction switching.
        """

        lg = self.live_guidance

        if not self._initialize_tracking():
            return

        side_length = float(side_length)
        target_speed = float(target_speed)

        if side_length <= 0.0:
            print("DRL square tracking: side_length must be positive")
            return

        lg.target_ne_pos = [float(start_north), float(start_east)]

        self._update_reference_values()

        directions = [
            np.array([1.0, 0.0]),   # north
            np.array([0.0, 1.0]),   # east
            np.array([-1.0, 0.0]),  # south
            np.array([0.0, -1.0]),  # west
        ]

        direction_index = 0
        distance_on_side = 0.0

        print(
            f"Starting DRL square tracking. "
            f"Start N={start_north:.2f}, E={start_east:.2f}, "
            f"side_length={side_length:.2f}, speed={target_speed:.2f}"
        )

        try:
            while True:
                start_time = time.time()
                dt_loop = float(getattr(lg, "cycletime", 0.1))

                step_distance = target_speed * dt_loop
                distance_on_side += step_distance

                if distance_on_side >= side_length:
                    distance_on_side = 0.0
                    direction_index = (direction_index + 1) % len(directions)

                direction = directions[direction_index]

                lg.target_ne_pos = [
                    float(lg.target_ne_pos[0] + direction[0] * step_distance),
                    float(lg.target_ne_pos[1] + direction[1] * step_distance),
                ]

                self._update_reference_values()
                self._send_drl_control_and_log()

                elapsed_time = time.time() - start_time

                if elapsed_time < dt_loop:
                    time.sleep(dt_loop - elapsed_time)

        except KeyboardInterrupt:
            print("DRL square tracking disabled. Otter is now in drift mode.")
            lg.otter.drift()
            self._save_log()