import numpy as np
import time
import math
import datetime
import pandas as pd
import os
import requests
import threading
import pymap3d as pm

class live_guidance():

    def __init__(self, ip, port, surge_PID, yaw_PID, surge_setpoint, otter, nmpc=None, use_nmpc=False, control_dt=0.1, third_order_ref=True):

        self.ip = ip
        self.port = port

        self.surge_PID = surge_PID
        self.yaw_PID = yaw_PID

        self.otter = otter

        self.surge_setpoint = surge_setpoint

        self.nmpc = nmpc
        self.use_nmpc = use_nmpc
        self.control_dt = control_dt 
        self.target_ref = third_order_ref

        self.distance_to_target = 0
        self.north_error = 0
        self.east_error = 0
        self.current_angle = 0
        self.function_time = 0

        self.max_force = 200
        self.cycletime = 0.1
        self.referance_point = [0, 0, 0]
        self.target = [0, 0]
        self.otter_ned_pos = [0, 0]
        self.total_distance_to_target = 0.0
        self.counter = 0
        self.yaw_setpoint = 0.0 #for pid heading
        self.last_valid_state = None

        self._ugps_lock = threading.Lock()
        self._ugps_latest = None  # lat, lon, depth, x, y, z, t

        self.ref_dist = 0.0
        self.ref_dist_dot = 0.0
        self.ref_dist_ddot = 0.0

        # test zeta/omega for surge reference
        self.zeta_ref = 0.9
        self.omega_n_ref = 0.6 #0.6

        self.nmpc_state_initialized = False

        self.otter_lock = threading.Lock()
            
    # filter signals by signals are floats and a finite values
    def _confirm_signal(self, key, default=None):
        value = self.otter.sorted_values.get(key, default)

        try:
            value = float(value)
        except (TypeError, ValueError):
            return default

        if not math.isfinite(value):
            return default

        return value

        
    # Filter signals to previous value if the new signal is larger than specified difference limit (prevent nmpc crash)
    def _filter_signal(self, new_values, previous_values, difference_limit=None, relative_limit=None):
        if previous_values is None:
            return new_values
        value_diff = new_values - previous_values 

        # Return previous reading if change is too high (hard limit) (gps errors)
        if difference_limit is not None and abs(value_diff) > difference_limit:
            return previous_values

        # Return previous reading if change is too large relative to the previous reading
        if relative_limit is not None:
            previous = max(abs(previous_values), 1e-6)
            if abs(value_diff) / previous > relative_limit:
                return previous_values
        
        return new_values

    # UGPS agnostic
    def ugps_get(self, url):
        try:
            r = requests.get(url, timeout=0.3)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    # UGPS read
    def ugps_reader(self, stop_event, URL_GLOBAL, URL_ACOUSTIC, otter=None):
        print("Starting UGPS reader thread...\n")
        session = requests.Session()
        i = 0

        while not stop_event.is_set():
            try:
                g = session.get(URL_GLOBAL, timeout=(0.15, 0.15))
                a = session.get(URL_ACOUSTIC, timeout=(0.15, 0.15))

                if g.status_code == 200 and a.status_code == 200:
                    g = g.json()
                    a = a.json()
                else:
                    g, a = None, None

            except requests.exceptions.RequestException:
                g, a = None, None

            if g and a:
                lat = float(g.get("lat", 0.0))
                lon = float(g.get("lon", 0.0))
                depth = -float(a.get("z", 0.0))
                x = float(a.get("x", 0.0))
                y = float(a.get("y", 0.0))
                z = float(a.get("z", 0.0))

                # latest UGPS 
                with self._ugps_lock:
                    self._ugps_latest = {
                        "lat": lat, "lon": lon, "depth": depth,
                        "x": x, "y": y, "z": z,
                        "t": time.time()
                    }

              
                if otter is not None:
                    otter.sorted_values["ugps_lat"] = lat
                    otter.sorted_values["ugps_lon"] = lon
                    otter.sorted_values["ugps_depth"] = depth
                    otter.sorted_values["ugps_x"] = x
                    otter.sorted_values["ugps_y"] = y
                    otter.sorted_values["ugps_z"] = z

                if i % 10 == 0:
                    print(f"Global:  Lat:{lat:.6f}, Lon:{lon:.6f}, Depth:{depth:.2f} m")
                    print(f"Local XYZ: X:{x:.2f} m, Y:{y:.2f} m, Z:{z:.2f} m")
                    print("-" * 40)
                i += 1
            else:
                print("Waiting for valid UGPS data...")
                pass

            time.sleep(0.5)
    
    # Update USV states - computes u,v in body from the heading and north/east change
    def current_state(self):
        """
        Reads the current Otter state for NMPC.

        State format:
            [x, y, psi, u, v, r]

        where:
            x, y  = N/E position relative to observer [m]
            psi   = yaw angle [rad]
            u, v  = body-frame surge/sway velocity [m/s]
            r     = yaw rate [rad/s]

        Debug version:
            - No aggressive filtering of x, y, psi
            - Uses latest available API values directly
            - Converts yaw rate from deg/s to rad/s
        """

        self.otter.update_values()

        # Position and heading
        x_raw = self._confirm_signal("north_from_observer", None)
        y_raw = self._confirm_signal("east_from_observer", None)
        psi_raw = self._get_heading_signal()

        # Inertial N/E velocities
        v_n_raw = self._confirm_signal("speed_n", 0.0)
        v_e_raw = self._confirm_signal("speed_e", 0.0)

        # Yaw rate from IMU
        r_raw = self._confirm_signal("current_rotational_velocities_3", 0.0)

        # Required values
        if x_raw is None or y_raw is None or psi_raw is None:
            print(
                "NMPC state missing | "
                f"x={x_raw}, y={y_raw}, psi={psi_raw}, "
                f"vn={v_n_raw}, ve={v_e_raw}, r={r_raw}"
            )

            if self.last_valid_state is not None:
                return self.last_valid_state

            return None

        # Convert to float
        x = float(x_raw)
        y = float(y_raw)
        psi = self.wrap_to_pi(float(psi_raw))

        v_n = float(v_n_raw)
        v_e = float(v_e_raw)

        # IMU angular velocity is assumed deg/s -> convert to rad/s
        r = math.radians(float(r_raw))

        # Convert inertial N/E velocity to body-frame velocity
        u = math.cos(psi) * v_n + math.sin(psi) * v_e
        v = -math.sin(psi) * v_n + math.cos(psi) * v_e

        state = np.array([x, y, psi, u, v, r], dtype=float)

        print(
            "RAW NMPC state | "
            f"x={x:.2f}, y={y:.2f}, "
            f"psi={math.degrees(psi):.1f} deg, "
            f"vn={v_n:.3f}, ve={v_e:.3f}, "
            f"u={u:.3f}, v={v:.3f}, "
            f"r={math.degrees(r):.1f} deg/s"
        )

        self.last_valid_state = state
        return state
    
    # UGPS target tracking
    def ugps_target_tracking(self, stop_event, ugps_timeout_s=2, logs_dir="../logs"):
        """
        Tracks a live target using latest UGPS geo2ned instead of simulated target.
        (ENSURE INERTIAL FRAME COORDINATES IN OTTER_API.PY AND WATERLINKED ARE THE SAME FOR CORRECT CALCULATIONS)
        Requires ugps_reader to be running in a separate thread and updating self._ugps_latest (could cause issues if irregular updates? test)
        ugps_timeout_s if no fresh UGPS data arrives hold or drift
        """

        self.otter.establish_connection(self.ip, self.port)
        self.otter.update_values()

        # Reference point for coordinate conversions / observer coordinates
        self.referance_point = [
            self.otter.sorted_values["lat"],
            self.otter.sorted_values["lon"],
            0.0
        ]
        self.otter.observer_coordinates = self.referance_point

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
        self.log = pd.DataFrame([self.otter.sorted_values], index=[current_datetime])

        # initial thrust
        self.otter.controller_inputs_torque(10, 0)
        time.sleep(2)
        self.otter.controller_inputs_torque(10, 0)
        time.sleep(1)

        mode = "NMPC" if self.use_nmpc else "PID"
        print(f"Starting {mode} UGPS target tracking (live).")

        try:
            while not stop_event.is_set():
                start_time = time.time()

                # latest UGPS from separate reader thread
                with self._ugps_lock:
                    ugps = None if self._ugps_latest is None else dict(self._ugps_latest)

                # drift if no signal
                if (ugps is None) or ((time.time() - ugps["t"]) > ugps_timeout_s):
                    print("UGPS missing -> drift mode")
                    self.otter.drift()
                    time.sleep(0.5)
                    continue

                # target position from UGPS local coordinates
                '''
                target_north = ugps["x"]
                target_east  = ugps["y"]
                '''
                target_north, target_east, target_down = self.ugps_geo_to_ned(
                    ugps["lat"],
                    ugps["lon"]
                )

                target_north = float(target_north)
                target_east = float(target_east)

                self.otter.sorted_values["target_north_from_observer"] = target_north
                self.otter.sorted_values["target_east_from_observer"] = target_east

                # Target position relative to observer frame
                self.target_ne_pos = [target_north, target_east]

                # Current USV position in observer frame
                n_usv = float(self.otter.sorted_values.get("north_from_observer", 0.0))
                e_usv = float(self.otter.sorted_values.get("east_from_observer", 0.0))

                # Error from USV to target
                self.north_error = target_north - n_usv
                self.east_error = target_east - e_usv

                # Estimate target velocity from consecutive UGPS target positions
                now = time.time()

                if not hasattr(self, "_last_ugps_target"):
                    v_north = 0.0
                    v_east = 0.0
                else:
                    last_n, last_e, last_t = self._last_ugps_target
                    dt_vel = max(now - last_t, 1e-6)

                    v_north = (target_north - last_n) / dt_vel
                    v_east = (target_east - last_e) / dt_vel

                self.target_v_ne = [float(v_north), float(v_east)]

                if np.hypot(v_north, v_east) > 1e-6:
                    self.path_heading = self.wrap_to_pi(
                        math.atan2(v_east, v_north)
                    )
                else:
                    self.path_heading = self.wrap_to_pi(
                        math.atan2(self.east_error, self.north_error)
                    )

                self._last_ugps_target = (target_north, target_east, now)

                ''' if ref model doesn't work:
                self.distance_to_target = float(np.hypot(self.north_error, self.east_error))

                self.current_angle = float(np.arctan2(self.east_error, self.north_error))
                self.yaw_setpoint = self.current_angle
                '''

                # error setpoint from reference model position/velocity/acceleration
                raw_dist = float(np.hypot(self.north_error, self.east_error))

                self.ref_dist, self.ref_dist_dot, self.ref_dist_ddot = \
                    self.third_order_reference(
                        self.ref_dist,
                        self.ref_dist_dot,
                        self.ref_dist_ddot,
                        raw_dist,
                        self.zeta_ref,
                        self.omega_n_ref,
                        self.cycletime
                    )

                self.distance_to_target = self.ref_dist

                self.yaw_setpoint = self.wrap_to_pi(
                    math.atan2(self.east_error, self.north_error)
                )

                self.current_angle = self.wrap_to_pi(
                    float(self.otter.sorted_values.get("yaw_rad", 0.0))
                )

                self.heading_error = self.wrap_to_pi(
                    self.yaw_setpoint - self.current_angle
                )

                # Compute control
                if self.use_nmpc:
                    tau_X, tau_N = self.calculate_forces_nmpc()
                else:
                    tau_X, tau_N = self.calculate_forces_pid()

                # Send control
                self.otter.controller_inputs_torque(tau_X, tau_N, self.surge_setpoint)

                # Logging
                self.otter.sorted_values["north_error"] = self.north_error
                self.otter.sorted_values["east_error"] = self.east_error
                self.otter.sorted_values["distance_to_target"] = self.distance_to_target
                self.otter.sorted_values["yaw_setpoint"] = self.yaw_setpoint
                self.otter.sorted_values["current_angle"] = self.current_angle
                self.otter.sorted_values["heading_error"] = self.heading_error
                self.otter.sorted_values["current_angle_deg"] = math.degrees(self.current_angle)
                self.otter.sorted_values["yaw_setpoint_deg"] = math.degrees(self.yaw_setpoint)
                self.otter.sorted_values["heading_error_deg"] = math.degrees(self.heading_error)
                self.otter.sorted_values["tau_X"] = tau_X
                self.otter.sorted_values["tau_N"] = tau_N
                self.otter.sorted_values["target_v_north"] = self.target_v_ne[0]
                self.otter.sorted_values["target_v_east"] = self.target_v_ne[1]
                self.otter.sorted_values["path_heading"] = self.path_heading
                self.otter.sorted_values["path_heading_deg"] = math.degrees(self.path_heading)

                current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
                temp_df = pd.DataFrame([self.otter.sorted_values], index=[current_datetime])
                self.log = pd.concat([self.log, temp_df])

                self.counter += 1
                self.total_distance_to_target += self.distance_to_target

                # Rate control
                elapsed_time = time.time() - start_time
                if elapsed_time < self.cycletime:
                    time.sleep(self.cycletime - elapsed_time)

        except KeyboardInterrupt:
            print("Tracking disabled. Otter is now in drift mode")
            self.otter.drift()

        finally:
            # Save log on exit
            if logs_dir:
                os.makedirs(logs_dir, exist_ok=True)
                filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_ugps.csv"
                file_path = os.path.join(logs_dir, filename)
                self.log.to_csv(file_path, sep=";")

    #straight simulated target
    # straight simulated target
    def target_tracking(
        self,
        forward_offset=10.0,
        starboard_offset=0.0,
        v_forward=1.0,
        v_starboard=0.0,
        use_ref_model=None
    ):
        if use_ref_model is None:
            use_ref_model = self.target_ref

        self.otter.establish_connection(self.ip, self.port)
        self.otter.update_values()

        self.referance_point = [
            self.otter.sorted_values["lat"],
            self.otter.sorted_values["lon"],
            0.0
        ]
        self.otter.observer_coordinates = self.referance_point

        # Initialize target based on the USV's initial heading.
        # Convention:
        #   forward_offset   = target forward of vessel
        #   starboard_offset = target to starboard/right of vessel
        psi0 = self.get_initial_heading()

        start_north = (
            forward_offset * np.cos(psi0)
            - starboard_offset * np.sin(psi0)
        )

        start_east = (
            forward_offset * np.sin(psi0)
            + starboard_offset * np.cos(psi0)
        )

        # Convert body-relative target velocity to fixed N/E velocity
        # using the initial USV heading.
        v_north = (
            v_forward * np.cos(psi0)
            - v_starboard * np.sin(psi0)
        )

        v_east = (
            v_forward * np.sin(psi0)
            + v_starboard * np.cos(psi0)
        )

        self.target_v_ne = [float(v_north), float(v_east)]

        if np.hypot(v_north, v_east) > 1e-6:
            self.path_heading = self.wrap_to_pi(math.atan2(v_east, v_north))
        else:
            self.path_heading = psi0

        self.target_ne_pos = [start_north, start_east]
        

        self.ref_dist = float(np.hypot(start_north, start_east))
        self.ref_dist_dot = 0.0
        self.ref_dist_ddot = 0.0

        self.update_target_reference(use_ref_model)

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
        self.log = pd.DataFrame([self.otter.sorted_values], index=[current_datetime])

        self.otter.controller_inputs_torque(10, 0)
        time.sleep(2)
        self.otter.controller_inputs_torque(10, 0)
        time.sleep(1)

        if self.use_nmpc:
            print(
                f"Starting NMPC straight tracking from heading-relative target. "
                f"Initial target N={start_north:.2f} m, E={start_east:.2f} m, "
                f"vN={v_north:.2f} m/s, vE={v_east:.2f} m/s, "
                f"psi0={math.degrees(psi0):.1f} deg"
            )
        else:
            print(
                f"Starting PID straight tracking from heading-relative target. "
                f"Initial target N={start_north:.2f} m, E={start_east:.2f} m, "
                f"vN={v_north:.2f} m/s, vE={v_east:.2f} m/s, "
                f"psi0={math.degrees(psi0):.1f} deg"
            )

        try:
            while True:
                start_time = time.time()

                # Move the target along a fixed straight line defined by
                # the USV's initial heading.
                self.target_ne_pos = [
                    self.target_ne_pos[0] + v_north * self.cycletime,
                    self.target_ne_pos[1] + v_east * self.cycletime,
                ]

                # Update errors / distance / yaw before controller
                self.update_target_reference(use_ref_model)

                # Compute control using updated target values
                if self.use_nmpc:
                    tau_X, tau_N = self.calculate_forces_nmpc()
                else:
                    tau_X, tau_N = self.calculate_forces_pid(stationary_tracking=False)

                # Send torques to Otter
                self.otter.controller_inputs_torque(
                    tau_X,
                    tau_N,
                    self.surge_setpoint
                )

                # Logging
                self.otter.sorted_values["north_error"] = self.north_error
                self.otter.sorted_values["east_error"] = self.east_error
                self.otter.sorted_values["distance_to_target"] = self.distance_to_target
                self.otter.sorted_values["yaw_setpoint"] = self.yaw_setpoint
                self.otter.sorted_values["current_angle"] = self.current_angle
                self.otter.sorted_values["heading_error"] = self.heading_error
                self.otter.sorted_values["yaw_setpoint_deg"] = math.degrees(self.yaw_setpoint)
                self.otter.sorted_values["current_angle_deg"] = math.degrees(self.current_angle)
                self.otter.sorted_values["heading_error_deg"] = math.degrees(self.heading_error)

                self.otter.sorted_values["tau_X"] = tau_X
                self.otter.sorted_values["tau_N"] = tau_N

                self.otter.sorted_values["target_north_from_observer"] = self.target_ne_pos[0]
                self.otter.sorted_values["target_east_from_observer"] = self.target_ne_pos[1]

                current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
                temp_df = pd.DataFrame([self.otter.sorted_values], index=[current_datetime])

                if current_datetime in self.log.index:
                    self.log.loc[current_datetime] = temp_df.loc[current_datetime]
                else:
                    self.log = pd.concat([self.log, temp_df])

                elapsed_time = time.time() - start_time

                self.counter += 1
                self.total_distance_to_target += self.distance_to_target

                if elapsed_time < self.cycletime:
                    time.sleep(self.cycletime - elapsed_time)

        except KeyboardInterrupt:
            print("Tracking disabled. Otter is now in drift mode")
            self.otter.drift()

            logs_dir = "../logs"
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)

            filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
            file_path = os.path.join(logs_dir, filename)
            self.log.to_csv(file_path, sep=";")

            time.sleep(10)

    #simulated target
    def circular_tracking(self, start_north, start_east, radius, v, use_ref_model=None):
        if use_ref_model is None:
            use_ref_model = self.target_ref

        self.otter.establish_connection(self.ip, self.port)
        self.otter.update_values()

        self.referance_point = [self.otter.sorted_values["lat"], self.otter.sorted_values["lon"], 0.0]
        self.otter.observer_coordinates = self.referance_point

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
        self.log = pd.DataFrame([self.otter.sorted_values], index=[current_datetime])

        self.otter.controller_inputs_torque(10, 0)
        time.sleep(2)
        self.otter.controller_inputs_torque(10, 0)
        time.sleep(1)

        self.function_time = time.time()

        # initialize target and reference model
        initial_north = start_north + radius
        initial_east = start_east

        self.target_ne_pos = [initial_north, initial_east]

        self.ref_dist = float(np.hypot(initial_north, initial_east))
        self.ref_dist_dot = 0.0
        self.ref_dist_ddot = 0.0

        self.update_target_reference(use_ref_model)

        print("Starting circular tracking")

        try:
            while True:
                start_time = time.time()

                # update circular target position first
                omega = v / radius
                theta = omega * (time.time() - self.function_time)

                self.target_ne_pos = [
                    start_north + radius * np.cos(theta),
                    start_east + radius * np.sin(theta)
                ]

                v_north = -radius * omega * np.sin(theta)
                v_east = radius * omega * np.cos(theta)

                self.target_v_ne = [float(v_north), float(v_east)]

                if np.hypot(v_north, v_east) > 1e-6:
                    self.path_heading = self.wrap_to_pi(math.atan2(v_east, v_north))
                else:
                    self.path_heading = self.wrap_to_pi(
                        math.atan2(self.east_error, self.north_error)
                    )

                # update errors / distance / yaw BEFORE controller
                self.update_target_reference(use_ref_model)

                # compute control using updated target values
                if self.use_nmpc:
                    tau_X, tau_N = self.calculate_forces_nmpc()
                else:
                    tau_X, tau_N = self.calculate_forces_pid()

                self.otter.controller_inputs_torque(tau_X, tau_N, self.surge_setpoint)

                # logging
                self.otter.sorted_values["north_error"] = self.north_error
                self.otter.sorted_values["east_error"] = self.east_error
                self.otter.sorted_values["distance_to_target"] = self.distance_to_target
                self.otter.sorted_values["yaw_setpoint"] = self.yaw_setpoint
                self.otter.sorted_values["current_angle"] = self.current_angle

                self.otter.sorted_values["tau_X"] = tau_X
                self.otter.sorted_values["tau_N"] = tau_N

                self.otter.sorted_values["target_north_from_observer"] = self.target_ne_pos[0]
                self.otter.sorted_values["target_east_from_observer"] = self.target_ne_pos[1]

                current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
                temp_df = pd.DataFrame([self.otter.sorted_values], index=[current_datetime])

                if current_datetime in self.log.index:
                    self.log.loc[current_datetime] = temp_df.loc[current_datetime]
                else:
                    self.log = pd.concat([self.log, temp_df])

                elapsed_time = time.time() - start_time

                self.counter += 1
                self.total_distance_to_target += self.distance_to_target

                if elapsed_time < self.cycletime:
                    time.sleep(self.cycletime - elapsed_time)

        except KeyboardInterrupt:
            print("Tracking disabled. Otter is now in drift mode")
            self.otter.drift()

            logs_dir = '../logs'
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)

            filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.csv'
            file_path = os.path.join(logs_dir, filename)
            self.log.to_csv(file_path, sep=';')

            time.sleep(10)

    #simulated target
    def square_tracking(self, start_north, start_east, side_length, target_speed):
        self.otter.establish_connection(self.ip, self.port)
        self.otter.update_values()


        self.referance_point = [self.otter.sorted_values["lat"], self.otter.sorted_values["lon"], 0.0]
        self.otter.observer_coordinates = self.referance_point
        self.target_ne_pos = [start_north, start_east]

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
        self.log = pd.DataFrame([self.otter.sorted_values], index=[current_datetime])


        self.otter.controller_inputs_torque(10, 0)
        time.sleep(2)
        self.otter.controller_inputs_torque(10, 0)
        time.sleep(1)

        curdir = "we"

        print(f"Starting tracking. North error is {start_north}m and east error is {start_east}m")
        try:
            while True:
                start_time = time.time()

                tau_X, tau_N = self.calculate_forces_pid()
                self.otter.controller_inputs_torque(tau_X, tau_N, self.surge_setpoint)

                self.otter.sorted_values["north_error"] = self.north_error
                self.otter.sorted_values["east_error"] = self.east_error
                self.otter.sorted_values["distance_to_target"] = self.distance_to_target
                self.otter.sorted_values["yaw_setpoint"] = self.yaw_setpoint
                self.otter.sorted_values["current_angle"] = self.current_angle

                self.otter.sorted_values["tau_X"] = tau_X
                self.otter.sorted_values["tau_N"] = tau_N

                if self.cycletime*target_speed*self.counter % side_length == 0:
                    if curdir == "ns":
                        curdir = "ew"
                    elif curdir == "ew":
                        curdir = "sn"
                    elif curdir == "sn":
                        curdir = "we"
                    elif curdir == "we":
                        curdir = "ns"

                if curdir == "ns":
                    self.target_ne_pos = [self.target_ne_pos[0] - (target_speed/(1/self.cycletime)), self.target_ne_pos[1]]
                elif curdir == "ew":
                    self.target_ne_pos = [self.target_ne_pos[0], self.target_ne_pos[1] - (target_speed/(1/self.cycletime))]
                elif curdir == "sn":
                    self.target_ne_pos = [self.target_ne_pos[0] + (target_speed/(1/self.cycletime)), self.target_ne_pos[1]]
                elif curdir == "we":
                    self.target_ne_pos = [self.target_ne_pos[0], self.target_ne_pos[1] + (target_speed / (1 / self.cycletime))]


                self.otter.sorted_values["target_north_from_observer"] = self.target_ne_pos[0]
                self.otter.sorted_values["target_east_from_observer"] = self.target_ne_pos[1]

                current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
                temp_df = pd.DataFrame([self.otter.sorted_values], index=[current_datetime])

                # This makes sure there is no duplicates of datetimes in the log
                if current_datetime in self.log.index:
                    self.log.loc[current_datetime] = temp_df.loc[current_datetime]
                else:
                    self.log = pd.concat([self.log, temp_df])



                elapsed_time = time.time() - start_time

                self.counter = self.counter + 1
                self.total_distance_to_target = self.total_distance_to_target + self.distance_to_target

                if elapsed_time < self.cycletime:
                    time.sleep(self.cycletime - elapsed_time)


        except KeyboardInterrupt:
            print("Tracking disabled. Otter is now in drift mode")
            self.otter.drift()

            logs_dir = '../logs'
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)

            filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.csv'
            file_path = os.path.join(logs_dir, filename)
            self.log.to_csv(file_path, sep=';')

            time.sleep(10)


    def stationary_target_tracking(self, forward_offset=5.0, starboard_offset=2.5):
        self.otter.establish_connection(self.ip, self.port)
        self.otter.update_values()

        self.referance_point = [
            self.otter.sorted_values["lat"],
            self.otter.sorted_values["lon"],
            0.0
        ]
        self.otter.observer_coordinates = self.referance_point

        # Initialize heading
        psi = self.get_initial_heading()

        # Target position relative to vessel heading
        # Convention:
        #   forward_offset   = target forward of vessel
        #   starboard_offset = target to starboard/right of vessel
        start_north = (
            forward_offset * np.cos(psi)
            - starboard_offset * np.sin(psi)
        )

        start_east = (
            forward_offset * np.sin(psi)
            + starboard_offset * np.cos(psi)
        )

        self.target_ne_pos = [start_north, start_east]
        self.target_v_ne = [0.0, 0.0]
        self.path_heading = self.wrap_to_pi(math.atan2(start_east, start_north))

        # Initialize target errors
        self.north_error = start_north
        self.east_error = start_east
        self.distance_to_target = float(np.hypot(start_north, start_east))

        self.yaw_setpoint = self.wrap_to_pi(
            math.atan2(self.east_error, self.north_error)
        )

        self.current_angle = self.wrap_to_pi(
            float(self.otter.sorted_values.get("yaw_rad", 0.0))
        )

        self.heading_error = self.wrap_to_pi(
            self.yaw_setpoint - self.current_angle
        )

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
        self.log = pd.DataFrame([self.otter.sorted_values], index=[current_datetime])

        # Small initial command
        self.otter.controller_inputs_torque(10, 0)
        time.sleep(2)
        self.otter.controller_inputs_torque(10, 0)
        time.sleep(1)

        print(
            f"Stationary target at N={start_north:.2f} m, E={start_east:.2f} m"
        )

        try:
            while True:
                start_time = time.time()

                # Update Otter values once through controller functions only.
                # Do not call current_state() here before calculate_forces_nmpc(),
                # because calculate_forces_nmpc() already calls current_state().

                n_usv = float(self.otter.sorted_values.get("north_from_observer", 0.0))
                e_usv = float(self.otter.sorted_values.get("east_from_observer", 0.0))

                n_t = float(self.target_ne_pos[0])
                e_t = float(self.target_ne_pos[1])

                self.north_error = n_t - n_usv
                self.east_error = e_t - e_usv
                self.distance_to_target = float(
                    np.hypot(self.north_error, self.east_error)
                )

                self.yaw_setpoint = self.wrap_to_pi(
                    math.atan2(self.east_error, self.north_error)
                )

                self.current_angle = self.wrap_to_pi(
                    float(self.otter.sorted_values.get("yaw_rad", 0.0))
                )

                self.heading_error = self.wrap_to_pi(
                    self.yaw_setpoint - self.current_angle
                )

                # Control
                if self.use_nmpc:
                    tau_X, tau_N = self.calculate_forces_nmpc()
                else:
                    tau_X, tau_N = self.calculate_forces_pid()

                self.otter.controller_inputs_torque(
                    tau_X,
                    tau_N,
                    self.surge_setpoint
                )

                # Logging
                self.otter.sorted_values["north_error"] = self.north_error
                self.otter.sorted_values["east_error"] = self.east_error
                self.otter.sorted_values["distance_to_target"] = self.distance_to_target
                self.otter.sorted_values["yaw_setpoint"] = self.yaw_setpoint
                self.otter.sorted_values["current_angle"] = self.current_angle
                self.otter.sorted_values["heading_error"] = self.heading_error
                self.otter.sorted_values["yaw_setpoint_deg"] = math.degrees(self.yaw_setpoint)
                self.otter.sorted_values["current_angle_deg"] = math.degrees(self.current_angle)
                self.otter.sorted_values["heading_error_deg"] = math.degrees(self.heading_error)
                self.otter.sorted_values["tau_X"] = tau_X
                self.otter.sorted_values["tau_N"] = tau_N
                self.otter.sorted_values["target_north_from_observer"] = self.target_ne_pos[0]
                self.otter.sorted_values["target_east_from_observer"] = self.target_ne_pos[1]

                current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
                temp_df = pd.DataFrame([self.otter.sorted_values], index=[current_datetime])

                if current_datetime in self.log.index:
                    self.log.loc[current_datetime] = temp_df.loc[current_datetime]
                else:
                    self.log = pd.concat([self.log, temp_df])

                elapsed_time = time.time() - start_time

                self.counter += 1
                self.total_distance_to_target += self.distance_to_target

                if elapsed_time < self.cycletime:
                    time.sleep(self.cycletime - elapsed_time)

        except KeyboardInterrupt:
            print("Tracking disabled. Otter is now in drift mode")
            self.otter.drift()

            logs_dir = "../logs"
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)

            filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
            file_path = os.path.join(logs_dir, filename)
            self.log.to_csv(file_path, sep=";")

            time.sleep(10)


    def calculate_forces_pid(self, ugps=False, ugps_lat=None, ugps_lon=None, ugps_h=None, stationary_tracking=True):
        self.otter.update_values()
        n_usv = float(self.otter.sorted_values["north_from_observer"])
        e_usv = float(self.otter.sorted_values["east_from_observer"])

        # Update NED target position from GPS if needed
        if ugps:
            if ugps_lat is None or ugps_lon is None:
                raise ValueError("ugps=True requires ugps_lat and ugps_lon")

            obs_lat = float(self.otter.sorted_values["observer_lat"])
            obs_lon = float(self.otter.sorted_values["observer_lon"])
            obs_h   = float(self.otter.sorted_values["observer_height"])

            if ugps_h is None:
                ugps_h = obs_h

            n_t, e_t, d_t = pm.geodetic2ned(
                float(ugps_lat), float(ugps_lon), float(ugps_h),
                obs_lat, obs_lon, obs_h
            )
            self.target_ne_pos = [float(n_t), float(e_t)]

        n_t = float(self.target_ne_pos[0])
        e_t = float(self.target_ne_pos[1])

        # Current heading first
        self.current_angle = self.wrap_to_pi(
            float(self.otter.sorted_values.get("yaw_rad", 0.0))
        )

        # Target relative to USV
        self.north_error = n_t - n_usv
        self.east_error  = e_t - e_usv
        self.distance_to_target = math.hypot(self.north_error, self.east_error)

        arrival_radius = 0.2

        if self.distance_to_target < arrival_radius:
            self.north_error = 0.0
            self.east_error = 0.0
            self.distance_to_target = 0.0
            
            self.yaw_setpoint = self.wrap_to_pi(
                math.atan2(self.east_error, self.north_error)
            )

        # remove if stationary
        elif self.distance_to_target < 3.0 and stationary_tracking:
            pass 
        else:
            self.yaw_setpoint = self.wrap_to_pi(
                math.atan2(self.east_error, self.north_error)
            )

        # Compute wrapped heading error
        heading_error = self.wrap_to_pi(self.yaw_setpoint - self.current_angle)

        tau_X = self.surge_PID.calculate_surge(
            self.surge_setpoint,
            self.distance_to_target,
            self.yaw_setpoint,
            self.current_angle
        )

        tau_N = self.yaw_PID.calculate_yaw(
            self.yaw_setpoint,
            self.current_angle,
            self.surge_setpoint,
            self.distance_to_target
        )

        # swap back to 
        # Prevent forward thrust if heading error is too large
        if stationary_tracking == True:
            target_is_behind = abs(heading_error) > (math.pi / 2)
            if target_is_behind and self.distance_to_target < self.surge_PID.pid.dp_reverse_radius:
                tau_X = tau_X          # reverse zone 
            elif abs(heading_error) > math.radians(35):
                tau_X = 0.0
            elif abs(heading_error) > math.radians(15):
                tau_X = tau_X * 0.3
        else:
            if abs(heading_error) > math.radians(20):
                tau_X = 0.0 

        # Saturation
        #tau_N = max(min(tau_N, self.max_force), -self.max_force)
        #remaining_force = self.max_force - abs(tau_N)
        #tau_X = max(min(tau_X, remaining_force), -remaining_force)

        tau_X = np.clip(
        tau_X,
        -self.otter.otter_control.max_surge_N,
        self.otter.otter_control.max_surge_N
        )

        tau_N = np.clip(
            tau_N,
            -self.otter.otter_control.max_yaw_N,
            self.otter.otter_control.max_yaw_N
        )

        return tau_X, tau_N

    def calculate_forces_nmpc(self):
        """
        Solves NMPC control.

        init_state:
            [x, y, psi, u, v, r]

        target_reference:
            [x_ref, y_ref, path_heading, target_v_north, target_v_east]
        """

        # Initialize observer reference once before using NMPC state feedback
        if not getattr(self, "nmpc_state_initialized", False):
            initialized = self.initialize_nmpc_state_reference()

            if not initialized:
                print("NMPC: waiting for observer reference initialization")
                return 0.0, 0.0

            self.nmpc_state_initialized = True

        init_state = self.current_state()

        if init_state is None:
            print("NMPC: waiting for valid Otter state")
            return 0.0, 0.0

        init_state = np.asarray(init_state, dtype=float)

        if not np.all(np.isfinite(init_state)):
            print("NMPC: non-finite state, skipping control step:", init_state)
            return 0.0, 0.0

        target_v_ne = getattr(self, "target_v_ne", [0.0, 0.0])

        target_reference = np.array([
            self.target_ne_pos[0],
            self.target_ne_pos[1],
            getattr(self, "path_heading", 0.0),
            target_v_ne[0],
            target_v_ne[1],
        ], dtype=float)

        if not np.all(np.isfinite(target_reference)):
            print("NMPC: invalid target reference:", target_reference)
            return 0.0, 0.0

        tau = self.nmpc.solve_control(init_state, target_reference)

        tau_X = float(tau[0])
        tau_N = float(tau[2])

        # Debug geometry
        psi = float(init_state[2])

        target_angle = math.atan2(
            target_reference[1] - init_state[1],
            target_reference[0] - init_state[0]
        )

        heading_error = self.wrap_to_pi(target_angle - psi)

        print(
            "NMPC debug | "
            f"x={init_state[0]:.2f}, y={init_state[1]:.2f}, "
            f"psi={math.degrees(psi):.1f} deg, "
            f"target_angle={math.degrees(target_angle):.1f} deg, "
            f"path_heading={math.degrees(target_reference[2]):.1f} deg, "
            f"heading_error={math.degrees(heading_error):.1f} deg, "
            f"u={init_state[3]:.2f}, v={init_state[4]:.2f}, "
            f"r={math.degrees(init_state[5]):.1f} deg/s, "
            f"target=({target_reference[0]:.2f}, {target_reference[1]:.2f}), "
            f"target_v=({target_reference[3]:.2f}, {target_reference[4]:.2f}), "
            f"tau_X={tau_X:.2f}, tau_N={tau_N:.2f}"
        )

        return tau_X, tau_N

    # get target ned position in live tracking 
    def ugps_geo_to_ned(self, ugps_lat, ugps_lon, ugps_h=0.0):
        obs_lat = self.otter.sorted_values["observer_lat"]
        obs_lon = self.otter.sorted_values["observer_lon"]
        obs_h   = self.otter.sorted_values["observer_height"]

        n, e, d = pm.geodetic2ned(ugps_lat, ugps_lon, ugps_h, obs_lat, obs_lon, obs_h)
        return float(n), float(e), float(d)
    
    def save_log(self):
        print("Tracking disabled. Otter is now in drift mode")
        self.otter.drift()
        logs_dir = './logs'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.csv'
        file_path = os.path.join(logs_dir, filename)
        try:
            self.log.to_csv(file_path, sep=';')
        except Exception as e:
            print(f"Error when trying to save the log: {e}")
    
        # helper function to update target reference through third order reference
    
    def update_target_reference(self, use_ref_model):
        """
        Updates target error, distance, desired heading, and actual heading

        target_ne_pos is assumed to be absolute N/E position relative to observer
        Otter position is read from sorted_values
        """

        self.otter.update_values()

        n_usv = float(self.otter.sorted_values.get("north_from_observer", 0.0))
        e_usv = float(self.otter.sorted_values.get("east_from_observer", 0.0))

        n_t = float(self.target_ne_pos[0])
        e_t = float(self.target_ne_pos[1])

        # Error from vessel to target
        self.north_error = n_t - n_usv
        self.east_error = e_t - e_usv

        raw_dist = float(np.hypot(self.north_error, self.east_error))

        if use_ref_model:
            self.ref_dist, self.ref_dist_dot, self.ref_dist_ddot = self.third_order_reference(
                self.ref_dist,
                self.ref_dist_dot,
                self.ref_dist_ddot,
                raw_dist,
                self.zeta_ref,
                self.omega_n_ref,
                self.cycletime
            )
            self.distance_to_target = self.ref_dist
        else:
            self.distance_to_target = raw_dist

        # Desired heading toward target
        self.yaw_setpoint = self.wrap_to_pi(
            math.atan2(self.east_error, self.north_error)
        )

        # Actual measured heading from Otter
        self.current_angle = self.wrap_to_pi(
            float(self.otter.sorted_values.get("yaw_rad", 0.0))
        )

        # Useful for logging/debugging
        self.heading_error = self.wrap_to_pi(self.yaw_setpoint - self.current_angle)

    #target reference model @ Fossen
    @staticmethod
    def third_order_reference(x_d, x_d_dot, x_d_ddot, x_ref, zeta, omega_n, dt):
        x = np.array([x_d, x_d_dot, x_d_ddot], dtype=float)

        Ad = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-omega_n**3,
            -(2.0*zeta + 1.0)*omega_n**2,
            -(2.0*zeta + 1.0)*omega_n]
        ], dtype=float)

        Bd = np.array([0.0, 0.0, omega_n**3], dtype=float)

        x_dot = Ad @ x + Bd * float(x_ref)
        x_next = x + dt * x_dot

        return float(x_next[0]), float(x_next[1]), float(x_next[2])
    
    @staticmethod
    def wrap_to_pi(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def initialize_nmpc_state_reference(self):
        self.otter.establish_connection(self.ip, self.port)
        self.otter.update_values()

        lat = self.otter.sorted_values.get("lat")
        lon = self.otter.sorted_values.get("lon")

        if lat is None or lon is None:
            print("NMPC: waiting for GPS reference initialization")
            return False

        self.referance_point = [lat, lon, 0.0]
        self.otter.observer_coordinates = self.referance_point

        # Give observer one update cycle to compute local N/E values
        self.otter.update_values()

        print("NMPC observer reference initialized:", self.referance_point)
        return True

    def get_initial_heading(self, max_wait_s=5.0):
        """
        Reads initial heading before creating a stationary target offset.
        """

        t0 = time.time()

        while time.time() - t0 < max_wait_s:
            self.otter.update_values()

            psi = self._get_heading_signal()

            if psi is not None:
                return self.wrap_to_pi(float(psi))

            time.sleep(0.1)

        print("No valid heading available. Using psi = 0.0")
        return 0.0

    def _get_heading_signal(self):
        """
        Returns heading angle psi in radians.

        Preferred:
            yaw_rad from Otter_api.py

        Fallback:
            raw current_orientation_3 in degrees

        Last fallback:
            course over ground in degrees
        """

        psi = self._confirm_signal("yaw_rad", None)

        if psi is not None:
            return self.wrap_to_pi(float(psi))

        psi_deg = self._confirm_signal("current_orientation_3", None)

        if psi_deg is not None:
            return self.wrap_to_pi(math.radians(float(psi_deg)))

        cog = self._confirm_signal("current_course_over_ground", None)

        if cog is not None:
            return self.wrap_to_pi(math.radians(float(cog)))

        if self.last_valid_state is not None:
            return float(self.last_valid_state[2])

        return None
        