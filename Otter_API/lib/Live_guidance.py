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
        self.omega_n_ref = 0.6

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
        self.otter.update_values()


        # Raw measurements
        x_raw   = self._confirm_signal("north_from_observer")
        y_raw   = self._confirm_signal("east_from_observer")
        psi_raw = self._get_heading_signal()

        v_n_raw = self._confirm_signal("speed_n")
        v_e_raw = self._confirm_signal("speed_e")
        r_raw = self._confirm_signal("current_rotational_velocities_3")

        if r_raw is None:
            if self.last_valid_state is not None:
                psi_prev = self.last_valid_state[2]
                dpsi = (psi_raw - psi_prev + math.pi) % (2 * math.pi) - math.pi
                r_raw = dpsi / self.cycletime
            else:
                r_raw = 0.0
        else:
            r_raw = float(r_raw)


        debug_values = {
            "north_from_observer": x_raw,
            "east_from_observer": y_raw,
            "heading": psi_raw,
            "speed_n": v_n_raw,
            "speed_e": v_e_raw,
            "yaw_rate": r_raw,
        }

        missing = [name for name, value in debug_values.items() if value is None]
    
        if missing:
            print("missing state values:", missing)
            print("Raw state debug:", debug_values)

        required = [x_raw, y_raw, psi_raw, v_n_raw, v_e_raw]


        # If init data missing, return None so NMPC to skip without crash
        if any(value is None for value in required):
            if self.last_valid_state is not None:
                return self.last_valid_state
            return None

        x_raw   = float(x_raw)
        y_raw   = float(y_raw)
        psi_raw = float(psi_raw)
        v_n_raw = float(v_n_raw)
        v_e_raw = float(v_e_raw)
        r_raw   = float(r_raw)

        # Convert inertial to body-frame
        u_body_raw = math.cos(psi_raw) * v_n_raw + math.sin(psi_raw) * v_e_raw
        v_body_raw = -math.sin(psi_raw) * v_n_raw + math.cos(psi_raw) * v_e_raw

        # Initial valid state
        if self.last_valid_state is None:
            state = np.array(
                [x_raw, y_raw, psi_raw, u_body_raw, v_body_raw, r_raw],
                dtype=float
            )
            self.last_valid_state = state
            return state

        x_prev, y_prev, psi_prev, u_prev, v_prev, r_prev = self.last_valid_state

        dt = self.cycletime
        u_max = 2.0
        r_max = 20.0 * math.pi / 180.0

        max_pos_change = u_max * dt
        max_psi_change = r_max * dt

        x = self._filter_signal(x_raw, x_prev, difference_limit=max_pos_change)
        y = self._filter_signal(y_raw, y_prev, difference_limit=max_pos_change)

        # Wrap heading to [-pi, pi]
        psi_difference = (psi_raw - psi_prev + math.pi) % (2 * math.pi) - math.pi
        psi_candidate = psi_prev + psi_difference
        psi = self._filter_signal(
            psi_candidate,
            psi_prev,
            difference_limit=max_psi_change
        )

        # Recompute body velocities using filtered heading
        u_body_raw = math.cos(psi) * v_n_raw + math.sin(psi) * v_e_raw
        v_body_raw = -math.sin(psi) * v_n_raw + math.cos(psi) * v_e_raw

        # Limit velocity/rate jumps
        u = self._filter_signal(u_body_raw, u_prev, relative_limit=0.5)
        v = self._filter_signal(v_body_raw, v_prev, relative_limit=0.5)
        r = self._filter_signal(r_raw, r_prev, relative_limit=0.5)

        state = np.array([x, y, psi, u, v, r], dtype=float)
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
        self.referance_point = [self.otter.sorted_values["lat"], self.otter.sorted_values["lon"], 0.0]
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

                # latest UGPS (from separate thread reader)
                with self._ugps_lock:
                    ugps = None if self._ugps_latest is None else dict(self._ugps_latest)

                #drift if no singal
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
                target_north, target_east, target_down = self.ugps_geo_to_ned(ugps["lat"], ugps["lon"])


                self.otter.sorted_values["target_north_from_observer"] = target_north
                self.otter.sorted_values["target_east_from_observer"]  = target_east

                # error
                self.north_error = target_north
                self.east_error  = target_east

                ''' if ref model doesn't work:
                self.distance_to_target = float(np.hypot(self.north_error, self.east_error))
      
                self.current_angle = float(np.arctan2(self.east_error, self.north_error))
                self.yaw_setpoint = self.current_angle
                '''
                # error setpoint from reference model position/velocity/acceleration
                dt = self.cycletime

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

                self.current_angle = float(np.arctan2(self.east_error, self.north_error))
                self.yaw_setpoint = self.current_angle

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
                self.otter.sorted_values["tau_X"] = tau_X
                self.otter.sorted_values["tau_N"] = tau_N

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
                self.log.to_csv(file_path, sep=';')

    #straight simulated target
    def target_tracking(self, start_north, start_east, v_north, v_east, use_ref_model=None):
        if use_ref_model is None:
            use_ref_model = self.target_ref
        self.otter.establish_connection(self.ip, self.port)
        self.otter.update_values()
        

        self.referance_point = [self.otter.sorted_values["lat"], self.otter.sorted_values["lon"], 0.0]
        self.otter.observer_coordinates = self.referance_point
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
            print(f"Starting NMPC tracking. North error is {start_north}m and east error is {start_east}m")
        else:
            print(f"Starting PID tracking. North error is {start_north}m and east error is {start_east}m")

        try:
            while True:
                start_time = time.time()

                # update moving target position first
                self.target_ne_pos = [
                    self.target_ne_pos[0] + v_north * self.cycletime,
                    self.target_ne_pos[1] + v_east  * self.cycletime,
                ]

                # update errors / distance / yaw BEFORE controller
                self.update_target_reference(use_ref_model)

                # compute control using updated target values
                if self.use_nmpc:
                    tau_X, tau_N = self.calculate_forces_nmpc()
                else:
                    tau_X, tau_N = self.calculate_forces_pid()

                # send torques to Otter
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


    def stationary_target_tracking(self, forward_offset=10.0, starboard_offset=5.0):
        self.otter.establish_connection(self.ip, self.port)
        self.otter.update_values()

        self.referance_point = [
            self.otter.sorted_values["lat"],
            self.otter.sorted_values["lon"],
            0.0
        ]
        self.otter.observer_coordinates = self.referance_point

        psi = self.get_initial_heading()

        # target position relative to vessel heading
        start_north = (
            forward_offset * np.cos(psi)
            - starboard_offset * np.sin(psi)
        )

        start_east = (
            forward_offset * np.sin(psi)
            + starboard_offset * np.cos(psi)
        )

        self.target_ne_pos = [start_north, start_east]

        # initialize errors once
        self.north_error = start_north
        self.east_error = start_east
        self.distance_to_target = float(np.hypot(start_north, start_east))
        self.current_angle = float(np.arctan2(self.east_error, self.north_error))
        self.yaw_setpoint = self.current_angle

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
        self.log = pd.DataFrame([self.otter.sorted_values], index=[current_datetime])

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

                #  target stays fixed
                self.north_error = self.target_ne_pos[0]
                self.east_error = self.target_ne_pos[1]

                self.distance_to_target = float(np.hypot(self.north_error, self.east_error))
                self.current_angle = float(np.arctan2(self.east_error, self.north_error))
                self.yaw_setpoint = self.current_angle

                # control
                if self.use_nmpc:
                    state = self.current_state()

                    if state is None:
                        print("No valid NMPC state -> drift/hold")
                        self.otter.drift()
                        time.sleep(0.2)
                        continue

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


    def calculate_forces_pid(self, ugps=False, ugps_lat=None, ugps_lon=None, ugps_h=None):
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
        self.current_angle = float(self.otter.sorted_values["current_orientation_3"]) * (math.pi / 180.0)

        # Target relative to USV
        self.north_error = n_t - n_usv
        self.east_error  = e_t - e_usv
        self.distance_to_target = math.hypot(self.north_error, self.east_error)

        arrival_radius = 0.2

        if self.distance_to_target < arrival_radius:
            self.north_error = 0.0
            self.east_error = 0.0
            self.distance_to_target = 0.0
            self.yaw_setpoint = self.current_angle
        else:
            self.yaw_setpoint = math.atan2(self.east_error, self.north_error)

        # Compute wrapped heading error
        heading_error = (self.yaw_setpoint - self.current_angle + math.pi) % (2 * math.pi) - math.pi

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

        # Prevent forward thrust if heading error is too large
        if abs(heading_error) > math.radians(20):
            tau_X = 0.0

        # Saturation
        tau_N = max(min(tau_N, self.max_force), -self.max_force)
        remaining_force = self.max_force - abs(tau_N)
        tau_X = max(min(tau_X, remaining_force), -remaining_force)

        return tau_X, tau_N

    def calculate_forces_nmpc(self):
        """
        solve_control(init_state, target_reference)

        init_state: np.array shape (6,)  -> [x, y, psi, u, v, r]
        target_reference: np.array shape (2,) -> [x_ref, y_ref]
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

        target_reference = np.array([
            self.target_ne_pos[0],
            self.target_ne_pos[1],
        ], dtype=float)

        if not np.all(np.isfinite(target_reference)):
            print("NMPC: invalid target reference:", target_reference)
            return 0.0, 0.0

        tau = self.nmpc.solve_control(init_state, target_reference)
        print("NMPC state:", init_state)
        print("NMPC target:", target_reference)
        print("NMPC tau:", tau)

        tau_X = float(tau[0])
        tau_N = float(tau[2])

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
        self.north_error = self.target_ne_pos[0]
        self.east_error = self.target_ne_pos[1]

        raw_dist = float(np.hypot(self.north_error, self.east_error))

        if use_ref_model:
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
        else:
            self.distance_to_target = raw_dist

        self.current_angle = float(np.arctan2(self.east_error, self.north_error))
        self.yaw_setpoint = self.current_angle

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
        t0 = time.time()

        while time.time() - t0 < max_wait_s:
            self.otter.update_values()

            psi = self._confirm_signal("current_angle", None)

            if psi is not None:
                return float(psi)

            time.sleep(0.1)

        print("No valid current_angle available. Using psi = 0.0")
        return 0.0
    
    def _get_heading_signal(self):
        # prefer IMU/yaw
        psi = self._confirm_signal("current_orientation_3")
        if psi is not None:
            return psi

        # fallback to course over ground but only works when moving
        cog = self._confirm_signal("current_course_over_ground")
        if cog is not None:
            return math.radians(cog)

        # fallback to previous heading
        if self.last_valid_state is not None:
            return self.last_valid_state[2]

        return None