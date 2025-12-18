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

    def __init__(self, ip, port, surge_PID, yaw_PID, surge_setpoint, otter, nmpc=None, use_nmpc=False, control_dt=0.1):

        self.ip = ip
        self.port = port

        self.surge_PID = surge_PID
        self.yaw_PID = yaw_PID

        self.otter = otter

        self.surge_setpoint = surge_setpoint

        self.nmpc = nmpc
        self.use_nmpc = use_nmpc
        self.control_dt = control_dt 

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
        self.last_valid_state = np.zeros(6)

        self._ugps_lock = threading.Lock()
        self._ugps_latest = None  # lat, lon, depth, x, y, z, t

        self.ref_dist = 0.0
        self.ref_dist_dot = 0.0
        self.ref_dist_ddot = 0.0

        # test zeta/omega for surge reference
        self.zeta_ref = 0.9
        self.omega_n_ref = 0.6
            
    # filter signals by signals are floats and a finite values
    def _confirm_signal(self, key, default=0.0):
        values = self.otter.sorted_values.get(key, default)
        try:
            values = float(values)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(values):
            return values

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
    
    # Update USV states 
    def current_state(self):
        self.otter.update_values()
        
        # raw values if not inf 
        x_raw   = self._confirm_signal("north_from_observer", 0.0)
        y_raw   = self._confirm_signal("east_from_observer", 0.0)
        psi_raw = self._confirm_signal("current_angle", 0.0)   

        u_raw   = self._confirm_signal("speed_n", 0.0)
        v_raw   = self._confirm_signal("speed_e", 0.0)
        r_raw   = self._confirm_signal("current_yaw_rate", 0.0)
      
        state = np.array([x_raw, y_raw, psi_raw, u_raw, v_raw, r_raw])
        self.last_valid_state = state                
        
        # Initial state
        if self.last_valid_state is None:
            state = np.array([x_raw, y_raw, psi_raw, u_raw, v_raw, r_raw])
            self.last_valid_state = state
            return state          

        x_prev, y_prev, psi_prev, u_prev, v_prev, r_prev = self.last_valid_state
        # Controller limits

        dt = self.cycletime            # sampling time 
        u_max = 2.0                    # max surge 
        r_max = 20.0 * math.pi/180.0   # max yaw 

        max_pos_change = u_max * dt    # Check changes in surge/heading are valid for timestep
        max_psi_change = r_max * dt   

        x = self._filter_signal(x_raw, x_prev, difference_limit=max_pos_change)
        y = self._filter_signal(y_raw, y_prev, difference_limit=max_pos_change)

        if psi_raw != psi_prev:
            psi_difference = (psi_raw - psi_prev + math.pi) % (2*math.pi) - math.pi # [-pi,pi] 
            psi_new = psi_prev + psi_difference # add difference to new
            psi = self._filter_signal(psi_new, psi_prev, difference_limit=max_psi_change)
        else:
            psi = psi_raw
        

        # max 50% velocity change per reading
        u = self._filter_signal(u_raw, u_prev, relative_limit=0.5) 
        v = self._filter_signal(v_raw, v_prev, relative_limit=0.5)
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
                # error setpoint from reference model
                raw_dist = float(np.hypot(self.north_error, self.east_error))

                self.ref_dist, self.ref_dist_dot, self.ref_dist_ddot = \
                    self.third_order_reference(
                        self.ref_dist,
                        self.ref_dist_dot,
                        self.ref_dist_ddot,
                        raw_dist,
                        self.zeta_ref,
                        self.omega_n_ref
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
    def target_tracking(self, start_north, start_east, v_north, v_east):
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


        if self.use_nmpc:
            print(f"Starting NMPC tracking. North error is {start_north}m and east error is {start_east}m")
        else:
            print(f"Starting PID tracking. North error is {start_north}m and east error is {start_east}m")

        try:
            while True:
                start_time = time.time()

                
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

                # update moving target position
                self.target_ne_pos = [
                    self.target_ne_pos[0] + (v_north / (1 / self.cycletime)),
                    self.target_ne_pos[1] + (v_east  / (1 / self.cycletime)),
                ]

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
    def circular_tracking(self, start_north, start_east, radius, v):
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

        print(f"Starting circular tracking")
        try:
            while True:
                start_time = time.time()

                omega = v / radius
                theta = omega * (time.time() - self.function_time)
                
                self.target_ne_pos = [start_north + radius * np.cos(theta), start_east + radius * np.sin(theta)]
                
                tau_X, tau_N = self.calculate_forces_pid()
                self.otter.controller_inputs_torque(tau_X, tau_N, self.surge_setpoint)

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

    def calculate_forces_pid(self, ugps=False, ugps_lat=None, ugps_lon=None, ugps_h=None):
        self.otter.update_values()
        n_usv = float(self.otter.sorted_values["north_from_observer"])
        e_usv = float(self.otter.sorted_values["east_from_observer"])

        # Update ned 
        if ugps:
            if ugps_lat is None or ugps_lon is None:
                raise ValueError("ugps=True requires ugps_lat and ugps_lon")

            obs_lat = float(self.otter.sorted_values["observer_lat"])
            obs_lon = float(self.otter.sorted_values["observer_lon"])
            obs_h   = float(self.otter.sorted_values["observer_height"])


            if ugps_h is None:
                ugps_h = obs_h

            n_t, e_t, d_t = pm.geodetic2ned(float(ugps_lat), float(ugps_lon), float(ugps_h),
                                        obs_lat, obs_lon, obs_h)
            self.target_ne_pos = [float(n_t), float(e_t)]

        n_t = float(self.target_ne_pos[0])
        e_t = float(self.target_ne_pos[1])

        # target relative to usv
        self.north_error = n_t - n_usv
        self.east_error  = e_t - e_usv
        self.distance_to_target = math.hypot(self.north_error, self.east_error)

        if ugps:
            arrival_radius = 0.5 # meters
            if self.distance_to_target < self.surge_setpoint:   # or specific with arrival_rad
                self.north_error = 0.0
                self.east_error = 0.0
                self.distance_to_target = 0.0

        # heading
        self.yaw_setpoint = math.atan2(self.east_error, self.north_error)
        self.current_angle = float(self.otter.sorted_values["current_orientation_3"]) * (math.pi / 180.0)

        tau_X = self.surge_PID.calculate_surge(self.surge_setpoint, self.distance_to_target,
                                            self.yaw_setpoint, self.current_angle)
        tau_N = self.yaw_PID.calculate_yaw(self.yaw_setpoint, self.current_angle,
                                        self.surge_setpoint, self.distance_to_target)

        # saturation
        tau_N = max(min(tau_N, self.max_force), -self.max_force)
        remaining_force = self.max_force - abs(tau_N)
        tau_X = max(min(tau_X, remaining_force), -remaining_force)

        return tau_X, tau_N

    def calculate_forces_nmpc(self):

        init_state = np.asarray(self.current_state(), dtype=float)
        
        if not np.all(np.isfinite(init_state)):
            print("NMPC: non-finite state, skipping control step:", init_state)
            return 0.0, 0.0
        
        target_reference = np.array([
            self.target_ne_pos[0],    # x_ref (north)
            self.target_ne_pos[1],    # y_ref (east)
        ], dtype=float)

        tau = self.nmpc.solve_control(init_state, target_reference)
        tau_X, tau_N = float(tau[0]), float(tau[1])
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

    #target reference model @Alexander Rambech 
    @staticmethod
    def third_order_reference(x_d, x_d_dot, x_d_ddot, x_ref, zeta, omega_n):
        x_desired = np.array([x_d, x_d_dot, x_d_ddot])
        x_reference = np.array([0, 0, x_ref])
    
        Ad = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [-omega_n**3, -(2*zeta+1)*omega_n**2, -(2*zeta+1)*omega_n]
        ])
        Bd = np.array([0, 0, omega_n**3])
    
        x_d, x_d_dot, x_d_ddot = Ad.dot(x_desired) + Bd.dot(x_reference)
    
        return x_d, x_d_dot, x_d_ddot