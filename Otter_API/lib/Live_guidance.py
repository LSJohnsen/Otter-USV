import numpy as np
import time
import math
import datetime
import pandas as pd
import os


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

    def calculate_forces_pid(self):

        self.otter.update_values()
        self.otter_ne_pos = [self.otter.sorted_values["north_from_observer"], self.otter.sorted_values["east_from_observer"]]

        self.north_error = self.target_ne_pos[0] - self.otter_ne_pos[0]
        self.east_error = self.target_ne_pos[1] - self.otter_ne_pos[1]
        self.distance_to_target = math.sqrt(self.north_error**2 + self.east_error**2)

        if self.distance_to_target < self.surge_setpoint:
            self.north_error = 0
            self.east_error = 0
            self.distance_to_target = 0

        self.yaw_setpoint = math.atan2(self.east_error, self.north_error)
        self.current_angle = (self.otter.sorted_values["current_orientation_3"]) * (math.pi / 180)


        tau_X = self.surge_PID.calculate_surge(self.surge_setpoint, self.distance_to_target, self.yaw_setpoint, self.current_angle)
        tau_N = self.yaw_PID.calculate_yaw(self.yaw_setpoint, self.current_angle, self.surge_setpoint, self.distance_to_target)

        tau_N = max(min(tau_N, self.max_force), -(self.max_force))

        remaining_force = self.max_force - abs(tau_N)

        tau_X = max(min(tau_X, remaining_force), -(remaining_force))

        return tau_X, tau_N

    def calculate_forces_nmpc(self):

        init_state = np.asarray(self.current_state(), dtype=float)
        
        if not np.all(np.isfinite(init_state)):
            print("NMPC: non-finite init_state, skipping control step:", init_state)
            return 0.0, 0.0
        
        target_reference = np.array([
            self.target_ne_pos[0],    # x_ref (north)
            self.target_ne_pos[1],    # y_ref (east)
        ], dtype=float)

        tau = self.nmpc.solve_control(init_state, target_reference)
        tau_X, tau_N = float(tau[0]), float(tau[1])
        return tau_X, tau_N


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
