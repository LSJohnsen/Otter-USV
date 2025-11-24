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

        self.last_valid_state = np.zeros(6)

    # validate values for nmpc 
    def _valid_value(self, key, default=0.0):
    
        val = self.otter.sorted_values.get(key, default)
        try:
            val = float(val)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(val):
            return default
        return val
    
    def current_state(self):
        self.otter.update_values()

        x   = self.otter.sorted_values["north_from_observer"]
        y   = self.otter.sorted_values["east_from_observer"]
        psi = self.otter.sorted_values["current_angle"]          # heading [rad]

        u   = self.otter.sorted_values["speed_n"]                # surge velocity
        v   = self.otter.sorted_values["speed_e"]                # sway velocity
        r   = self.otter.sorted_values["current_yaw_rate"]       # yaw rate

        state = np.array([x, y, psi, u, v, r])
        self.last_valid_state = state                           # update valid values
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

                # --- choose controller ---
                if self.use_nmpc:
                    tau_X, tau_N = self.calculate_forces_nmpc()
                else:
                    tau_X, tau_N = self.calculate_forces_pid()

                # send torques to Otter
                self.otter.controller_inputs_torque(tau_X, tau_N, self.surge_setpoint)

                # logging (same for both controllers)
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

        self.otter.update_values()
        init_state = self.current_state()

        
        target_reference = np.array([
            self.target_ne_pos[0],    # x_ref (north)
            self.target_ne_pos[1],    # y_ref (east)
        ])

        tau = self.nmpc.solve_control(init_state, target_reference)
        tau_X, tau_N = tau[0], tau[1]
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
