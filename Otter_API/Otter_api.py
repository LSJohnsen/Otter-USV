import lib.Control_no_mapping as Control
import lib.Connector as Connector
import time
import pymap3d as pm
import math
import numpy as np

#
#  This is the complete API for the Otter. This can be imported in custom programs and then the functions can be called to communicate with the otter.
#

class otter():

    def __init__(self):

        self.verbose = True

        # Creates instances of the connector and control classes
        self.otter_connector = Connector.otter_connector()
        self.otter_control = Control.otter_control()


        # The observer coordinates for the geodetic to ned conversion. This can be changed manually
        self.observer_coordinates = [59.908642666666665, 10.71945885, 0.0]


        # Creates an empty dictionary for the values'
        self.values = {}
        self.sorted_values = {}


        # Variables and lists
        self.geo2ned_from_observer = [0.0, 0.0, 0.0]

        self.tau_N_neg = False

        self.sorted_values["current_time"] = time.time()

        self.sorted_values["tau_X"] = 0.0
        self.sorted_values["tau_N"] = 0.0

        self.sorted_values["controller_X_cmd"] = 0.0
        self.sorted_values["controller_N_cmd"] = 0.0

        self.sorted_values["PMARMAN_X"] = 0.0
        self.sorted_values["PMARMAN_N"] = 0.0

        self.sorted_values["distance_to_target"] = 0.0
        self.sorted_values["yaw_setpoint"] = 0.0
        self.sorted_values["target_north_from_observer"] = 0.0
        self.sorted_values["target_east_from_observer"] = 0.0

        

        self.prev_ned_for_velocity = None
        self.prev_time_for_velocity = None
        self.filtered_speed_n = 0.0
        self.filtered_speed_e = 0.0
        self.velocity_alpha = 0.2

    # Tries to establish connection to the otter. Default values are in place for testing on a local machine with a test server. Returns boolean
    def establish_connection(self, ip, port):
        return self.otter_connector.establish_connection(ip, port)

    # Tries to close the socket connection to the Otter. Returns boolean
    def close_connection(self):
        return self.otter_connector.close_connection()

    # Checks the current connection status to the Otter. Returns boolean
    def check_connection(self):
        return self.otter_connector.check_connection()

    # Reads a message from the Otter and returns it and stores it in "last_message_recieved". Returns the last message recieved.
    def read_message(self):
        return self.otter_connector.read_message()

    # Sends a message to the Otter, with the option of adding checksum (defaults to False). Returns boolean
    def send_message(self, message, checksum_needed = False):
        return self.otter_connector.send_message(message, checksum_needed)

    # Tries to update all the values in the dictionary "values" with the current values from the Otter. Requires connection established. Returns updated dictionary "values"
    def update_values(self):
        self.otter_connector.update_values(timeout=0.3)

        self.values["current_position"] = self.otter_connector.current_position
        self.values["previous_position"] = self.otter_connector.previous_position
        self.values["last_speed_update"] = self.otter_connector.last_speed_update
        self.values["current_course_over_ground"] = self.otter_connector.current_course_over_ground
        self.values["current_speed"] = self.otter_connector.current_speed
        self.values["current_fuel_capacity"] = self.otter_connector.current_fuel_capacity
        self.values["current_orientation"] = self.otter_connector.current_orientation
        self.values["current_rotational_velocities"] = self.otter_connector.current_rotational_velocities
        self.values["observer_coordinates"] = self.observer_coordinates
        self.values["geo2ned_from_observer"] = self.geo2ned_from_observer

        # Basic measurements
        self.sorted_values["lat"] = self.values["current_position"][0]
        self.sorted_values["lon"] = self.values["current_position"][1]
        self.sorted_values["height"] = self.values["current_position"][2]

        self.sorted_values["previous_lat"] = self.values["previous_position"][0]
        self.sorted_values["previous_lon"] = self.values["previous_position"][1]
        self.sorted_values["previous_height"] = self.values["previous_position"][2]

        self.sorted_values["last_speed_update"] = self.values["last_speed_update"]
        self.sorted_values["current_course_over_ground"] = self.values["current_course_over_ground"]
        self.sorted_values["current_speed"] = self.values["current_speed"]
        self.sorted_values["current_fuel_capacity"] = self.values["current_fuel_capacity"]

        self.sorted_values["current_orientation_1"] = self.values["current_orientation"][0]
        self.sorted_values["current_orientation_2"] = self.values["current_orientation"][1]
        self.sorted_values["current_orientation_3"] = self.values["current_orientation"][2]

        self.sorted_values["current_rotational_velocities_1"] = self.values["current_rotational_velocities"][0]
        self.sorted_values["current_rotational_velocities_2"] = self.values["current_rotational_velocities"][1]
        self.sorted_values["current_rotational_velocities_3"] = self.values["current_rotational_velocities"][2]

        self.sorted_values["observer_lat"] = self.values["observer_coordinates"][0]
        self.sorted_values["observer_lon"] = self.values["observer_coordinates"][1]
        self.sorted_values["observer_height"] = self.values["observer_coordinates"][2]

        # NED position
        self.geo2ned_position()

        self.sorted_values["north_from_observer"] = self.geo2ned_from_observer[0]
        self.sorted_values["east_from_observer"] = self.geo2ned_from_observer[1]
        self.sorted_values["down_from_observer"] = self.geo2ned_from_observer[2]

        # Time update
        self.sorted_values["previous_time"] = self.sorted_values["current_time"]
        self.sorted_values["current_time"] = time.time()

        self.sorted_values["cycle_time"] = (
            self.sorted_values["current_time"]
            - self.sorted_values["previous_time"]
        )

        # Velocity estimation from N/E position
        cur_ned = np.array([
            self.sorted_values["north_from_observer"],
            self.sorted_values["east_from_observer"]
        ], dtype=float)

        cur_time = self.sorted_values["current_time"]

        if self.prev_ned_for_velocity is not None and self.prev_time_for_velocity is not None:
            dt_vel = cur_time - self.prev_time_for_velocity

            if dt_vel > 1e-3:
                vel_ned = (cur_ned - self.prev_ned_for_velocity) / dt_vel

                raw_speed_n = float(vel_ned[0])
                raw_speed_e = float(vel_ned[1])

                # Reject impossible GPS jumps
                speed_abs = math.hypot(raw_speed_n, raw_speed_e)

                if speed_abs < 3.0:
                    self.sorted_values["speed_n"] = raw_speed_n
                    self.sorted_values["speed_e"] = raw_speed_e
                else:
                    # Keep previous velocity if GPS jump is unreasonable
                    self.sorted_values["speed_n"] = self.sorted_values.get("speed_n", 0.0)
                    self.sorted_values["speed_e"] = self.sorted_values.get("speed_e", 0.0)
            else:
                self.sorted_values["speed_n"] = self.sorted_values.get("speed_n", 0.0)
                self.sorted_values["speed_e"] = self.sorted_values.get("speed_e", 0.0)
        else:
            self.sorted_values["speed_n"] = 0.0
            self.sorted_values["speed_e"] = 0.0

        self.prev_ned_for_velocity = cur_ned.copy()
        self.prev_time_for_velocity = cur_time

        # Heading
        psi_deg = float(self.sorted_values["current_orientation_3"])

        # IMPORTANT:
        # Use this if your IMU yaw is already in the same convention as your model:
        psi_rad = self.wrap_to_pi(math.radians(psi_deg))

        # Use this instead if IMU yaw is compass-like and must be inverted:
        # psi_rad = self.wrap_to_pi(math.radians(-psi_deg))

        self.sorted_values["yaw_rad"] = psi_rad
        self.sorted_values["yaw_deg_wrapped"] = math.degrees(psi_rad)

        # Convert inertial velocity to body-frame velocity
        v_n = self.sorted_values["speed_n"]
        v_e = self.sorted_values["speed_e"]

        self.sorted_values["speed_surge"] = (
            math.cos(psi_rad) * v_n
            + math.sin(psi_rad) * v_e
        )

        self.sorted_values["speed_sway"] = (
            -math.sin(psi_rad) * v_n
            + math.cos(psi_rad) * v_e
        )

        return self.values

    # Takes the otter coordinates and converts it to north east down observed from the observer coordinates
    def geo2ned_position(self):
        n, e, d = pm.geodetic2ned(self.sorted_values["lat"], self.sorted_values["lon"], self.sorted_values["height"], self.sorted_values["observer_lat"], self.sorted_values["observer_lon"], self.sorted_values["observer_height"])
        self.geo2ned_from_observer = [n, e, d]
        self.values["geo2ned_from_observer"] = self.geo2ned_from_observer

    
    # Tries to set the Otter in manual control mode, controlling the x, y and torques. force_y is not in use.
    def set_manual_control_mode(self, force_x, force_y, torque_z):
        if self.check_connection():
            return self.otter_control.set_manual_control_mode(force_x, force_y, torque_z, self.otter_connector)

        else:
            "No connection to Otter"
            return False

    # Takes inputs tau_X and tau_N (N) and returns the control speeds n1 and n2 (rad/s)
    def controlAllocation(self, tau_X, tau_N):
        return self.otter_control.controlAllocation(tau_X, tau_N)

    # Tries to make the Otter enter drift mode. Returns boolean
    def drift(self):
        if self.check_connection():
            return self.otter_control.drift(self.otter_connector)

        else:
            "No connection to Otter"
            return False

    # Tries to set the trusters manually. a and b are individual thrusters and their values range from -1 to 1
    def set_thrusters(self, a, b):
        return self.otter_control.set_thrusters(a, b, self.otter_connector)
    

    # Takes inputs from signals in the form of tau_X (surge) and tau_N (yaw) in N, converts it using control allocation
    # and turns the engines the desired speeds.

    def controller_inputs_torque(self, tau_X, tau_N, surge_setpoint=1, on_linux=False):
        """
        Generic live controller input for PID, NMPC, DRL, etc.

        Input:
            tau_X [N]
            tau_N [Nm]

        Live Otter command:
            $PMARMAN,X_cmd,0,N_cmd

        where:
            X_cmd = tau_X / max_surge_N
            N_cmd = tau_N / max_yaw_N
        """

        tau_X = float(tau_X)
        tau_N = float(tau_N)

        if on_linux:
            if "distance_to_target" in self.sorted_values:
                if self.sorted_values["distance_to_target"] < surge_setpoint:
                    tau_X = 0.0
                    tau_N = 0.0

        X_cmd, N_cmd = self.otter_control.tau_to_live_XN_signal(tau_X, tau_N)

        # Store physical controller outputs
        self.sorted_values["tau_X"] = tau_X
        self.sorted_values["tau_N"] = tau_N

        # Store actual normalized signals sent to PMARMAN
        self.sorted_values["controller_X_cmd"] = X_cmd
        self.sorted_values["controller_N_cmd"] = N_cmd

        # Optional equivalent/debug values
        self.sorted_values["PMARMAN_X"] = X_cmd
        self.sorted_values["PMARMAN_N"] = N_cmd

        if self.verbose:
            print(
                f"Controller input: "
                f"tau_X={tau_X:.3f}, tau_N={tau_N:.3f}, "
                f"X_cmd={X_cmd:.3f}, N_cmd={N_cmd:.3f}"
            )

        return self.otter_control.set_live_XN_signal(
            X_cmd,
            N_cmd,
            self.otter_connector
        )

    # Takes input in radS for each propeller and sends the command to the Otter
    def controller_inputs_radS(self, n1, n2, on_linux=False, surge_setpoint=1):
        """
        Debug/helper only.

        Converts simulated propeller speeds to equivalent physical tau,
        then maps tau to live PMARMAN X/N signal.
        """

        force_x, force_z, speed = self.otter_control.interpolate_force_values(n1, n2, 3)

        if on_linux:
            if "distance_to_target" in self.sorted_values:
                if self.sorted_values["distance_to_target"] < surge_setpoint:
                    force_x = 0.0
                    force_z = 0.0

        return self.controller_inputs_torque(
            force_x,
            force_z,
            on_linux=on_linux,
            surge_setpoint=surge_setpoint
        )

    def testrun(self):
        self.values["current_position"] = [0.0, 0.0, 0.0]
        self.values["current_course_over_ground"] = 45
        cur_time = time.time()
        cycle_time = 0.1
        counter = 0

        self.observer_coordinates = [0.0, 0.0, 0.0]

        while True:

            start_time = time.time()
            self.values["previous_position"] = self.values["current_position"].copy()

            self.values["current_position"][0] = self.values["current_position"][0] + (1/100000)*cycle_time
            self.update_values()

            if counter % 10 == 0:
                print(self.sorted_values["speed_surge"])
                print(self.sorted_values["speed_sway"])

            counter = counter + 1


            if (time.time() - start_time) < cycle_time:
                time.sleep(cycle_time)



    @staticmethod
    def wrap_to_pi(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
    
# This runs if this script is run by itself and not imported into another program. Make sure to have a connection available to the Otter or the test server running.
if __name__ == "__main__":


    otter = otter()


    # Establishes a socket connection to the Otter with IP and the PORT'
    #otter.establish_connection("10.0.5.1", 32001) 
    otter.establish_connection("192.168.53.2", 32001) 


    # Write test commands under here:


