import lib.Control as Control
import lib.Connector as Connector
import time
import pymap3d as pm
import math


#
#   This is the complete API for the Otter. This can be imported in custom programs and then the functions can be called to communicate with the otter.
#

class otter():

    def __init__(self):

        self.verbose = True

        # Creates instances of the connector and control classes
        self.otter_connector = Connector.otter_connector()
        self.otter_control = Control.otter_control()


        # The observer coordinates for the geodetic to ned conversion. This can be changed manually
        self.observer_coordinates = [59.908642666666665, 10.71945885, 0.0]


        # Creates an empty dictionary for the values
        self.values = {}
        self.sorted_values = {}


        # Variables and lists
        self.geo2ned_from_observer = [0.0, 0.0, 0.0]

        self.tau_N_neg = False

        self.sorted_values["current_time"] = time.time()

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
        self.otter_connector.update_values(timeout = 0.1)
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
        #self.values["yaw"] = self.otter_connector.yaw


        # Sorts all the data into individuall int and float values in a dictionary sorted_values
        self.sorted_values["lat"] = self.values["current_position"][0]
        self.sorted_values["lon"] = self.values["current_position"][1]
        self.sorted_values["height"] = self.values["current_position"][2]
        self.sorted_values["previous_lat"] = self.values["previous_position"][0]
        self.sorted_values["previous_lon"] = self.values["previous_position"][1]
        self.sorted_values["previous_height"] = self.values["previous_position"][2]
        self.sorted_values["last_speed_update"] = self.values["last_speed_update"]
        self.sorted_values["current_course_over_ground"] = self.values["current_course_over_ground"]        #DEG, 0 deg is north.
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

        self.geo2ned_position()

        self.sorted_values["north_from_observer"] = self.values["geo2ned_from_observer"][0]
        self.sorted_values["east_from_observer"] = self.values["geo2ned_from_observer"][1]
        self.sorted_values["down_from_observer"] = self.values["geo2ned_from_observer"][2]

        self.sorted_values["previous_time"] = self.sorted_values["current_time"]
        self.sorted_values["current_time"] = time.time()
        self.sorted_values["cycle_time"] = self.sorted_values["current_time"] - self.sorted_values["previous_time"]

        prev_pos_ned = pm.geodetic2ned(self.sorted_values["previous_lat"], self.sorted_values["previous_lon"], self.sorted_values["previous_height"], self.sorted_values["observer_lat"], self.sorted_values["observer_lon"], self.sorted_values["observer_height"])
        cur_pos_ned = pm.geodetic2ned(self.sorted_values["lat"], self.sorted_values["lon"], self.sorted_values["height"], self.sorted_values["observer_lat"], self.sorted_values["observer_lon"], self.sorted_values["observer_height"])

        diff_n = cur_pos_ned[0] - prev_pos_ned[0]
        diff_e = cur_pos_ned[1] - prev_pos_ned[1]



        self.sorted_values["speed_n"] = diff_n / self.sorted_values["cycle_time"]
        self.sorted_values["speed_e"] = diff_e / self.sorted_values["cycle_time"]

        self.sorted_values["speed_surge"] = self.sorted_values["speed_n"] * math.cos(self.sorted_values["current_course_over_ground"] * (math.pi/180))
        self.sorted_values["speed_sway"] = self.sorted_values["speed_n"] * -math.sin(self.sorted_values["current_course_over_ground"] * (math.pi/180))


        return self.values

    # Takes the otter coordinates and converts it to north east down observed from the observer coordinates
    def geo2ned_position(self):
        n, e, d = pm.geodetic2ned(self.sorted_values["lat"], self.sorted_values["lon"], self.sorted_values["height"], self.sorted_values["observer_lat"], self.sorted_values["observer_lon"], self.sorted_values["observer_height"])
        self.geo2ned_from_observer = [n, e, d]

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
    def controller_inputs_torque(self, tau_X, tau_N, on_linux=False, surge_setpoint=1):


        # Inverses the yaw if it is negative because of the throttle map
        if tau_N < 0 and tau_X >= 0:
            n1, n2 = self.controlAllocation(tau_X, tau_N*-1)
            self.tau_N_neg = True
            self.tau_X_neg = False
        elif tau_N < 0 and tau_X < 0:
            n1, n2 = self.controlAllocation(tau_X * -1, tau_N * -1)
            self.tau_N_neg = True
            self.tau_X_neg = True
        elif tau_N >= 0 and tau_X < 0:
            n1, n2 = self.controlAllocation(tau_X * -1, tau_N)
            self.tau_N_neg = False
            self.tau_X_neg = True
        else:
            n1, n2 = self.controlAllocation(tau_X, tau_N)
            self.tau_N_neg = False
            self.tau_X_neg = False

        ##### Experimental #####
        if n1 < 0:  #
            n1 = 0.1  #
        if n2 < 0:  # Makes the thursters unable to go in reverse
            n2 = 0.1

        if self.tau_N_neg:
            self.sorted_values["n1"] = (n2*60) /(2*math.pi)                                 # Stores n1 and n2 as rpm for easier plotting
            self.sorted_values["n2"] = (n1*60) /(2*math.pi)
        else:
            self.sorted_values["n1"] = (n1*60) /(2*math.pi)
            self.sorted_values["n2"] = (n2*60) /(2*math.pi)

        #otter_torques, speed = self.otter_control.find_closest(f"{n1};{n2}")
        #torque_x = otter_torques[0]
        #torque_z = otter_torques[1]

        # Use interpolating throttle map or linear throttle map
    #    throttle_left, throttle_right = self.otter_control.radS_to_throttle_linear(n1, n2)
        #throttle_left, throttle_right = self.otter_control.radS_to_throttle_interpolation(n1, n2)           #
        #return self.set_thrusters(throttle_left, throttle_right)                                            #  For interpolating 1D throttle map


        torque_z, torque_x, speed = self.otter_control.interpolate_force_values(n1, n2, 3)                  # Speed is in rads

        if torque_z < 0.05:
            torque_z = 0.0

        if self.tau_N_neg:
            torque_z = torque_z * -1
        if self.tau_X_neg:
            torque_x = torque_x * -1


        #Scipy 2D interpolate has some bugs on linux........
        if on_linux:
            if "distance_to_target" in self.sorted_values:
                if self.sorted_values["distance_to_target"] < surge_setpoint:
                    torque_x = 0
                    torque_z = 0

                #return self.set_manual_control_mode(torque_x, 0.0, torque_z)

        return self.set_manual_control_mode(torque_x, 0.0, torque_z)

    # Takes input in radS for each propeller and sends the command to the Otter
    def controller_inputs_radS(self, n1, n2, on_linux=False, surge_setpoint=1):
        if n1 < n2:
            torque_z, torque_x, speed = self.otter_control.interpolate_force_values(n2, n1, 3)          # Inverts the yaw direciton because of the interpolation map

            # Scipy 2D interpolate has some bugs on linux........
            if on_linux:
                if "distance_to_target" in self.sorted_values:
                    if self.sorted_values["distance_to_target"] < surge_setpoint:
                        torque_x = 0
                        torque_z = 0
            return self.set_manual_control_mode(torque_x, 0.0, torque_z * -1)
        else:
            torque_z, torque_x, speed = self.otter_control.interpolate_force_values(n1, n2, 3)
            # Scipy 2D interpolate has some bugs on linux........
            if on_linux:
                if "distance_to_target" in self.sorted_values:
                    if self.sorted_values["distance_to_target"] < surge_setpoint:
                        torque_x = 0
                        torque_z = 0
            return self.set_manual_control_mode(torque_x, 0.0, torque_z)

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




# This runs if this script is run by itself and not imported into another program. Make sure to have a connection available to the Otter or the test server running.
if __name__ == "__main__":


    otter = otter()


    # Establishes a socket connection to the Otter with IP and the PORT'
    otter.establish_connection("10.0.5.1", 32001) 


    # Write test commands under here:



