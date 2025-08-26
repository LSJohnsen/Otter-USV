import numpy as np
from numpy import round, pi
import math
from scipy.interpolate import CubicSpline
import pandas as pd
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import os




class otter_control():

    def __init__(self):


        # This enables the printing of messages. Used for debugging. Slows down the software a bit.
        self.verbose = True

        # Scaling values for the engines
        self.scale_x_pos = 0.42
        self.scale_x_neg = 0.45
        self.scale_z_pos = 0.95
        self.scale_z_neg = 0.95

        # Data for one pontoon
        self.B_pont = 0.25  # Beam of one pontoon (m)
        y_pont = 0.395      # Distance from centerline to waterline centroid (m)


        # Propeller configuration/input matrix
        self.l1 = -y_pont  # Lever arm, left propeller (m)
        self.l2 = y_pont  # Lever arm, right propeller (m)
        self.k_pos = 0.02216 / 2  # Positive Bollard, one propeller
        self.k_neg = 0.01289 / 2  # Negative Bollard, one propeller
        B = self.k_pos * np.array([[1, 1], [-self.l1, -self.l2]])
        self.Binv = np.linalg.inv(B)


        # Propeller speed values
        self.max_rpm = 1108
        self.min_rpm = 73
        self.max_radS = (self.max_rpm * (2*pi) / 60)
        self.min_radS = (self.min_rpm * (2*pi) / 60)
        self.radS_spectrum = self.max_radS - self.min_radS
        # Assuming linear throttle
        self.radS_per_percent = self.radS_spectrum / 100

        # Throttle values for the thruster
        rpm_values = [0, 68, 85, 140, 200, 270, 340, 440, 640, 920, 1108]
        throttle_values = [15, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        self.rpm_to_throttle_spline = CubicSpline(rpm_values, throttle_values, extrapolate=False)
        self.throttle_to_rpm_spline = CubicSpline(throttle_values, rpm_values, extrapolate=False)


        # Max values for movement
        self.max_surge_N = 200
        self.max_yaw_N = 115


        self.n1_neg = False
        self.n2_neg = False

        self.thl_neg = False
        self.thr_neg = False


        path = os.path.dirname(os.path.abspath(__file__))
        self.throttle_map_name = os.path.join(path, 'throttle_map_v2_noneg.csv')

        self.throttledf = pd.read_csv(self.throttle_map_name, index_col=0, sep=";")

        self.throttledf = self.throttledf.dropna(axis=1, how='all')

        # Drop rows where all values are NaN
        self.throttledf = self.throttledf.dropna(axis=0, how='all')

        self.rpm_left, self.rpm_right, self.force_x, self.force_z = self.load_and_prepare_data(
            self.throttle_map_name)
        self.tree = cKDTree(np.vstack((self.rpm_left, self.rpm_right)).T)



    # Sets the Otter in drift mode with zero trust
    def drift(self, otter_connector):
        if self.verbose:
            print("Otter entering drift mode")
        message_to_send = "$PMARABT"
        return otter_connector.send_message(message_to_send, False)


    # Sets the Otter in manual mode with specific forces and torques
    # The message must be repeated every 3 seconds or else the otter will enter drift mode.
    def set_manual_control_mode(self, force_x, force_y, torque_z, otter_connector):
        if self.verbose:
            print("Otter entering manual control mode with X force:", force_x, "Y force:", force_y, "Z torque:", torque_z)

        """
        uncomment if not using interpolating throttle map
        
        # Scaling for the values since 0 rpm is a field of values (not only x=0)
        if force_x > 0:
            force_x = (force_x * self.scale_x_pos) + 0.18
        elif force_x < 0:
            force_x = (force_x * self.scale_x_neg) - 0.15
        else:
            force_x = 0.0

        if torque_z > 0.05:
            torque_z = (torque_z * self.scale_z_pos) + 0.05
        elif torque_z < -0.05:
            torque_z = (torque_z * self.scale_z_neg) - 0.05
        else:
            torque_z = 0.0"""


        force_x = str(force_x)
        force_y = str(force_y)
        torque_z = str(torque_z)


        message_to_send = "$PMARMAN," + force_x + "," + force_y + "," + torque_z

        return otter_connector.send_message(message_to_send, True)


    # Sets the trusters manually
    # Calculates the force_x (surge) and torque_z (yaw) "backwards"

    def set_thrusters(self, a, b, otter_connector):

        if self.verbose:
            print("Setting Otter thrusters to", a, b)

        # Calculate linear force
        CG = np.array([0.2, 0, -0.2])
        l_y = 0.395
        l_x = 2.0/2 + CG[0]
        alpha_a = pi/2 - np.arctan2(l_x, l_y)
        alpha_b = pi/2 - np.arctan2(l_x, -l_y)
        c_x = 1/(np.cos(alpha_a)+np.cos(alpha_b))

        self.F_x = c_x * (a * np.cos(alpha_a) + b * np.cos(alpha_b))

        # Calculate rotational force
        beta_a = pi/2 - alpha_a
        beta_b = pi/2 - alpha_b
        c_t = 1/(np.cos(beta_a)-np.cos(beta_b))

        self.F_z = c_t * (a * np.cos(beta_a) + b * (np.cos(beta_b)))

        return self.set_manual_control_mode(self.F_x, 0.0, self.F_z, otter_connector)


    # Takes inputs tau_X and tau_Y (N) and returns the control speeds n1 and n2 (rad/s)
    def controlAllocation(self, tau_X, tau_N):


        # Checks if tau_x or tau_n is over the max N set
        if tau_X > self.max_surge_N:
            tau_X = self.max_surge_N

        if tau_N > self.max_yaw_N:
            tau_N = self.max_yaw_N


        tau = np.array([tau_X, tau_N])  # tau = B * u_alloc
        u_alloc = np.matmul(self.Binv, tau)  # u_alloc = inv(B) * tau

        # u_alloc = abs(n) * n --> n = sign(u_alloc) * sqrt(u_alloc)
        n1 = np.sign(u_alloc[0]) * math.sqrt(abs(u_alloc[0]))
        n2 = np.sign(u_alloc[1]) * math.sqrt(abs(u_alloc[1]))

        return n1, n2


    # Gives a percentage throttle from the rad/s value in n1 and n2 using interpolation
    def radS_to_throttle_interpolation(self, n1, n2):
        if n1 < 0:
            self.n1_neg = True
        if n2 < 0:
            self.n2_neg = True
        if n1 > 0:
            self.n1_neg = False
        if n2 > 0:
            self.n2_neg = False


        n1_rpm = abs(n1) / ((2*pi) / 60)
        n2_rpm = abs(n2) / ((2*pi) / 60)

        if n1_rpm > self.max_rpm:
            n1_rpm = self.max_rpm

        if n2_rpm > self.max_rpm:
            n2_rpm = self.max_rpm

        n1_throttle = self.rpm_to_throttle_spline(n1_rpm) / 100
        n2_throttle = self.rpm_to_throttle_spline(n2_rpm) / 100

        if n1_throttle < 0:
            n1_throttle = n1_throttle * -1
        if n2_throttle < 0:
            n2_throttle = n2_throttle * -1

        if self.n1_neg:
            n1_throttle = n1_throttle * -1
        if self.n2_neg:
            n2_throttle = n2_throttle * -1

    #    n1_throttle = np.round(n1_throttle, 2)
    #    n2_throttle = np.round(n2_throttle, 2)

        return n1_throttle, n2_throttle

    # Returns radS for inputted throttle 0-1
    def throttle_to_rads_interpolation(self, throttle_left, throttle_right):
        if throttle_left < 0:
            self.thl_neg = True
            throttle_left = throttle_left * -1
        else:
            self.thl_neg = False
        if throttle_right < 0:
            self.thr_neg = True
            throttle_right = throttle_right * -1
        else:
            self.thr_neg = False


        if throttle_left >= 0.60:
            throttle_left = 0.60
        if throttle_left <= 0.15:
            throttle_left = 0.15
        if throttle_right >= 0.60:
            throttle_right = 0.60
        if throttle_right <= 0.15:
            throttle_right = 0.15



        n1 = self.throttle_to_rpm_spline(throttle_left * 100)
        n2 = self.throttle_to_rpm_spline(throttle_right * 100)

        n1 = n1 * ((2 * math.pi)/60)
        n2 = n2 * ((2 * math.pi) / 60)

        if n1 < 0:
            n1 = n1 * -1
        if n2 < 0:
            n2 = n2 * -1

        if self.thl_neg:
            n1 = n1 * -1
        if self.thr_neg:
            n2 = n2 * -1

        return n1, n2

    # Gives throttle using a linear throttle percentage
    def radS_to_throttle_linear(self, n1, n2):
        throttle_left = n1 / self.radS_per_percent
        throttle_right = n2 / self.radS_per_percent

        throttle_left = throttle_left / 100
        throttle_right = throttle_right / 100

        return throttle_left, throttle_right


    # Finds the closest throttle values from the inputet rpm's in the throttle map
    def find_closest(self, input_value):

        target_x, target_y = map(float, input_value.strip("()").split(';'))

        target_x = (target_x*60)/(2*math.pi)
        target_y = (target_y * 60) / (2 * math.pi)


        closest_distance = float('inf')
        closest_indices = None

        for column in self.throttledf.columns:
            for row_index, value in self.throttledf[column].items():
                if pd.notna(value):
                    cell_x, cell_y = map(float, value.split(';'))
                    distance = np.sqrt((cell_x - target_x) ** 2 + (cell_y - target_y) ** 2)

                    if distance < closest_distance:
                        closest_distance = distance
                        closest_indices = (column, row_index)
                        speed = self.throttledf[f"{float(column):.2f}"][float(row_index)]



        return closest_indices, speed

    # Loads and prepares data for interpolating throttle 2D throttle map
    def load_and_prepare_data(self, csv_file_path):
        df = pd.read_csv(csv_file_path, delimiter=';', quotechar='"', index_col=0)
        force_x, force_z = np.meshgrid(df.index.astype(float), df.columns.astype(float), indexing='ij')

        rpm_left = []
        rpm_right = []
        for i, row in enumerate(df.itertuples(index=False)):
            for j, cell in enumerate(row):
                if pd.notna(cell):
                    l_rpm, r_rpm = map(float, cell.split(';'))
                    rpm_left.append(l_rpm)
                    rpm_right.append(r_rpm)

        return np.array(rpm_left), np.array(rpm_right), force_x.ravel(), force_z.ravel()

    # Interpolates throttle values in 2D using the throttle map. Adjust k value for how many "neighboring" values to look at for interpolation
    def interpolate_force_values(self, rads_left, rads_right, k=3):
        rpm_left = (rads_left * 60) / (2 * math.pi)
        rpm_right = (rads_right * 60) / (2 * math.pi)



        distances, indices = self.tree.query([(rpm_left, rpm_right)], k)
        weights = 1 / (distances[0] + 1e-10)
        normalized_weights = weights / np.sum(weights)

        force_x_interp = np.sum(normalized_weights * self.force_x[indices[0]])
        force_z_interp = np.sum(normalized_weights * self.force_z[indices[0]])




        interpolated_rpm_left = np.sum(normalized_weights * self.rpm_left[indices[0]])
        interpolated_rpm_right = np.sum(normalized_weights * self.rpm_right[indices[0]])

        speed = [(interpolated_rpm_left*2*math.pi)/60, (interpolated_rpm_right*2*math.pi)/60]

        return force_x_interp, force_z_interp, speed



    # Applies emergency brakes using reverse trusting until the speed of the Otter is below zero
    def EMERGENCY_BRAKES(self, otter_connector):
        print("APPLYING EMERGENCY BRAKES")

        self.set_thrusters(-1, -1)

        # Checks if the speed is above zero. If above zero then it will update the values until zero speed or lower is reached.
        while otter_connector.current_speed > 0:
            otter_connector.update_values()

        print("Otter stopped")
        print("Entering drift mode")

        # Enters drift mode when the speed is zero or lower.
        self.drift()


if __name__ == "__main__":
    otter = otter_control()