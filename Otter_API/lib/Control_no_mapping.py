import numpy as np
from numpy import pi
import math
from scipy.interpolate import CubicSpline
import pandas as pd
from scipy.spatial import cKDTree
import os


class otter_control():

    def __init__(self):

        # This enables the printing of messages. Used for debugging.
        self.verbose = True


        # Scaling values from old setup, kept for compatibility

        self.scale_x_pos = 0.42
        self.scale_x_neg = 0.45
        self.scale_z_pos = 0.95
        self.scale_z_neg = 0.95

  
        # Propeller / pontoon data

        self.B_pont = 0.25
        y_pont = 0.395

        # Lever arms
        self.l1 = -y_pont
        self.l2 = y_pont

        # Bollard constants
        self.k_pos = 0.02216 / 2.0
        self.k_neg = 0.01289 / 2.0

        # Allocation matrix for positive-thrust approximation
        B = self.k_pos * np.array([
            [1.0, 1.0],
            [-self.l1, -self.l2]
        ])

        self.Binv = np.linalg.inv(B)


        # Propeller speed values

        self.max_rpm = 1108.0
        self.min_rpm = 73.0

        self.max_radS = self.max_rpm * 2.0 * pi / 60.0
        self.min_radS = self.min_rpm * 2.0 * pi / 60.0

        self.radS_spectrum = self.max_radS - self.min_radS
        self.radS_per_percent = self.radS_spectrum / 100.0


        # Throttle interpolation values, kept for old compatibility

        rpm_values = [0, 68, 85, 140, 200, 270, 340, 440, 640, 920, 1108]
        throttle_values = [15, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60]

        self.rpm_to_throttle_spline = CubicSpline(
            rpm_values,
            throttle_values,
            extrapolate=False
        )

        self.throttle_to_rpm_spline = CubicSpline(
            throttle_values,
            rpm_values,
            extrapolate=False
        )

  
        # Generalized force limits
        # Used for mapping physical controller outputs to PMARMAN signals.

        self.max_surge_N = 200.0
        self.max_yaw_N = 115.0


        # Internal state / logging values
        self.n1_neg = False
        self.n2_neg = False

        self.thl_neg = False
        self.thr_neg = False

        self.F_x = 0.0
        self.F_z = 0.0

        self.n1 = 0.0
        self.n2 = 0.0

        self.X_cmd = 0.0
        self.N_cmd = 0.0


        # Old throttle map loading, kept for compatibility
        path = os.path.dirname(os.path.abspath(__file__))
        self.throttle_map_name = os.path.join(path, "throttle_map_v2_noneg.csv")

        self.throttledf = pd.read_csv(self.throttle_map_name, index_col=0, sep=";")
        self.throttledf = self.throttledf.dropna(axis=1, how="all")
        self.throttledf = self.throttledf.dropna(axis=0, how="all")

        self.rpm_left, self.rpm_right, self.force_x, self.force_z = self.load_and_prepare_data(
            self.throttle_map_name
        )

        self.tree = cKDTree(np.vstack((self.rpm_left, self.rpm_right)).T)

    # ----------------------------------------------------------------------
    # Basic Otter modes
    # ----------------------------------------------------------------------

    def drift(self, otter_connector):
        """
        Sets the Otter in drift mode with zero thrust.
        """

        if self.verbose:
            print("Otter entering drift mode")

        message_to_send = "$PMARABT"

        return otter_connector.send_message(message_to_send, False)

    def set_manual_control_mode(self, force_x, force_y, torque_z, otter_connector):
        """
        Sends PMARMAN.

        Correct live architecture:
            $PMARMAN,X,Y,N

        where:
            X = normalized surge command in [-1, 1]
            Y = normalized sway command, normally 0
            N = normalized yaw command in [-1, 1]

        Do not send physical forces such as 200 or -115 here.
        """

        force_x = float(np.clip(force_x, -1.0, 1.0))
        force_y = float(np.clip(force_y, -1.0, 1.0))
        torque_z = float(np.clip(torque_z, -1.0, 1.0))

        if self.verbose:
            print(
                "Otter entering manual control mode with "
                f"X signal: {force_x:.3f}, "
                f"Y signal: {force_y:.3f}, "
                f"N signal: {torque_z:.3f}"
            )

        message_to_send = f"$PMARMAN,{force_x:.3f},{force_y:.3f},{torque_z:.3f}"

        return otter_connector.send_message(message_to_send, True)

    # Correct live-guidance architecture


    def tau_to_live_XN_signal(self, tau_X, tau_N):
        """
        Converts physical/generalized controller outputs to normalized PMARMAN
        surge/yaw commands.

        Input:
            tau_X : surge force-like command [N]
            tau_N : yaw moment-like command [Nm]

        Output:
            X_cmd : normalized surge command in [-1, 1]
            N_cmd : normalized yaw command in [-1, 1]

        This is the correct live-guidance mapping:
            tau_X, tau_N -> X_cmd, N_cmd -> $PMARMAN,X_cmd,0,N_cmd
        """

        tau_X = float(np.clip(tau_X, -self.max_surge_N, self.max_surge_N))
        tau_N = float(np.clip(tau_N, -self.max_yaw_N, self.max_yaw_N))

        X_cmd = tau_X / self.max_surge_N
        N_cmd = tau_N / self.max_yaw_N

        X_cmd = float(np.clip(X_cmd, -1.0, 1.0))
        N_cmd = float(np.clip(N_cmd, -1.0, 1.0))

        self.X_cmd = X_cmd
        self.N_cmd = N_cmd

        return X_cmd, N_cmd

    def set_live_XN_signal(self, X_cmd, N_cmd, otter_connector):
        """
        Sends an already-normalized live surge/yaw command.

        Input:
            X_cmd in [-1, 1]
            N_cmd in [-1, 1]

        Sends:
            $PMARMAN,X_cmd,0,N_cmd
        """

        X_cmd = float(np.clip(X_cmd, -1.0, 1.0))
        N_cmd = float(np.clip(N_cmd, -1.0, 1.0))

        self.X_cmd = X_cmd
        self.N_cmd = N_cmd

        if self.verbose:
            print(
                f"Setting live X/N signal: "
                f"X={X_cmd:.3f}, N={N_cmd:.3f}"
            )

        return self.set_manual_control_mode(
            X_cmd,
            0.0,
            N_cmd,
            otter_connector
        )

    def set_live_controller_tau(self, tau_X, tau_N, otter_connector):
        """
        Main live-guidance function for PID, NMPC, DRL, etc.

        Input:
            tau_X [N]
            tau_N [Nm]

        Internally:
            tau_X, tau_N
            -> tau_to_live_XN_signal()
            -> X_cmd, N_cmd in [-1, 1]
            -> $PMARMAN,X_cmd,0,N_cmd

        This function does NOT calculate n1/n2, because PMARMAN does not
        accept individual propeller commands.
        """

        X_cmd, N_cmd = self.tau_to_live_XN_signal(tau_X, tau_N)

        if self.verbose:
            print(
                "Live controller tau input: "
                f"tau_X={tau_X:.3f}, tau_N={tau_N:.3f}, "
                f"X_cmd={X_cmd:.3f}, N_cmd={N_cmd:.3f}"
            )

        return self.set_live_XN_signal(
            X_cmd,
            N_cmd,
            otter_connector
        )


    # Simulation architecture

    def controlAllocation(self, tau_X, tau_N):
        """
        Simulation/physical allocation.

        Takes desired tau_X [N] and tau_N [Nm],
        returns signed propeller speeds n1, n2 [rad/s].

        This should be used by the simulator, because the simulator dynamics
        expects propeller shaft speeds.

        Do not use this output directly with PMARMAN.
        """

        tau_X = float(np.clip(tau_X, -self.max_surge_N, self.max_surge_N))
        tau_N = float(np.clip(tau_N, -self.max_yaw_N, self.max_yaw_N))

        tau = np.array([tau_X, tau_N], dtype=float)

        # tau = B @ u_alloc
        # u_alloc = |n|n
        u_alloc = self.Binv @ tau

        n1 = np.sign(u_alloc[0]) * math.sqrt(abs(u_alloc[0]))
        n2 = np.sign(u_alloc[1]) * math.sqrt(abs(u_alloc[1]))

        n1 = float(np.clip(n1, -self.max_radS, self.max_radS))
        n2 = float(np.clip(n2, -self.max_radS, self.max_radS))

        self.n1 = n1
        self.n2 = n2

        return n1, n2


    # Old set_thrusters kept, but now clearly treated as X/N signal input
    def set_thrusters(self, a, b, otter_connector):
        """
        Kept for old API compatibility.

        Important:
            Since PMARMAN is $PMARMAN,X,Y,N, this function now treats:

                a = X surge signal in [-1, 1]
                b = N yaw signal in [-1, 1]

            It does not mean left/right propeller command.
        """

        if self.verbose:
            print("Setting Otter X/N signals to", a, b)

        return self.set_live_XN_signal(a, b, otter_connector)


    # helper functions for debug/simulation comparison
    def live_XN_signal_to_tau(self, X_cmd, N_cmd):
        """
        Converts normalized live X/N signal back to equivalent physical tau.

        Useful for logging/debugging only.
        """

        X_cmd = float(np.clip(X_cmd, -1.0, 1.0))
        N_cmd = float(np.clip(N_cmd, -1.0, 1.0))

        tau_X = X_cmd * self.max_surge_N
        tau_N = N_cmd * self.max_yaw_N

        return tau_X, tau_N

    def radS_to_normalized_debug(self, n1, n2):
        """
        Converts propeller rad/s to normalized values for plotting only.

        This is NOT used for PMARMAN live control.
        """

        n1 = float(np.clip(n1, -self.max_radS, self.max_radS))
        n2 = float(np.clip(n2, -self.max_radS, self.max_radS))

        n1_norm = n1 / self.max_radS
        n2_norm = n2 / self.max_radS

        return n1_norm, n2_norm


    # Old throttle conversion functions, kept for compatibility

    def radS_to_throttle_interpolation(self, n1, n2):
        """
        Gives percentage throttle from rad/s using interpolation.

        Kept for old scripts/debugging.
        """

        if n1 < 0:
            self.n1_neg = True
        if n2 < 0:
            self.n2_neg = True
        if n1 > 0:
            self.n1_neg = False
        if n2 > 0:
            self.n2_neg = False

        n1_rpm = abs(n1) / ((2.0 * pi) / 60.0)
        n2_rpm = abs(n2) / ((2.0 * pi) / 60.0)

        n1_rpm = min(n1_rpm, self.max_rpm)
        n2_rpm = min(n2_rpm, self.max_rpm)

        n1_throttle = self.rpm_to_throttle_spline(n1_rpm) / 100.0
        n2_throttle = self.rpm_to_throttle_spline(n2_rpm) / 100.0

        if n1_throttle < 0:
            n1_throttle = -n1_throttle
        if n2_throttle < 0:
            n2_throttle = -n2_throttle

        if self.n1_neg:
            n1_throttle = -n1_throttle
        if self.n2_neg:
            n2_throttle = -n2_throttle

        return n1_throttle, n2_throttle

    def throttle_to_rads_interpolation(self, throttle_left, throttle_right):
        """
        Returns rad/s for input throttle values.

        Kept for old scripts/debugging.
        """

        if throttle_left < 0:
            self.thl_neg = True
            throttle_left = -throttle_left
        else:
            self.thl_neg = False

        if throttle_right < 0:
            self.thr_neg = True
            throttle_right = -throttle_right
        else:
            self.thr_neg = False

        throttle_left = float(np.clip(throttle_left, 0.15, 0.60))
        throttle_right = float(np.clip(throttle_right, 0.15, 0.60))

        n1 = self.throttle_to_rpm_spline(throttle_left * 100.0)
        n2 = self.throttle_to_rpm_spline(throttle_right * 100.0)

        n1 = n1 * ((2.0 * math.pi) / 60.0)
        n2 = n2 * ((2.0 * math.pi) / 60.0)

        if n1 < 0:
            n1 = -n1
        if n2 < 0:
            n2 = -n2

        if self.thl_neg:
            n1 = -n1
        if self.thr_neg:
            n2 = -n2

        return n1, n2

    def radS_to_throttle_linear(self, n1, n2):
        """
        Gives throttle using a linear throttle percentage.

        Kept for old scripts/debugging.
        """

        throttle_left = n1 / self.radS_per_percent
        throttle_right = n2 / self.radS_per_percent

        throttle_left = throttle_left / 100.0
        throttle_right = throttle_right / 100.0

        return throttle_left, throttle_right


    # Old throttle-map functions, kept for compatibility

    def find_closest(self, input_value):
        """
        Finds closest throttle-map entry from input rad/s pair.

        input_value format:
            "n1;n2"
        """

        target_x, target_y = map(float, input_value.strip("()").split(";"))

        target_x = (target_x * 60.0) / (2.0 * math.pi)
        target_y = (target_y * 60.0) / (2.0 * math.pi)

        closest_distance = float("inf")
        closest_indices = None
        speed = None

        for column in self.throttledf.columns:
            for row_index, value in self.throttledf[column].items():
                if pd.notna(value):
                    cell_x, cell_y = map(float, value.split(";"))

                    distance = np.sqrt(
                        (cell_x - target_x) ** 2
                        + (cell_y - target_y) ** 2
                    )

                    if distance < closest_distance:
                        closest_distance = distance
                        closest_indices = (column, row_index)
                        speed = self.throttledf[f"{float(column):.2f}"][float(row_index)]

        return closest_indices, speed

    def load_and_prepare_data(self, csv_file_path):
        """
        Loads and prepares data for interpolating the old 2D throttle map.
        """

        df = pd.read_csv(csv_file_path, delimiter=";", quotechar='"', index_col=0)

        force_x, force_z = np.meshgrid(
            df.index.astype(float),
            df.columns.astype(float),
            indexing="ij"
        )

        rpm_left = []
        rpm_right = []

        for _, row in enumerate(df.itertuples(index=False)):
            for _, cell in enumerate(row):
                if pd.notna(cell):
                    l_rpm, r_rpm = map(float, cell.split(";"))
                    rpm_left.append(l_rpm)
                    rpm_right.append(r_rpm)

        return (
            np.array(rpm_left),
            np.array(rpm_right),
            force_x.ravel(),
            force_z.ravel()
        )

    def interpolate_force_values(self, rads_left, rads_right, k=3):
        """
        Interpolates force values in 2D using the old throttle map.

        Kept for old scripts/debugging.
        """

        rpm_left = (rads_left * 60.0) / (2.0 * math.pi)
        rpm_right = (rads_right * 60.0) / (2.0 * math.pi)

        distances, indices = self.tree.query([(rpm_left, rpm_right)], k)

        weights = 1.0 / (distances[0] + 1e-10)
        normalized_weights = weights / np.sum(weights)

        force_x_interp = np.sum(normalized_weights * self.force_x[indices[0]])
        force_z_interp = np.sum(normalized_weights * self.force_z[indices[0]])

        interpolated_rpm_left = np.sum(
            normalized_weights * self.rpm_left[indices[0]]
        )
        interpolated_rpm_right = np.sum(
            normalized_weights * self.rpm_right[indices[0]]
        )

        speed = [
            (interpolated_rpm_left * 2.0 * math.pi) / 60.0,
            (interpolated_rpm_right * 2.0 * math.pi) / 60.0
        ]

        return force_x_interp, force_z_interp, speed



    def EMERGENCY_BRAKES(self, otter_connector):
        """
        Applies reverse command until the Otter speed is below zero,
        then enters drift mode.
        """

        print("APPLYING EMERGENCY BRAKES")

        self.set_live_XN_signal(-1.0, 0.0, otter_connector)

        while otter_connector.current_speed > 0:
            otter_connector.update_values()

        print("Otter stopped")
        print("Entering drift mode")

        self.drift(otter_connector)


if __name__ == "__main__":
    otter = otter_control()