import numpy as np
import math
from lib.gnc import Smtrx, Hmtrx, Rzyx, m2c, crossFlowDrag, sat, attitudeEuler, third_order_reference
import pandas as pd
from numba import jit, cuda
from pathlib import Path
from lib.Performance_metrics import PerformanceMetrics
from logs.IO import log_params
from wave_model1 import WaveModel
from wind_model import WindModel

class otter_simulator():

    def __init__(self, target_list,                 # if using list of target coordinates (reaching target creates new target point)
                 use_target_coordinates,            # tracking based on coordinates (dynamic)
                 surge_target_radius,               
                 use_moving_target, 
                 moving_target_start, 
                 moving_target_increase, 
                 end_when_last_target_reached, 
                 verbose, 
                 store_force_file,              
                 circular_target,
                 use_waves=True,
                 use_wind=True):

        # Variable initializations:
        self.use_target_coordinates = use_target_coordinates
        self.use_moving_target = use_moving_target
        self.moving_target_increase = moving_target_increase
        self.moving_target = moving_target_start
        self.target_list = target_list
        self.surge_setpoint = surge_target_radius
        self.last_target = target_list[-1]
        self.end_when_last_target_reached = end_when_last_target_reached
        self.verbose = verbose
        self.store_force_file = store_force_file
        self.circular_target = circular_target
        self.target_circle_start_x = 0
        self.target_circle_start_y = 0
        self.target_radius = 40

        self.use_waves = use_waves
        self.wave_model = None
        self.wave_time = 0.0            # running simulation time for wave phase
        self.use_wind = use_wind
        self.wind_model = None
        self.dp_reverse_radius = 10

        
        #generally Hs 0.02-0.3 and tp 1-2 <- shorter waves
        if self.use_waves:
            self.wave_model = WaveModel(
                Hs=0.5,
                Tp=1.0,
                mean_dir=0.0,
                N=12,
                gain_X=20.0,
                gain_Y=35.0,
                gain_N=8.0,
                spread_std=np.deg2rad(30.0),
                seed=1,
            )
        
        # check correct speeds based on "calm docking"
        if self.use_wind:
            self.wind_model = WindModel(
                mean_speed=2.0,
                mean_dir=np.deg2rad(45.0),
                gust_std=0.5,
                gust_time_constant=5.0,
                Cx=0.8,
                Cy=1.2,
                Cn=0.25,
                A_front=0.15,
                A_side=0.35,
                L_ref=1.0,
                seed=1,
            )
        
        self.max_force = 200                                                                    # Combined max force in yaw and surge. Used for saturation of control forces
        self.V_c = 0.0                                                                          # Starting speed (m/s)
        starting_yaw_angle = 0.0                                                                # Starting yaw angle


        self.distance_to_target = 100                                                           # If not using target coordinates or a moving target, but instead using a surge distance and a heading:
        self.yaw_setpoint = -90                                                                 # If not using target coordinates or a moving target, but instead using a surge distance and a heading:

        self.force_array = np.empty((0, 2), float)

        self.tau_X = 0.0
        self.tau_N = 0.0
        
        
        HERE = Path(__file__).resolve().parent
        csv_path = HERE / "lib" / "throttle_map_v2_noneg.csv"

        self.throttledf = pd.read_csv(csv_path, index_col=0, sep=";")
        self.throttledf = self.throttledf.dropna(axis=1, how='all')
        # Drop rows where all values are NaN
        self.throttledf = self.throttledf.dropna(axis=0, how='all')


        self.n1neg = False
        self.n2neg = False

        # For third order trajectory reference
        self.ref_dist = 0.0
        self.ref_dist_dot = 0.0
        self.ref_dist_ddot = 0.0

        # test zeta/omega for surge reference
        self.zeta_ref = 0.9
        self.omega_n_ref = 0.6

        #performance indices
        self.metrics = PerformanceMetrics()


        ##################################################################################################################################################################################################################
        #        # Below is everything for the simulation of the dynamics of the Otter! This is mostly from "python_vehicle_simulator" authored by Thor I. Fossen.                                                       #
        ##################################################################################################################################################################################################################

        # Constants
        D2R = math.pi / 180     # deg2rad
        self.g = 9.81           # acceleration of gravity (m/s^2)
        rho = 1026              # density of water (kg/m^3)

        self.beta_c = starting_yaw_angle * D2R

        # Initialize the Otter USV model
        self.T_n = 1.0  # propeller time constants (s)
        self.L = 2.0    # Length (m)
        self.B = 1.08   # beam (m)
        self.nu = np.array([0, 0, 0, 0, 0, 0], float)   # velocity vector
        self.u_actual = np.array([0, 0], float)         # propeller revolution states

        self.controls = ["Left propeller shaft speed (rad/s)", "Right propeller shaft speed (rad/s)"]
        self.dimU = len(self.controls)

        # Vehicle parameters
        m = 55.0                                            # mass (kg)
        self.mp = 25.0                                      # Payload (kg)
        self.m_total = m + self.mp
        self.rp = np.array([0.05, 0, -0.35], float)         # location of payload (m)
        rg = np.array([0.2, 0, -0.2], float)                # CG for hull only (m)
        rg = (m * rg + self.mp * self.rp) / (m + self.mp)   # CG corrected for payload
        self.S_rg = Smtrx(rg)
        self.H_rg = Hmtrx(rg)
        self.S_rp = Smtrx(self.rp)

        R44 = 0.4 * self.B  # radii of gyration (m)
        R55 = 0.25 * self.L
        R66 = 0.25 * self.L
        T_yaw = 1.0         # time constant in yaw (s)
        Umax = 6 * 0.5144   # max forward speed (m/s)


        # Data for one pontoon
        self.B_pont = 0.25  # beam of one pontoon (m)
        y_pont = 0.395      # distance from centerline to waterline centroid (m)
        Cw_pont = 0.75      # waterline area coefficient (-)
        Cb_pont = 0.4       # block coefficient, computed from m = 55 kg

        # Inertia dyadic, volume displacement and draft
        nabla = (m + self.mp) / rho  # volume
        self.T = nabla / (2 * Cb_pont * self.B_pont * self.L)  # draft
        Ig_CG = m * np.diag(np.array([R44 ** 2, R55 ** 2, R66 ** 2]))
        self.Ig = Ig_CG - m * self.S_rg @ self.S_rg - self.mp * self.S_rp @ self.S_rp

        # Experimental propeller data including lever arms
        self.l1 = -y_pont  # lever arm, left propeller (m)
        self.l2 = y_pont  # lever arm, right propeller (m)
        self.k_pos = 0.02216 / 2  # Positive Bollard, one propeller
        self.k_neg = 0.01289 / 2  # Negative Bollard, one propeller
        self.n_max = math.sqrt((0.5 * 24.4 * self.g) / self.k_pos)  # max. prop. rev.
        self.n_min = -math.sqrt((0.5 * 13.6 * self.g) / self.k_neg) # min. prop. rev.

        # MRB_CG = [ (m+mp) * I3  O3      (Fossen 2021, Chapter 3)
        #               O3       Ig ]
        MRB_CG = np.zeros((6, 6))
        MRB_CG[0:3, 0:3] = (m + self.mp) * np.identity(3)
        MRB_CG[3:6, 3:6] = self.Ig
        MRB = self.H_rg.T @ MRB_CG @ self.H_rg

        # Hydrodynamic added mass (best practice)
        Xudot = -0.1 * m
        Yvdot = -1.5 * m
        Zwdot = -1.0 * m
        Kpdot = -0.2 * self.Ig[0, 0]
        Mqdot = -0.8 * self.Ig[1, 1]
        Nrdot = -1.7 * self.Ig[2, 2]

        self.MA = -np.diag([Xudot, Yvdot, Zwdot, Kpdot, Mqdot, Nrdot])

        # System mass matrix
        self.M = MRB + self.MA
        self.Minv = np.linalg.inv(self.M)

        # Hydrostatic quantities (Fossen 2021, Chapter 4)
        Aw_pont = Cw_pont * self.L * self.B_pont  # waterline area, one pontoon
        I_T = (
            2
            * (1 / 12)
            * self.L
            * self.B_pont ** 3
            * (6 * Cw_pont ** 3 / ((1 + Cw_pont) * (1 + 2 * Cw_pont)))
            + 2 * Aw_pont * y_pont ** 2
        )
        I_L = 0.8 * 2 * (1 / 12) * self.B_pont * self.L ** 3
        KB = (1 / 3) * (5 * self.T / 2 - 0.5 * nabla / (self.L * self.B_pont))
        BM_T = I_T / nabla  # BM values
        BM_L = I_L / nabla
        KM_T = KB + BM_T    # KM values
        KM_L = KB + BM_L
        KG = self.T - rg[2]
        GM_T = KM_T - KG    # GM values
        GM_L = KM_L - KG

        G33 = rho * self.g * (2 * Aw_pont)  # spring stiffness
        G44 = rho * self.g * nabla * GM_T
        G55 = rho * self.g * nabla * GM_L
        G_CF = np.diag([0, 0, G33, G44, G55, 0])  # spring stiff. matrix in CF
        LCF = -0.2
        H = Hmtrx(np.array([LCF, 0.0, 0.0]))  # transform G_CF from CF to CO
        self.G = H.T @ G_CF @ H

        # Natural frequencies
        w3 = math.sqrt(G33 / self.M[2, 2]) #heave
        w4 = math.sqrt(G44 / self.M[3, 3]) #roll
        w5 = math.sqrt(G55 / self.M[4, 4]) #pitch


        # Linear damping terms (hydrodynamic derivatives)
        Xu = -24.4 *self.g / Umax  # specified using the maximum speed
        Yv = 0
        Zw = -2 * 0.3 * w3 * self.M[2, 2]  # specified using relative damping 
        Kp = -2 * 0.2 * w4 * self.M[3, 3]  # sqrt(g44/m44)  
        Mq = -2 * 0.4 * w5 * self.M[4, 4]
        Nr = -self.M[5, 5] / T_yaw  # specified by the time constant T_yaw

        self.D = -np.diag([Xu, Yv, Zw, Kp, Mq, Nr])

        self.mass = m + self.mp

    def simulate(self, N, sampleTime, otter, surge_PID, yaw_PID, trajectory_reference=True):
        
        counter = 0                         #
        reached_target_time = 0             #
        self.reached_yaw_target_time = 0    #  For tuning, prints time in console
        finished = False                    #
        finished_yaw = False                #
        asd = 0

        yaw_setpoint = 0                    # Heading setpoint, this will be updated in the loop if using a target


        DOF = 6  # degrees of freedom
        t = 0  # initial simulation time

        # Initial state vectors
        eta = np.array([0, 0, 0, 0, 0, 0], float)   # position/attitude, user editable, eta[0] = north, eta[1] = east, eta[5] = yaw angle
        nu = self.nu                                # velocity
        u_actual = self.u_actual                    # actual inputs

        # Intitial target array
        self.targetData = np.array([self.moving_target[0], self.moving_target[1]])


        # Table used to store the simulation data
        simData = np.empty([0, 2 * DOF + 2 * self.dimU], float)
        self.metrics.reset()

        # Sets the first target from the target list
        self.target_counter = 0
        self.target_coordinates = self.target_list[self.target_counter]

        # total distance throughout simulation (for metrics)
        dist_tot = 0

        
        # Main simulation loop
        distanceHistory = 0
        heading_error = 0.0
        i = 0
        self.target_counter = 0
        self.target_coordinates = self.target_list[self.target_counter]
        self.stationary_target = np.array([self.moving_target[0], self.moving_target[1]], dtype=float)
        while i < (N + 1):
            t = i * sampleTime

            # Select active target
            if self.use_moving_target:
                # Update moving target position
                if not self.circular_target:
                    if counter % (1 / sampleTime) == 0:
                        if 15000 <= counter < 25000:
                            self.moving_target[0] += self.moving_target_increase[0]
                            self.moving_target[1] -= self.moving_target_increase[1]
                        elif 25000 <= counter < 35000:
                            self.moving_target[0] -= self.moving_target_increase[0] / 4
                            self.moving_target[1] -= self.moving_target_increase[1] / 4
                        elif 35000 <= counter < 50000:
                            self.moving_target[0] -= self.moving_target_increase[0] * 4
                        elif counter > 50000:
                            pass
                        else:
                            self.moving_target[0] += self.moving_target_increase[0]
                            self.moving_target[1] += self.moving_target_increase[1]
                else:
                    omega = 1.5 / self.target_radius
                    asd += sampleTime
                    theta = omega * asd
                    self.moving_target[0] = self.target_circle_start_x + self.target_radius * np.cos(theta)
                    self.moving_target[1] = self.target_circle_start_y - 20 + self.target_radius * np.sin(theta)

                target_north = self.moving_target[0]
                target_east = self.moving_target[1]

            elif self.use_target_coordinates:
                # Only use waypoint/coordinate trajectory if explicitly enabled
                target_north = self.target_coordinates[0]
                target_east = self.target_coordinates[1]

            else:
                # Fixed stationary target
                # Make sure self.stationary_target exists, e.g.
                # self.stationary_target = np.array([x_target, y_target], dtype=float)
                target_north = self.stationary_target[0]
                target_east = self.stationary_target[1]

            

            # Distance from vessel to active target
            north_distance = target_north - eta[0]
            east_distance = target_east - eta[1]
            raw_distance = math.sqrt(north_distance**2 + east_distance**2)

            dist_tot += raw_distance

            # Waypoint switching only when waypoint tracking is explicitly enabled
            if self.use_target_coordinates and not self.use_moving_target:
                if raw_distance < self.surge_setpoint and (self.target_counter + 1) < len(self.target_list):
                    self.target_counter += 1
                    self.target_coordinates = self.target_list[self.target_counter]

                    target_north = self.target_coordinates[0]
                    target_east = self.target_coordinates[1]
                    north_distance = target_north - eta[0]
                    east_distance = target_east - eta[1]
                    raw_distance = math.sqrt(north_distance**2 + east_distance**2)

                if self.end_when_last_target_reached:
                    if self.target_coordinates == self.last_target and raw_distance < self.surge_setpoint:
                        i = N
                        print(f"Time is: {counter * sampleTime}s!")

            # Optional trajectory shaping for approach
            use_ref = False

            if trajectory_reference:
                if self.use_moving_target:
                    use_ref = True
                else:
                    use_ref = raw_distance > 3

            if use_ref:
                self.ref_dist, self.ref_dist_dot, self.ref_dist_ddot = third_order_reference(
                    self.ref_dist,
                    self.ref_dist_dot,
                    self.ref_dist_ddot,
                    raw_distance,
                    self.zeta_ref,
                    self.omega_n_ref,
                    sampleTime
                )
                self.distance_to_target = self.ref_dist
            else:
                self.distance_to_target = raw_distance

            angle = eta[5]

            # Hover / station-keeping when close to the target
            if raw_distance <= self.surge_setpoint:
                self.distance_to_target = 0
                self.yaw_setpoint = angle
            elif raw_distance <= 5 and not self.use_moving_target:
                pass
            else:
                self.yaw_setpoint = math.atan2(east_distance, north_distance)

            heading_error = (self.yaw_setpoint - angle + np.pi) % (2 * np.pi) - np.pi

            if raw_distance < 5:
                target_is_behind = abs(heading_error) > (np.pi / 2)
                if target_is_behind:
                    heading_scale = 1.0   # reverse zone active, let surge run
                else:
                    # normal fine-approach scaling
                    if abs(heading_error) > np.deg2rad(35):
                        heading_scale = 0.0
                    elif abs(heading_error) > np.deg2rad(15):
                        heading_scale = 0.3
                    else:
                        heading_scale = 1.0
            else:
                heading_scale = max(0.0, np.cos(heading_error))

            if i % 5 == 0:
                # raw controller outputs
                tau_X_cmd = surge_PID.calculate_surge(
                    self.surge_setpoint,
                    self.distance_to_target,
                    self.yaw_setpoint,
                    angle
                )

                tau_N_cmd = yaw_PID.calculate_yaw(
                    self.yaw_setpoint,
                    angle,
                    self.surge_setpoint,
                    self.distance_to_target
                )

                # reduce surge when heading is poor
                self.tau_X = tau_X_cmd * heading_scale
                self.tau_N = tau_N_cmd

            else:
                # hold previous commands between controller updates
                self.tau_X = self.tau_X
                self.tau_N = self.tau_N


            # Optional broad safety clipping only
            self.tau_X = np.clip(self.tau_X, -self.max_force, self.max_force)
            self.tau_N = np.clip(self.tau_N, -self.max_force, self.max_force)


            if self.store_force_file:
                forces = np.array([self.tau_X, self.tau_N])
                self.force_array = np.vstack((self.force_array, forces))


            # Signed allocation
            # controlAllocation must accept signed tau_N and return signed n1, n2
            n1, n2 = otter.controlAllocation(self.tau_X, self.tau_N)


            # Use signed propeller speeds directly.
            u_control = np.array([n1, n2], dtype=float)


            # Store simulation data in simData
            signals = np.append(np.append(np.append(eta, nu), u_control), u_actual)
            simData = np.vstack([simData, signals])

            # Propagate vehicle and attitude dynamics
            [nu, u_actual] = self.dynamics(eta, nu, u_actual, u_control, sampleTime)
            eta = attitudeEuler(eta, nu, sampleTime)

            # Counts and prints the current number of simulation
            counter = counter +1

            # Prints if target is reached used for tuning and debugging
            if self.verbose:
                # Prints every 100 samples simulated
                if counter % 100 == 0:
                    print(f"Running #{counter}")

                # Stores time taken to reach desired target, used for tuning and debugging
                if self.distance_to_target < self.surge_setpoint+2 and not finished:
                    reached_target_time = counter * sampleTime
                    finished = True

                # Stores time it took if desired yaw is reached, used for tuning and debugging
                if (angle > 3.12 or angle < -3.12) and not finished_yaw:
                    self.reached_yaw_target_time = counter * sampleTime
                    finished_yaw = True


            newTargetData = [self.moving_target[0], self.moving_target[1]]
            self.targetData = np.vstack([self.targetData, newTargetData])


            # IAE ISU indices from lib
            self.metrics.update(
                distance_to_target=self.distance_to_target,
                heading_error=heading_error,
                u1=u_actual[0],
                u2=u_actual[1],
                dt=sampleTime,
            )
            
            
            
            i = i + 1
        
        

        simTime = np.arange(start=0, stop=t+sampleTime, step=sampleTime)[:, None]
        targetData = self.targetData

        if self.store_force_file:
            np.savetxt("force_array.csv", self.force_array, delimiter=";", header="tau_X;tau_N", comments="")

        if self.verbose:
            self.IAE_dist, self.IAE_head = self.metrics.get_IAE()
            self.ISU = self.metrics.get_ISU()
            self.ISU_normalized = self.metrics.get_ISU_normalized()
            self.IAU = self.metrics.get_IAU()
            print(f"IAE distance = {self.IAE_dist:.2f}")
            print(f"IAE heading  = {self.IAE_head:.2f}")
            print(f"ISU normalized = {self.ISU_normalized:.2f}")
            print(f"ISU = {self.ISU:.2f}")
            print(f"IAU = {self.IAU:.2f}")
            print(f"AVG distance to target = {dist_tot/i:.2f}")
            print(f"Reached target in {reached_target_time:2f}s (0 if target not reached)")
            print(f"Reached yaw target in {self.reached_yaw_target_time:.2f}s")

            param_dict = {
                "Control_method": "PID",
                "IAE_distance": self.IAE_dist,
                "IAE_heading": self.IAE_head,
                "ISU": self.ISU,
                "ISU_normalized": self.ISU_normalized,
                "IAU": self.IAU,
                "avg_distance_to_target": dist_tot / i,
                "reached_target_time": reached_target_time,
                "reached_yaw_target_time": self.reached_yaw_target_time}

            log_params(param_dict, filename="parameters.txt", verbose=self.verbose)

        return (simTime, simData, targetData)


    def simulate_NMPC(self, N, sampleTime, otter, nmpc, control_dt=0.1, trajectory_reference=True):

        counter = 0
        reached_target_time = 0
        self.reached_yaw_target_time = 0
        finished = False
        finished_yaw = False
        asd = 0.0

        DOF = 6
        t = 0.0

        # Initial state
        eta = np.array([0, 0, 0, 0, 0, 0], dtype=float)
        nu = self.nu.copy()
        u_actual = self.u_actual.copy()

        # Initial target logging
        self.targetData = np.array([self.moving_target[0], self.moving_target[1]], dtype=float)
        simData = np.empty([0, 2 * DOF + 2 * self.dimU], dtype=float)

        self.metrics.reset()

        # Waypoint setup only if enabled
        self.target_counter = 0
        if self.use_target_coordinates and len(self.target_list) > 0:
            self.target_coordinates = self.target_list[self.target_counter]

        # Fixed stationary target for station-keeping when not using moving target or waypoint list
        self.stationary_target = np.array([self.moving_target[0], self.moving_target[1]], dtype=float)

        dist_tot = 0.0

        # NMPC timing
        k_per_solve = max(1, int(round(control_dt / sampleTime)))
        last_tau_u = np.zeros(3, dtype=float)   # [X, Y, N]

        i = 0
        while i < (N + 1):
            t = i * sampleTime

            
            # active target 
            if self.use_moving_target:
                if not self.circular_target:
                    if counter % int(round(1 / sampleTime)) == 0:
                        if 15000 <= counter < 25000:
                            self.moving_target[0] += self.moving_target_increase[0]
                            self.moving_target[1] -= self.moving_target_increase[1]
                        elif 25000 <= counter < 35000:
                            self.moving_target[0] -= self.moving_target_increase[0] / 4
                            self.moving_target[1] -= self.moving_target_increase[1] / 4
                        elif 35000 <= counter < 50000:
                            self.moving_target[0] -= self.moving_target_increase[0] * 4
                        elif counter > 50000:
                            pass
                        else:
                            self.moving_target[0] += self.moving_target_increase[0]
                            self.moving_target[1] += self.moving_target_increase[1]
                else:
                    omega = 1.5 / self.target_radius
                    asd += sampleTime
                    theta = omega * asd
                    self.moving_target[0] = self.target_circle_start_x + self.target_radius * np.cos(theta)
                    self.moving_target[1] = self.target_circle_start_y - 20 + self.target_radius * np.sin(theta)

                target_north = self.moving_target[0]
                target_east = self.moving_target[1]

            elif self.use_target_coordinates:
                target_north = self.target_coordinates[0]
                target_east = self.target_coordinates[1]

            else:
                # Stationary target without waypoint list
                target_north = self.stationary_target[0]
                target_east = self.stationary_target[1]

        
            # Distance and waypoint
            north_distance = target_north - eta[0]
            east_distance = target_east - eta[1]
            raw_distance = math.sqrt(north_distance**2 + east_distance**2)

            dist_tot += raw_distance

            if self.use_target_coordinates and not self.use_moving_target:
                if raw_distance < self.surge_setpoint and (self.target_counter + 1) < len(self.target_list):
                    self.target_counter += 1
                    self.target_coordinates = self.target_list[self.target_counter]

                    target_north = self.target_coordinates[0]
                    target_east = self.target_coordinates[1]
                    north_distance = target_north - eta[0]
                    east_distance = target_east - eta[1]
                    raw_distance = math.sqrt(north_distance**2 + east_distance**2)

                if self.end_when_last_target_reached:
                    if np.allclose(self.target_coordinates, self.last_target) and raw_distance < self.surge_setpoint:
                        print(f"Time is: {counter * sampleTime}s!")
                        break

            
            # reference shaping - ignore traj reference for stationary
            use_ref = trajectory_reference and self.use_moving_target

            if use_ref:
                self.ref_dist, self.ref_dist_dot, self.ref_dist_ddot = third_order_reference(
                    self.ref_dist,
                    self.ref_dist_dot,
                    self.ref_dist_ddot,
                    raw_distance,
                    self.zeta_ref,
                    self.omega_n_ref,
                    sampleTime
                )
                self.distance_to_target = self.ref_dist
                
            else:
                self.distance_to_target = raw_distance

            # Station-keeping yaw behavior for metrics/plotting
            
            angle = eta[5]
            if self.use_moving_target:
                # never zero distance for moving target
                self.yaw_setpoint = math.atan2(east_distance, north_distance)
            else:
                if raw_distance <= self.surge_setpoint:
                    self.distance_to_target = 0.0
                    self.yaw_setpoint = angle
                else:
                    self.yaw_setpoint = math.atan2(east_distance, north_distance)

            heading_error = (self.yaw_setpoint - angle + np.pi) % (2 * np.pi) - np.pi

        
            # NMPC update
            if i % k_per_solve == 0:
                # x = [x, y, psi, u, v, r]
                x3dof = np.array([
                    eta[0], eta[1], eta[5],
                    nu[0],  nu[1],  nu[5]
                ], dtype=float)

                # Build NMPC target reference:
                # target_ref = [target_north, target_east, path_heading, target_v_north, target_v_east]

                if self.use_moving_target and not self.circular_target:
                    v_north = float(self.moving_target_increase[0])
                    v_east = float(self.moving_target_increase[1])

                    if np.hypot(v_north, v_east) > 1e-6:
                        path_heading = math.atan2(v_east, v_north)
                    else:
                        path_heading = math.atan2(east_distance, north_distance)

                elif self.use_moving_target and self.circular_target:
                    omega = 1.5 / self.target_radius
                    theta = omega * asd

                    v_north = -self.target_radius * omega * math.sin(theta)
                    v_east = self.target_radius * omega * math.cos(theta)

                    if np.hypot(v_north, v_east) > 1e-6:
                        path_heading = math.atan2(v_east, v_north)
                    else:
                        path_heading = math.atan2(east_distance, north_distance)

                else:
                    # Station-keeping / fixed target
                    v_north = 0.0
                    v_east = 0.0
                    path_heading = math.atan2(east_distance, north_distance)


                # Use trajectory-shaped position reference for moving targets if enabled
                if trajectory_reference and self.use_moving_target:
                    direction = np.array([north_distance, east_distance], dtype=float)
                    norm_dir = np.linalg.norm(direction)

                    if norm_dir > 1e-6:
                        unit_dir = direction / norm_dir
                        ref_dist_used = min(self.ref_dist, raw_distance)

                        target_ref = np.array([
                            eta[0] + ref_dist_used * unit_dir[0],
                            eta[1] + ref_dist_used * unit_dir[1],
                            path_heading,
                            v_north,
                            v_east,
                        ], dtype=float)
                    else:
                        target_ref = np.array([
                            target_north,
                            target_east,
                            path_heading,
                            v_north,
                            v_east,
                        ], dtype=float)
                else:
                    target_ref = np.array([
                        target_north,
                        target_east,
                        path_heading,
                        v_north,
                        v_east,
                    ], dtype=float)


                try:
                    tau_u = nmpc.solve_control(x3dof, target_ref)
                    last_tau_u = tau_u.copy()
                except Exception as e:
                    print("NMPC solve failed:", e)
                    tau_u = last_tau_u

            # Hold previous NMPC command between solves
            # last_tau_u[0] = surge force X [N]
            # last_tau_u[2] = yaw moment N [Nm]
            self.tau_X = float(last_tau_u[0])
            self.tau_N = float(last_tau_u[2])   # ignore sway force

            # safety clipping only
            # This is not the final actuator constraint; controlAllocation handles coupled feasibility.
            self.tau_X = np.clip(self.tau_X, -self.max_force, self.max_force)
            self.tau_N = np.clip(self.tau_N, -self.max_force, self.max_force)

            if self.store_force_file:
                forces = np.array([self.tau_X, self.tau_N])
                self.force_array = np.vstack((self.force_array, forces))

            # Allocate signed generalized force/moment to signed propeller speeds
            n1, n2 = otter.controlAllocation(self.tau_X, self.tau_N)

            # Use signed propeller speeds directly
            u_control = np.array([n1, n2], dtype=float)


            # Store data and propagate dynamics
            signals = np.append(np.append(np.append(eta, nu), u_control), u_actual)
            simData = np.vstack([simData, signals])

            [nu, u_actual] = self.dynamics(eta, nu, u_actual, u_control, sampleTime)
            eta = attitudeEuler(eta, nu, sampleTime)

            self.metrics.update(
                distance_to_target=self.distance_to_target,
                heading_error=heading_error,
                u1=u_actual[0],
                u2=u_actual[1],
                dt=sampleTime,
            )

            # Logging / progress
            if self.verbose:
                if counter % 100 == 0:
                    print(f"Running #{counter}")

                if self.distance_to_target < self.surge_setpoint + 2 and not finished:
                    reached_target_time = counter * sampleTime
                    finished = True

                if (angle > 3.12 or angle < -3.12) and not finished_yaw:
                    self.reached_yaw_target_time = counter * sampleTime
                    finished_yaw = True

            # Log active target position
            newTargetData = [target_north, target_east]
            self.targetData = np.vstack([self.targetData, newTargetData])

            counter += 1
            i += 1

        simTime = np.arange(start=0, stop=t + sampleTime, step=sampleTime)[:, None]
        targetData = self.targetData

        if self.store_force_file:
            np.savetxt("force_array.csv", self.force_array, delimiter=";", header="tau_X;tau_N", comments="")

        if self.verbose:
            self.IAE_dist, self.IAE_head = self.metrics.get_IAE()
            self.ISU = self.metrics.get_ISU()
            self.ISU_normalized = self.metrics.get_ISU_normalized()
            self.IAU = self.metrics.get_IAU()

            print(f"IAE distance = {self.IAE_dist:.2f}")
            print(f"IAE heading  = {self.IAE_head:.2f}")
            print(f"ISU normalized = {self.ISU_normalized:.2f}")
            print(f"ISU = {self.ISU:.2f}")
            print(f"IAU = {self.IAU:.2f}")
            print(f"AVG distance to target = {dist_tot / max(i, 1):.2f}")
            print(f"Reached target in {reached_target_time:.2f}s (0 if target not reached)")
            print(f"Reached yaw target in {self.reached_yaw_target_time:.2f}s")

            param_dict = {
                "Control_method": "NMPC",
                "IAE_distance": self.IAE_dist,
                "IAE_heading": self.IAE_head,
                "ISU": self.ISU,
                "ISU_normalized": self.ISU_normalized,
                "IAU": self.IAU,
                "avg_distance_to_target": dist_tot / max(i, 1),
                "reached_target_time": reached_target_time,
                "reached_yaw_target_time": self.reached_yaw_target_time,
            }

            log_params(param_dict, filename="parameters.txt", verbose=self.verbose)

        return (simTime, simData, targetData)
    

    def dynamics(self, eta, nu, u_actual, u_control, sampleTime):
        """
        [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime) integrates
        the Otter USV equations of motion using Euler's method.
        """

        # Input vector
        n = np.array([u_actual[0], u_actual[1]])

        # Current velocities
        u_c = self.V_c * math.cos(self.beta_c - eta[5])  # current surge vel.
        v_c = self.V_c * math.sin(self.beta_c - eta[5])  # current sway vel.

        nu_c = np.array([u_c, v_c, 0, 0, 0, 0], float)  # current velocity vector
        Dnu_c = np.array([nu[5]*v_c, -nu[5]*u_c, 0, 0, 0, 0],float) # derivative
        nu_r = nu - nu_c  # relative velocity vector

        # Rigid body and added mass Coriolis and centripetal matrices
        # CRB_CG = [ (m+mp) * Smtrx(nu2)          O3   (Fossen 2021, Chapter 6)
        #              O3                   -Smtrx(Ig*nu2)  ]
        CRB_CG = np.zeros((6, 6))
        CRB_CG[0:3, 0:3] = self.m_total * Smtrx(nu[3:6])
        CRB_CG[3:6, 3:6] = -Smtrx(np.matmul(self.Ig, nu[3:6]))
        CRB = self.H_rg.T @ CRB_CG @ self.H_rg  # transform CRB from CG to CO

        CA = m2c(self.MA, nu_r)
        CA[5, 0] = 0  # assume that the Munk moment in yaw can be neglected
        CA[5, 1] = 0  # if nonzero, must be balanced by adding nonlinear damping
        CA[0, 5] = 0
        CA[1, 5] = 0

        C = CRB + CA

        # Payload force and moment expressed in BODY
        R = Rzyx(eta[3], eta[4], eta[5])
        f_payload = np.matmul(R.T, np.array([0, 0, self.mp * self.g], float))
        m_payload = np.matmul(self.S_rp, f_payload)
        g_0 = np.array([ f_payload[0],f_payload[1],f_payload[2],
                         m_payload[0],m_payload[1],m_payload[2] ])

        # Control forces and moments - with propeller revolution saturation
        thrust = np.zeros(2)
        for i in range(0, 2):

            n[i] = sat(n[i], self.n_min, self.n_max)  # saturation, physical limits

            if n[i] > 0:  # positive thrust
                thrust[i] = self.k_pos * n[i] * abs(n[i])
            else:  # negative thrust
                thrust[i] = self.k_neg * n[i] * abs(n[i])

        # Control forces and moments
        tau = np.array(
            [
                thrust[0] + thrust[1],
                0,
                0,
                0,
                0,
                -self.l1 * thrust[0] - self.l2 * thrust[1],
            ]
        )

        # Hydrodynamic linear damping + nonlinear yaw damping
        tau_damp = -np.matmul(self.D, nu_r)
        tau_damp[5] = tau_damp[5] - 10 * self.D[5, 5] * abs(nu_r[5]) * nu_r[5]

        # State derivatives (with dimension)
        tau_crossflow = crossFlowDrag(self.L, self.B_pont, self.T, nu_r)


        # wave forces
        tau_wave = np.zeros(6)
        if self.use_waves and self.wave_model is not None:
            tau_wave = self.wave_model.get_tau_wave(self.wave_time, eta, nu)
            self.wave_time += sampleTime

        # wind forces
        tau_wind = np.zeros(6)
        if self.use_wind and self.wind_model is not None:
            tau_wind = self.wind_model.get_tau_wind(sampleTime, eta, nu)

        sum_tau = (
            tau
            + tau_damp
            + tau_crossflow
            + tau_wave
            + tau_wind
            - np.matmul(C, nu_r)
            - np.matmul(self.G, eta)
            + g_0
        )
        
        nu_dot = Dnu_c + np.matmul(self.Minv, sum_tau)  # USV dynamics
        n_dot = (u_control - n) / self.T_n  # propeller dynamics

        # Forward Euler integration [k+1]
        nu = nu + sampleTime * nu_dot
        n = n + sampleTime * n_dot

        u_actual = np.array(n, float)

        

        return nu, u_actual
    


