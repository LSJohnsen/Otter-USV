
# Simulator object for MPC control based on Otter simulator 


import numpy as np
import math
from lib.gnc import Smtrx, Hmtrx, Rzyx, m2c, crossFlowDrag, sat, attitudeEuler
import pandas as pd
from numba import jit, cuda
from pathlib import Path

class otter_simulator():

    def __init__(self, 
                 target_list, 
                 use_target_coordinates, 
                 surge_target_radius, 
                 use_moving_target, 
                 moving_target_start, 
                 moving_target_increase, 
                 end_when_last_target_reached, 
                 verbose, 
                 store_force_file, 
                 circular_target):

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
        self.mp = 25.0                                       # Payload (kg)
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


    def simulate_NMPC(self, N, sampleTime, otter, nmpc, control_dt=0.1):

        counter = 0
        reached_target_time = 0
        self.reached_yaw_target_time = 0
        finished = False
        finished_yaw = False
        asd = 0.0

        DOF = 6
        t = 0.0

        # Initial state
        eta = np.array([0, 0, 0, 0, 0, 0], float)
        nu = self.nu.copy()
        u_actual = self.u_actual.copy()

        # target data
        self.targetData = np.array([self.moving_target[0], self.moving_target[1]])
        simData = np.empty([0, 2 * DOF + 2 * self.dimU], float)

        # first target
        self.target_counter = 0
        self.target_coordinates = self.target_list[self.target_counter]
        dist_tot = 0.0

        # NMPC timing
        k_per_solve = max(1, int(round(control_dt / sampleTime)))
        last_tau_u = np.zeros(3)  # [X, Y, N]

        for i in range(N + 1):
            t = i * sampleTime

            # ------------------ target handling (same as simulate) ------------------
            if self.use_target_coordinates:
                north_distance = self.target_coordinates[0] - eta[0]
                east_distance  = self.target_coordinates[1] - eta[1]
                self.distance_to_target = math.sqrt(north_distance**2 + east_distance**2)

                # switch to next target
                if self.distance_to_target < self.surge_setpoint and (self.target_counter + 1) < len(self.target_list):
                    self.target_counter += 1
                    self.target_coordinates = self.target_list[self.target_counter]
                    north_distance = self.target_coordinates[0] - eta[0]
                    east_distance  = self.target_coordinates[1] - eta[1]
                    self.distance_to_target = math.sqrt(north_distance**2 + east_distance**2)

                # end when last target reached
                if self.end_when_last_target_reached and self.target_coordinates is self.last_target:
                    if self.distance_to_target < self.surge_setpoint:
                        print(f"Time is: {counter * sampleTime}s!")
                        break

                self.yaw_setpoint = math.atan2(east_distance, north_distance)

            if self.use_moving_target:
                north_distance = self.moving_target[0] - eta[0]
                east_distance  = self.moving_target[1] - eta[1]
                self.distance_to_target = math.sqrt(north_distance**2 + east_distance**2)
                dist_tot += self.distance_to_target

                if self.distance_to_target <= self.surge_setpoint:
                    north_distance = 0.0
                    east_distance  = 0.0
                    self.distance_to_target = 0.0

                self.yaw_setpoint = math.atan2(east_distance, north_distance)

                if not self.circular_target:
                    if counter % int(1 / sampleTime) == 0:
                        # your moving target logic (unchanged)...
                        pass
                else:
                    omega = 1.5 / 50
                    asd += sampleTime
                    theta = omega * asd
                    self.moving_target[0] = -20 + 40 * np.cos(theta)
                    self.moving_target[1] = -20 + 40 * np.sin(theta)

            # -----------------------------------------------------------------------
            # NMPC update at slower rate
            # -----------------------------------------------------------------------
            if i % k_per_solve == 0:
                # extract 3DOF state [x, y, psi, u, v, r]
                x3dof = np.array([eta[0], eta[1], eta[5],
                                  nu[0],  nu[1],  nu[5]], dtype=float)

                target_ref = np.array([self.target_coordinates[0],
                                       self.target_coordinates[1]], dtype=float)

                try:
                    tau_u = nmpc.solve_control(x3dof, target_ref)
                    last_tau_u = tau_u.copy()
                except Exception as e:
                    # fallback: keep previous control
                    tau_u = last_tau_u

            # use last_tau_u between MPC updates
            tau_X = float(last_tau_u[0])
            tau_N = float(last_tau_u[2])  # ignore sway component

            # -----------------------------------------------------------------------
            # same saturation and allocation logic as PID version
            # -----------------------------------------------------------------------
            self.tau_X = tau_X
            self.tau_N = tau_N

            # saturate yaw
            self.tau_N = max(min(self.tau_N, self.max_force), -self.max_force)
            remaining_force = self.max_force - abs(self.tau_N)

            # prioritize yaw, limit surge
            if self.tau_X > remaining_force:
                self.tau_X = remaining_force
            elif self.tau_X < -remaining_force:
                self.tau_X = -remaining_force

            if self.store_force_file:
                forces = np.array([self.tau_X, self.tau_N])
                self.force_array = np.vstack((self.force_array, forces))

            # thruster allocation
            if self.tau_N < 0:
                n1, n2 = otter.controlAllocation(self.tau_X, -self.tau_N)
                self.tau_N_neg = True
            else:
                n1, n2 = otter.controlAllocation(self.tau_X, self.tau_N)
                self.tau_N_neg = False

            if n1 < 0:
                self.n1neg = True
                n1 = -n1
            if n2 < 0:
                self.n2neg = True
                n2 = -n2

            torque_z, torque_x, speed = otter.otter_control.interpolate_force_values(n1, n2, 3)

            if self.n1neg:
                n1 = -n1
                self.n1neg = False
            if self.n2neg:
                n2 = -n2
                self.n2neg = False

            if self.tau_N_neg:
                n1_calc = n2
                n2_calc = n1
            else:
                n1_calc = n1
                n2_calc = n2

            u_control = np.array([n1_calc, n2_calc])

            # store data
            signals = np.append(np.append(np.append(eta, nu), u_control), u_actual)
            simData = np.vstack([simData, signals])

            # propagate dynamics
            [nu, u_actual] = self.dynamics(eta, nu, u_actual, u_control, sampleTime)
            eta = attitudeEuler(eta, nu, sampleTime)

            counter += 1

            if self.verbose:
                if counter % 100 == 0:
                    print(f"Running #{counter}")
                if self.distance_to_target < self.surge_setpoint + 2 and not finished:
                    reached_target_time = counter * sampleTime
                    finished = True

                angle = eta[5]
                if (angle > 3.12 or angle < -3.12) and not finished_yaw:
                    self.reached_yaw_target_time = counter * sampleTime
                    finished_yaw = True

            newTargetData = [self.moving_target[0], self.moving_target[1]]
            self.targetData = np.vstack([self.targetData, newTargetData])

        print(f"AVG distance to target = {dist_tot / max(1, counter)}")

        simTime = np.arange(start=0, stop=t + sampleTime, step=sampleTime)[:, None]
        targetData = self.targetData

        if self.store_force_file:
            np.savetxt("force_array.csv", self.force_array, delimiter=";", header="tau_X;tau_N", comments="")

        if self.verbose:
            print(f"Reached target in {reached_target_time}s")
            print(f"Reached yaw target in {self.reached_yaw_target_time}s")

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
        sum_tau = (
            tau
            + tau_damp
            + tau_crossflow
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

