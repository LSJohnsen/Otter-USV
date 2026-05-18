import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

'''
remember to move tuning params to main
'''


import casadi as ca
from model_3dof import Otter3DOF 
from casadi_sim import otter_simulator
from casadi_control.lib.usv_params import usv_params_6dof

import casadi as ca
import numpy as np


# to improve computation time
def smooth_relu(z, eps=1e-3):
    return 0.5 * (z + ca.sqrt(z**2 + eps))


class NMPCControl:
    def __init__(self, f, N, sampleTime=0.02, solver=None, mode="tracking"):
        self.N = N
        self.sampleTime = sampleTime
        self.f = f
        self.F = self.function_integrator(self.f, self.sampleTime)

        # Control bounds [X, Y, N], with Y fixed to 0 by bounds INCREASE 
        self.u_min = ca.DM([-116, 0, -75])
        self.u_max = ca.DM([150, 0, 75])

        self.current_solver = None
        self.solver = solver

        # Controller mode
        self.mode = None

        # Position tracking weight
        # Higher = follows target harder, lower = smoother/slower motion
        self.Q_weight = None

        # Control effort weight
        # Higher = less thrust/yaw moment, lower = more aggressive control
        self.R_weight = None

        # Control rate weight
        # Higher = smoother input changes, lower = faster but more jittery
        self.I_weight = None

        # Heading alignment weight
        # Higher = prioritizes desired heading, lower = allows more drift
        self.w_psi = None

        # Close-distance threshold for damping/precision behavior
        self.d_hold = None

        # Wider threshold for approach behavior
        self.d_approach = None

        # Velocity damping weight [u, v, r]
        # Higher = stronger damping of surge/sway/yaw rate
        self.Q_vel = None

        # Terminal position weight
        # Higher = prioritizes final position accuracy
        self.Qf_pos = None

        # Terminal velocity weight
        # Higher = encourages stopping at end of horizon
        self.Qf_vel = None

        # Extra position precision near target
        self.Q_precision = None

        # Heading blending distances for straight-line tracking
        # Far: LOS heading, close: path heading
        self.d_heading_close = 3.0
        self.d_heading_far = 8.0

        self.set_mode(mode)
        self.control_specification()

    def set_mode(self, mode: str):
        """
        mode:
            "tracking"       -> moving target / straight-line tracking
            "stationkeeping" -> fixed-target holding
        """
        mode = mode.lower()
        
        if mode == "tracking":
            self.mode = "tracking"

            # No reverse surge thrust in tracking mode
            self.u_min = ca.DM([0.0, 0.0, -73.0])
            self.u_max = ca.DM([150.0, 0.0, 73.0])

            # Position tracking
            self.Q_weight = ca.diag(ca.DM([25.0, 25.0]))

            # Control effort [X, Y, N]
            self.R_weight = ca.diag(ca.DM([0.040, 0.001, 0.004]))

            # Control-rate smoothing [dX, dY, dN]
            self.I_weight = ca.diag(ca.DM([0.001, 0.001, 0.25]))

            # heading support
            self.w_psi = 3.0

            # weight for matching velocity of target
            self.matching_weight = 5.0

            #squared cost penalty for heading error
            self.heading_misalignment_cost = 5.0 
            

            # not used but initialized outside tracking
            self.d_heading_close = 3.0
            self.d_heading_far = 10.0
            self.d_approach = 1.0
            self.d_hold = 4.0

            self.Q_vel = ca.diag(ca.DM([0.0, 0.0, 0.0]))
            self.Qf_vel = ca.diag(ca.DM([0.0, 0.0, 0.0]))
            self.Q_precision = ca.diag(ca.DM([0.0, 0.0]))
    
        
        # if mode == "tracking":
        #     self.mode = "tracking"

        #     # No reverse surge thrust
        #     self.u_min = ca.DM([0.0, 0.0, -73.0])
        #     self.u_max = ca.DM([150.0, 0.0, 73.0])

        #     # Position tracking [north, east]
        #     self.Q_weight = ca.diag(ca.DM([20.0, 20.0]))

        #     # Control effort [X, Y, N]
        #     self.R_weight = ca.diag(ca.DM([0.015, 0.001, 0.001]))

        #     # Control-rate smoothing [dX, dY, dN]
        #     self.I_weight = ca.diag(ca.DM([0.003, 0.001, 0.05]))

        #     # Heading alignment
        #     self.w_psi = 5.0

        #     # Body velocity tracking [u, v, r]
        #     self.Q_vel = ca.diag(ca.DM([10.0, 1.0, 1.0]))

        #     # Not used, but keep initialized
        #     self.matching_weight = 0.0
        #     self.heading_misalignment_cost = 0.0

        #     self.d_heading_close = 3.0
        #     self.d_heading_far = 10.0
        #     self.d_approach = 1.0
        #     self.d_hold = 4.0

        #     self.Qf_vel = ca.diag(ca.DM([5.0, 1.0, 1.0]))
        #     self.Q_precision = ca.diag(ca.DM([0.0, 0.0]))
        
                

        elif mode == "stationkeeping":
            self.mode = "stationkeeping"

            self.Q_weight = ca.diag(ca.DM([1.0, 1.0]))
            self.R_weight = ca.diag(ca.DM([0.0075, 0.001, 0.0075]))  
            
            '''
            self.R_weight = ca.diag(ca.DM([0.5, 0.001, 0.1]))  # brukbar
            self.R_weight = ca.diag(ca.DM([0.2, 0.001, 0.05]))  #bedre
            self.R_weight = ca.diag(ca.DM([0.15, 0.001, 0.05]))  #beste <2m spin
            
            '''
            self.I_weight = 0.5*ca.diag(ca.DM([0.1, 0.05, 0.1]))

            ''' circling:
            self.Q_weight = ca.diag(ca.DM([1.0, 1.0]))
            self.R_weight = ca.diag(ca.DM([0.5, 0.001, 0.5]))
            self.I_weight = ca.diag(ca.DM([0.1, 0.05, 0.1]))
            '''

            self.w_psi = 0.0

            self.d_approach = 6.0
            self.d_hold = 5.0

            self.d_heading_close = 0.5
            self.d_heading_far = 6.0

            self.Q_vel = ca.diag(ca.DM([5.0, 4.0, 8.0]))
            self.Qf_vel = ca.diag(ca.DM([6.0, 6.0, 3.0]))
            self.Q_precision = ca.diag(ca.DM([10.0, 10.0]))

        else:
            raise ValueError(f"Unknown NMPC mode: {mode}")

    def function_integrator(self, f, sampleTime):
        x = ca.SX.sym("x", 6)          # [x, y, psi, u, v, r]
        tau_u = ca.SX.sym("tau", 3)    # [X, 0, N]

        x_dot = f(x, tau_u)

        dae = {
            "x": x,
            "p": ca.vertcat(tau_u),
            "ode": x_dot,
        }

        integrator_options = ca.integrator(
            "integrator",
            "rk",
            dae,
            {
                "tf": sampleTime,
                "simplify": True,
                "number_of_finite_elements": 4,
            },
        )

        F = ca.Function(
            "F",
            [x, tau_u],
            [integrator_options(x0=x, p=ca.vertcat(tau_u))["xf"]],
        )

        return F

    def control_specification(self):
        N = self.N
        opti = ca.Opti()

        # Decision variables
        x = opti.variable(6, N + 1)      # [x, y, psi, u, v, r]
        tau_u = opti.variable(3, N)      # [X, Y, N]

        # Parameters
        x0 = opti.parameter(6)

        # Target reference:
        #   t_ref[0] = target north
        #   t_ref[1] = target east
        #   t_ref[2] = path heading
        #   t_ref[3] = target north velocity
        #   t_ref[4] = target east velocity
        t_ref = opti.parameter(5)

        # Tunable weights as Opti parameters
        Q = opti.parameter(2, 2)
        R = opti.parameter(3, 3)
        I = opti.parameter(3, 3)

        opti.set_value(Q, self.Q_weight)
        opti.set_value(R, self.R_weight)
        opti.set_value(I, self.I_weight)

        # Initial condition
        opti.subject_to(x[:, 0] == x0)

        # Input bounds
        u_min_H = ca.repmat(self.u_min, 1, N)
        u_max_H = ca.repmat(self.u_max, 1, N)
        opti.subject_to(opti.bounded(u_min_H, tau_u, u_max_H))

        # Coupled thruster constraints
        # X = T_L + T_R
        # N = l * (T_L - T_R)
        l_thruster = 1.08 / 2.0

        T_min = -116.0 / 2.0
        T_max = 150.0 / 2.0

        X_force = tau_u[0, :]
        N_moment = tau_u[2, :]

        T_L = 0.5 * (X_force + N_moment / l_thruster)
        T_R = 0.5 * (X_force - N_moment / l_thruster)

        opti.subject_to(opti.bounded(T_min, T_L, T_max))
        opti.subject_to(opti.bounded(T_min, T_R, T_max))

        objective_cost = 0

        for k in range(N):
            next_x = self.F(x[:, k], tau_u[:, k])
            opti.subject_to(x[:, k + 1] == next_x)

            target_v = t_ref[3:5]
            target_pos = t_ref[0:2] + k * self.sampleTime * target_v

            pos_error = x[0:2, k] - target_pos

            dist = ca.norm_2(pos_error)
            eps = 1e-6

            sigma_approach = dist**2 / (dist**2 + self.d_approach**2)
            sigma_hold = dist**2 / (dist**2 + self.d_hold**2)
            near_target = 1.0 - sigma_hold

            tracking_cost = pos_error.T @ Q @ pos_error
            precision_cost = near_target * (
                pos_error.T @ self.Q_precision @ pos_error
            )

            psi = x[2, k]

            dx = target_pos[0] - x[0, k]
            dy = target_pos[1] - x[1, k]

            psi_los = ca.atan2(dy, dx)
            psi_path = t_ref[2]

            # alpha = 1 far from target -> LOS heading
            # alpha = 0 close to target -> path heading
            alpha_raw = (dist - self.d_heading_close) / (
                self.d_heading_far - self.d_heading_close + eps
            )
            alpha = ca.fmax(0.0, ca.fmin(1.0, alpha_raw))

            los_x = ca.cos(psi_los)
            los_y = ca.sin(psi_los)

            path_x = ca.cos(psi_path)
            path_y = ca.sin(psi_path)

            ref_x = alpha * los_x + (1.0 - alpha) * path_x
            ref_y = alpha * los_y + (1.0 - alpha) * path_y

            psi_ref = ca.atan2(ref_y, ref_x)

            dpsi_raw = psi - psi_ref
            dpsi = ca.atan2(ca.sin(dpsi_raw), ca.cos(dpsi_raw))

            control_cost = 0.1 * (tau_u[:, k].T @ R @ tau_u[:, k])
            
            if self.mode == "tracking":
                u_body = x[3, k]
                v_body = x[4, k]
                psi = x[2, k]

                # Target path direction
                path_heading = t_ref[2]

                path_tangent = ca.vertcat(
                    ca.cos(path_heading),
                    ca.sin(path_heading)
                )

                path_normal = ca.vertcat(
                    -ca.sin(path_heading),
                    ca.cos(path_heading)
                )

                # Distance-based transition
                far_target = 1.0 - near_target

                # Vessel velocity in inertial frame
                vessel_v_north = ca.cos(psi) * u_body - ca.sin(psi) * v_body
                vessel_v_east  = ca.sin(psi) * u_body + ca.cos(psi) * v_body

                vessel_vel = ca.vertcat(
                    vessel_v_north,
                    vessel_v_east
                )

                # Target velocity in inertial frame
                target_vel = ca.vertcat(
                    t_ref[3],
                    t_ref[4]
                )

                # Velocity error decomposed along and across the target path
                vel_error = vessel_vel - target_vel

                along_vel_error = path_tangent.T @ vel_error
                cross_vel_error = path_normal.T @ vel_error

                # Stronger along-path velocity matching close to target
                along_vel_weight = self.matching_weight * (1.0 + 4.0 * near_target)

                # Weaker cross-velocity matching close to target
                cross_vel_weight = 0.2 * self.matching_weight * far_target

                vel_match_cost = (
                    along_vel_weight * along_vel_error**2
                    + cross_vel_weight * cross_vel_error**2
                )

                # Cross-track position error relative to target path
                path_error = x[0:2, k] - target_pos
                cross_track_error = path_normal.T @ path_error

                cross_track_cost = self.heading_misalignment_cost * cross_track_error**2

                # LOS heading
                dx = target_pos[0] - x[0, k]
                dy = target_pos[1] - x[1, k]

                los_heading = ca.atan2(dy, dx)

                los_heading_error = ca.atan2(
                    ca.sin(psi - los_heading),
                    ca.cos(psi - los_heading)
                )

                # Path heading alignment
                path_heading_error = ca.atan2(
                    ca.sin(psi - path_heading),
                    ca.cos(psi - path_heading)
                )

                # Heading should help approach, not dominate close tracking
                los_heading_cost = far_target * self.w_psi * los_heading_error**2

                path_heading_cost = (
                    self.w_psi
                    * (0.5 * far_target + 0.25 * near_target)
                    * path_heading_error**2
                )

                heading_align_cost = los_heading_cost + path_heading_cost

                objective_cost += (
                    tracking_cost
                    + control_cost
                    + vel_match_cost
                    + cross_track_cost
                    + heading_align_cost
                )
                
            
            # if self.mode == "tracking":
            #     # Vessel states
       
            #     eta_pos = x[0:2, k]
            #     psi = x[2, k]

            #     nu = x[3:6, k]  # [u, v, r]

            #     # Predicted target/reference along horizon
            #     target_v = t_ref[3:5]
            #     target_pos = t_ref[0:2] + k * self.sampleTime * target_v

            #     path_heading = t_ref[2]

              
            #     # Reference surge speed from target inertial velocity
              
            #     target_speed = ca.norm_2(target_v)

            #     nu_ref = ca.vertcat(
            #         target_speed,  # desired surge speed
            #         0.0,           # desired sway velocity
            #         0.0            # desired yaw rate
            #     )

            #     # Position tracking cost
              
            #     pos_error = eta_pos - target_pos

            #     tracking_cost = pos_error.T @ Q @ pos_error

            #     # Heading tracking cost
               
            #     heading_error = ca.atan2(
            #         ca.sin(psi - path_heading),
            #         ca.cos(psi - path_heading)
            #     )

            #     heading_cost = self.w_psi * heading_error**2

            #     # Body velocity tracking cost
            #     vel_error = nu - nu_ref

            #     vel_cost = vel_error.T @ self.Q_vel @ vel_error

            #     # Control effort cost
            #     control_cost = 0.1 * (tau_u[:, k].T @ R @ tau_u[:, k])

            #     # Control rate smoothing cost
            #     if k == 0:
            #         delta_tau = tau_u[:, k]
            #     else:
            #         delta_tau = tau_u[:, k] - tau_u[:, k - 1]

            #     input_rate_cost = delta_tau.T @ I @ delta_tau


            #     # Total standard tracking objective
            #     objective_cost += (
            #         tracking_cost
            #         + heading_cost
            #         + vel_cost
            #         + control_cost
            #         + input_rate_cost
            #     )
            
            elif self.mode == "stationkeeping":

                # Heading should matter during approach, but not dominate close to target
                heading_cost = sigma_approach**2 * self.w_psi * (1 - ca.cos(dpsi))

                # General velocity damping close to target
                vel = x[3:6, k]
                vel_cost = near_target * (vel.T @ self.Q_vel @ vel)

                # Reduce forward surge when heading error is large, mainly far from target
                heading_alignment = (1 + ca.cos(dpsi)) / 2
                X_force_k = tau_u[0, k]
                surge_misalignment_cost = 0.03 * sigma_approach * (1 - heading_alignment) * X_force_k**2

                # Convert body-fixed velocity to inertial velocity
                u_body = x[3, k]
                v_body = x[4, k]

                vel_n = ca.cos(psi) * u_body - ca.sin(psi) * v_body
                vel_e = ca.sin(psi) * u_body + ca.cos(psi) * v_body

                # vector from vessel to target
                eps = 1e-6
                e_to_target_n = (t_ref[0] - x[0, k]) / (dist + eps)
                e_to_target_e = (t_ref[1] - x[1, k]) / (dist + eps)


                # Positive radial velocity means moving toward the target
                v_radial = vel_n * e_to_target_n + vel_e * e_to_target_e

                v_tang_n     = vel_n - v_radial * e_to_target_n
                v_tang_e     = vel_e - v_radial * e_to_target_e
                v_tang_sq    = v_tang_n**2 + v_tang_e**2

                tangential_damping_cost = 25.0 * near_target * v_tang_sq

                # Damp radial motion close to target, regardless of direction
                radial_damping_cost    = 15.0 * near_target * v_radial**2


                objective_cost += (
                    tracking_cost
                    + control_cost
                    + heading_cost
                    + vel_cost
                    + surge_misalignment_cost
                    + radial_damping_cost
                    + tangential_damping_cost
                    + precision_cost
                )

            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        # control rate penalty
        control_rate = tau_u[:, 1:] - tau_u[:, :-1]
        control_rate_cost = ca.sumsqr(I @ control_rate)
        objective_cost += control_rate_cost

        # Terminal penalties for stationkeeping
        if self.mode == "stationkeeping":

            terminal_vel = x[3:6, N]
            terminal_vel_cost = terminal_vel.T @ self.Qf_vel @ terminal_vel

            objective_cost += terminal_vel_cost

        opti.minimize(objective_cost)
        opti.solver('ipopt', self.solver_options())

        self.opti = opti
        self.x = x
        self.tau_u = tau_u
        self.x0 = x0
        self.t_ref = t_ref
        self.Q = Q
        self.R = R
        self.I = I

    def solver_options(self):
        return {
            "ipopt": {
                "print_level": 0,
                "max_iter": 100,

                # Do not require very tight convergence every real-time step
                "tol": 1e-3,
                "acceptable_tol": 5e-3,
                "acceptable_iter": 3,

                # Warm-start options
                "warm_start_init_point": "yes",
                "warm_start_bound_push": 1e-4,
                "warm_start_mult_bound_push": 1e-4,
                "warm_start_slack_bound_push": 1e-4,

                # Solver
                "linear_solver": "mumps",
                "print_timing_statistics": "no",
            },
            "print_time": False,
            "expand": True,
        }

    def solve_control(self, init_state, target_reference, psi_ref=None):
        """
        init_state: np.array shape (6,) -> [x, y, psi, u, v, r]
        target_reference: np.array shape (2,) -> [x_ref, y_ref]
        """
        self.opti.set_value(self.x0, init_state)
        self.opti.set_value(self.t_ref, target_reference)


        # refresh prams
        self.opti.set_value(self.Q, self.Q_weight)
        self.opti.set_value(self.R, self.R_weight)
        self.opti.set_value(self.I, self.I_weight)


        if self.current_solver is not None:
            x_prev = self.current_solver.value(self.x)
            u_prev = self.current_solver.value(self.tau_u)

            # Shift state prediction forward
            x_init = np.hstack([x_prev[:, 1:], x_prev[:, -1:]])

            # Shift control sequence forward
            u_init = np.hstack([u_prev[:, 1:], u_prev[:, -1:]])

            self.opti.set_initial(self.x, x_init)
            self.opti.set_initial(self.tau_u, u_init)
        else:
            self.opti.set_initial(self.x, 0)
            self.opti.set_initial(self.tau_u, 0)

        solve_control = self.opti.solve()
        self.current_solver = solve_control

        return solve_control.value(self.tau_u)[:, 0].flatten()