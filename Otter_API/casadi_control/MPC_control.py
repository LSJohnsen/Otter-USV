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


import casadi as ca
import numpy as np


class NMPCControl:
    def __init__(self, f, N, sampleTime=0.02, solver=None, mode="tracking"):
        self.N = N
        self.sampleTime = sampleTime
        self.f = f
        self.F = self.function_integrator(self.f, self.sampleTime)

        # Control bounds [X, Y, N], with Y fixed to 0 by bounds
        self.u_min = ca.DM([-116,   0, -73])
        self.u_max = ca.DM([ 150,   0,  73])

        self.current_solver = None
        self.solver = solver

        # Controller weights
        self.mode = None
        self.Q_weight = None
        self.R_weight = None
        self.I_weight = None
        self.w_psi = None
        self.d_hold = None
        self.Q_vel = None
        self.Qf_pos = None
        self.Qf_vel = None

        self.set_mode(mode)
        self.control_specification()

    def set_mode(self, mode: str):
        """
        mode:
            "tracking"       -> simpler NMPC for pure pursuit target tracking
            "stationkeeping" -> blended NMPC for fixed-target holding
        """
        mode = mode.lower()

        if mode == "tracking":
            self.mode = "tracking"

            # tracking-oriented tuning
            self.Q_weight = ca.diag(ca.DM([1.0, 1.0]))                  # error
            self.R_weight = ca.diag(ca.DM([0.001, 0.001, 0.001]))       # actuator magnitude          
            self.I_weight = 0.01 * ca.diag(ca.DM([0.05, 0.05, 5]))      # actuator rate of change 
            # self.I_weight = ca.diag(ca.DM([0.05, 0, 5]))              # slightly slower convergence but much smoother actions

            self.w_psi = 2.0

            # Not currently used in tracking mode
            self.d_hold = 1.0
            self.Q_vel = ca.diag(ca.DM([1.0, 2.0, 0.5]))
            self.Qf_pos = ca.diag(ca.DM([5.0, 5.0]))
            self.Qf_vel = ca.diag(ca.DM([2.0, 3.0, 1.0]))

        elif mode == "stationkeeping":
            self.mode = "stationkeeping"

            # More stabilization-oriented tuning
            self.Q_weight = ca.diag(ca.DM([1.0, 1.0]))
            self.R_weight = 0.001 * ca.DM.eye(3)
            self.I_weight = 0.01 * ca.diag(ca.DM([0.05, 0.05, 0.01]))

            self.w_psi = 5.0
            self.d_hold = 3.0

            self.Q_vel = ca.diag(ca.DM([3.0, 6.0, 2.0]))
            self.Qf_pos = ca.diag(ca.DM([100.0, 100.0]))
            self.Qf_vel = ca.diag(ca.DM([8.0, 10.0, 4.0]))

        else:
            raise ValueError(f"Unknown NMPC mode: {mode}")

    def function_integrator(self, f, sampleTime):
        x = ca.SX.sym('x', 6)         # [x, y, psi, u, v, r]
        tau_u = ca.SX.sym('tau', 3)   # [X, 0, N]

        x_dot = f(x, tau_u)

        dae = {
            'x': x,
            'p': ca.vertcat(tau_u),
            'ode': x_dot
        }

        integrator_options = ca.integrator(
            'integrator',
            'rk',
            dae,
            {'tf': sampleTime, 'simplify': True, 'number_of_finite_elements': 4}
        )

        F = ca.Function(
            'F',
            [x, tau_u],
            [integrator_options(x0=x, p=ca.vertcat(tau_u))['xf']]
        )
        return F

    def control_specification(self):
        N = self.N
        opti = ca.Opti()

        # decision variables
        x = opti.variable(6, N + 1)      # [x, y, psi, u, v, r]
        tau_u = opti.variable(3, N)      # [X, Y, N]

        # parameters
        x0 = opti.parameter(6)
        t_ref = opti.parameter(2)

        # tunable weights as Opti parameters
        Q = opti.parameter(2, 2)
        R = opti.parameter(3, 3)
        I = opti.parameter(3, 3)

        opti.set_value(Q, self.Q_weight)
        opti.set_value(R, self.R_weight)
        opti.set_value(I, self.I_weight)

        # initial condition
        opti.subject_to(x[:, 0] == x0)

        # input bounds
        u_min_H = ca.repmat(self.u_min, 1, N)
        u_max_H = ca.repmat(self.u_max, 1, N)
        opti.subject_to(opti.bounded(u_min_H, tau_u, u_max_H))

        objective_cost = 0

        for k in range(N):
            # dynamics
            next_x = self.F(x[:, k], tau_u[:, k])
            opti.subject_to(x[:, k + 1] == next_x)

            # position tracking
            pos_error = x[0:2, k] - t_ref
            tracking_cost = pos_error.T @ Q @ pos_error

            # heading-to-target cost
            psi = x[2, k]
            psi_ref = ca.atan2(
                t_ref[1] - x[1, k],
                t_ref[0] - x[0, k]
            )
            dpsi = psi - psi_ref

            # control effort
            control_step = tau_u[:, k]
            control_cost = 0.1 * (control_step.T @ R @ control_step)

            if self.mode == "tracking":
                # Simpler original target-tracking cost
                heading_cost = self.w_psi * (1 - ca.cos(dpsi))

                objective_cost += tracking_cost + control_cost + heading_cost

            elif self.mode == "stationkeeping":
                # Blended station-keeping formulation
                dist = ca.norm_2(pos_error)
                sigma = dist**2 / (dist**2 + self.d_hold**2)

                # keep some heading guidance near target, but reduced
                heading_scale = 0.2 + 0.8 * sigma
                heading_cost = heading_scale * self.w_psi * (1 - ca.cos(dpsi))

                # stronger damping near target
                vel = x[3:6, k]   # [u, v, r]
                vel_cost = (1 - sigma) * (vel.T @ self.Q_vel @ vel)

                objective_cost += tracking_cost + control_cost + heading_cost + vel_cost

            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        # control rate penalty
        control_rate = tau_u[:, 1:] - tau_u[:, :-1]
        control_rate_cost = ca.sumsqr(I @ control_rate)
        objective_cost += control_rate_cost

        # Terminal penalties only for stationkeeping
        if self.mode == "stationkeeping":
            terminal_pos_error = x[0:2, N] - t_ref
            terminal_pos_cost = terminal_pos_error.T @ self.Qf_pos @ terminal_pos_error

            terminal_vel = x[3:6, N]
            terminal_vel_cost = terminal_vel.T @ self.Qf_vel @ terminal_vel

            objective_cost += terminal_pos_cost + terminal_vel_cost

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
                "print_level": 2,
                "max_iter": 150,
                "tol": 1e-4,
                "acceptable_tol": 1e-3,
                "acceptable_iter": 5,
                "linear_solver": "mumps",
                "warm_start_init_point": "yes",
                "warm_start_bound_push": 1e-3,
                "warm_start_mult_bound_push": 1e-3,
                "print_timing_statistics": "no",
            },
            "print_time": False,
            "expand": True
        }

    def solve_control(self, init_state, target_reference):
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
            self.opti.set_initial(self.x, self.current_solver.value(self.x))
            self.opti.set_initial(self.tau_u, self.current_solver.value(self.tau_u))
        else:
            self.opti.set_initial(self.x, 0)
            self.opti.set_initial(self.tau_u, 0)

        solve_control = self.opti.solve()
        self.current_solver = solve_control

        return solve_control.value(self.tau_u)[:, 0].flatten()