import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Find alternative method

'''
remember to move tuning params to main
'''


import casadi as ca
from model_3dof import Otter3DOF 
from casadi_sim import otter_simulator
from casadi_control.lib.usv_params import usv_params_6dof

class NMPCControl:
    def __init__(self, f, N, sampleTime=0.02, solver=None):

        # Target and initial states
        #self.current_target = None
        #self.previous_target = None
        #self.target = None
        #self.initial_state = None
        #self.current_solver = None
        self.N = N
        self.sampleTime = sampleTime
        self.f = f
        self.F = self.function_integrator(self.f, self.sampleTime)

        # controller weights
        self.Q_weight = ca.diag(ca.DM([1.0, 1.0]))       # state weights
        self.R_weight = 0.001*ca.DM.eye(3)                   # Controller weight
        self.I_weight = 0.01*ca.diag(ca.DM([0.05, 0.05, 0.2]))    # Rate of change weight (control input)

        # Control bounds (NM forces in surge, sway, yaw), parsed to throttle map in sim
        self.u_min = ca.DM([-116,   0, -73])                   
        self.u_max = ca.DM([ 150,   0,  73]) 

        # Solver
        self.current_solver = None
        self.solver = solver
        self.control_specification()
       
    def function_integrator(self, f, sampleTime):

        x = ca.SX.sym('x', 6)         # [eta,nu]
        tau_u = ca.SX.sym('tau', 3)   # [X,0,N]
        #tau_c = ca.SX.sym('p', 3)       # currents (remove?)

        # Get 3DOF model from Otter3DOF func
        x_dot = f(x, tau_u) # gets ODE from 3dof model
        #ode = f(x, u, tau_c) 
        
        #xdot = ode (model rhs)
        dae = {'x': x, 'p': ca.vertcat(tau_u), 'ode': x_dot} #force as input and current (remove currnent?)
        #dae = {'x': x, 'p': ca.vertcat(u, tau_c), 'ode': ode}

        # Integrator options with runge kutta 4 integrator
        integrator_options = ca.integrator('integrator',
                                    'rk',
                                    dae,
                                    {'tf': sampleTime, 'simplify': True, 'number_of_finite_elements': 4})

        #Function takes variables [x, tau_u)], and creates function for next integration step 
        F = ca.Function('F', [x, tau_u], [integrator_options(x0=x, p=ca.vertcat(tau_u))['xf']]) 
        return F

    def control_specification(self):
 
        N = self.N
        opti = ca.Opti()

        # control variables
        x = opti.variable(6,N+1)            # States
        tau_u = opti.variable(3,N)          # Controls (N+1)?

        #parameters
        x0 = opti.parameter(6)          # initial state for every control step 
        t_ref = opti.parameter(2)       # target reference (x,y)

        #Weights
        Q = opti.parameter(2, 2)        # tracking error weights
        R = opti.parameter(3, 3)        # MV weights, control change magnitude penalty
        I = opti.parameter(3, 3)        # rate of change penalty 
        self.w_psi = 1

        opti.set_value(Q, self.Q_weight)
        opti.set_value(R, self.R_weight)
        opti.set_value(I, self.I_weight)
           

        # Initial conditions & bounds
        # Initial state
        opti.subject_to(x[:,0] == x0)  

        # control bounds
        u_min_H = ca.repmat(self.u_min, 1, N)
        u_max_H = ca.repmat(self.u_max, 1, N)
        opti.subject_to(opti.bounded(u_min_H, tau_u, u_max_H))

        # objective cost over horizon N
        objective_cost = 0
        for k in range(N):

            next_x = self.F(x[:,k], tau_u[:,k])
            opti.subject_to(x[:,k+1] == next_x)

            # tracking cost
            tracking_error = x[0:2,k] - t_ref                           # x,y - target reference error at step k
            tracking_cost = tracking_error.T @ Q @ tracking_error       # transpose works without editor color?


            psi = x[2, k]       # yaw
            psi_ref = ca.atan2(t_ref[1] - x[1, k],
                            t_ref[0] - x[0, k])

            dpsi = psi - psi_ref
            heading_cost = self.w_psi * (1 - ca.cos(dpsi))  

            
            # control cost
            control_step = tau_u[:,k]                                   # tau_u at step k
            control_cost = 0.1*(control_step.T @ R @ control_step) 

            objective_cost += tracking_cost + control_cost + heading_cost

        # control rate objective cost between each step 
        control_rate = tau_u[:,1:] - tau_u[:,:-1]                   # difference between current and previous tau
        control_rate_cost = ca.sumsqr(I @ control_rate)             # sum of squares for every row from current step
        objective_cost += control_rate_cost
        

        #minimize objective cost
        opti.minimize(objective_cost)

        #cost_func = ca.Function('objective_cost',
        #                [x, tau_u, t_ref, R, Q],
        #                [objective_cost])  

        #opti.minimize(cost_func(x, tau_u, t_ref, R, Q))

        #Solver
        opti.solver('ipopt', self.solver_options())

        self.opti = opti
        self.x, self.tau_u  = x, tau_u
        self.x0, self.t_ref = x0, t_ref
        self.Q, self.R, self.I = Q, R, I

    def solver_options(self):
        return {"ipopt": {                              #Ipopt optimization
            "print_level": 2,                           #Verbose
            "max_iter": 150,                          #iteration cap? 
            "tol": 1e-4,                                #stopping tolerance prev 1e6
            "acceptable_tol": 1e-3,                     #early stop if adequate prev 1e4
            "acceptable_iter": 5,                       #terminate early if enough acceptable iterations
            "linear_solver": "mumps",                   #linear solver?
            "warm_start_init_point": "yes",             #Warm start -> use previous memory
            "warm_start_bound_push": 1e-3,
            "warm_start_mult_bound_push": 1e-3,         # "hessian_approximation": "limited-memory",  #limited memory LBFGS
            "print_timing_statistics": "no"              
        },
        "print_time": False,
        "expand": True}
    
    def solve_control(self, init_state, target_reference):
        """
        init_state: np.array shape (6,)  -> [x, y, psi, u, v, r]
        target_reference: np.array shape (2,) -> [x_ref, y_ref]
        """

        self.opti.set_value(self.x0, init_state)            # x0 .- USV initial state
        self.opti.set_value(self.t_ref, target_reference)   # t_ref - target reference point


        if self.current_solver is not None:
            self.opti.set_initial(self.x, self.current_solver.value(self.x))
            self.opti.set_initial(self.tau_u, self.current_solver.value(self.tau_u))

        else:
            self.opti.set_initial(self.x, 0)
            self.opti.set_initial(self.tau_u, 0)

        solve_control = self.opti.solve()
        self.current_solver = solve_control  

        return solve_control.value(self.tau_u)[:,0].flatten()
    




