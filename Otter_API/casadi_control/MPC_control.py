import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Find alternative method


import casadi as ca
from model_3dof import Otter3DOF 
from casadi_control.lib import MPC_config  

class NMPCControl:
    def __init__(self, f, N, sampleTime=0.02, solver=None):

        # Target states
        self.current_target = None
        self.previous_target = None
        self.target = None

        # MPC options
        self.N = N
        self.function = self.function_integrator(f, sampleTime)     # init integrator function 

        # Weights - modify to tune controller 
        self.Q_weight = ca.diag(0.05, 0, 0.05)  # state weights
        self.R_weight = ca.diag(1000, 000)      # Controller weight
        self.I_weight = ca.diag(1, 1, 1)        # Rate of change weight (control input)

        # Controls (NM forces in surge, sway, yaw)
        self.u_min = ca.DM([-116,   0, -73])                   
        self.u_max = ca.DM([ 150,   0,  73]) 

        # Solver
        self.solver = solver
       
    def function_integrator(self, f, sampleTime):

        x = ca.SX.sym('x', 6)         # [eta,nu]
        tau_u = ca.SX.sym('tau', 3)   # [X,0,N]
        #pc = ca.SX.sym('p', 3)       # currents (remove?)

        # Get 3DOF model from Otter3DOF func
        ode = f(x, tau_u) 
        #ode = f(x, u, pc) 
        
        #xdot = ode (model rhs)
        dae = {'x': x, 'p': ca.vertcat(tau_u), 'ode': ode} #force as input and current (remove currnent?)
        #dae = {'x': x, 'p': ca.vertcat(u, pc), 'ode': ode}

        # Integrator options with runge kutta 4 integrator
        integrator_options = ca.integrator('integrator',
                                    'rk',
                                    dae,
                                    {'tf': sampleTime, 'simplify': True, 'number_finite_elements': 4})

        #Function takes variables [x, tau_u)], and creates function for next integration step 
        F = ca.Function('F', [x, tau_u], [integrator_options(x0=x, p=ca.vertcat(tau_u))['xf']]) 
        return F

    def control_specification(self):
 
        N = self.N
        opti = ca.Opti()

        # control variables
        x = opti.variable(6,N+1)        # States
        tau_u = opti.variable(3,N+1)    # Controls

        #parameters
        x0 = opti.parameter(6)          # initial state for every control step 
        t_ref = opti.parameter(2)       # target reference 

        #Weights
        Q = opti.parameter(2, 2)        # tracking error weights
        R = opti.parameter(3, 3)        # MV weights, control change magnitude penalty
        I = opti.parameter(3, 3)        # rate of change penalty 
        
        
        opti.minimize(cost(x,u,target,R,Q))
        # Cost function parameters - ENDRE 
      

        # Initial conditions & bounds
        # Initial state
        opti.subject_to(x[:,0] == x0)  

        # controls
        u_min_H = ca.repmat(self.u_min, 1, N)
        u_max_H = ca.repmat(self.u_max, 1, N)
        opti.subject_to(opti.bounded(u_min_H, tau_u, u_max_H))

        # objective cost over horizon N 
        ojective_cost = 0
        for k in range(N):
            next_x = self.F(x[:,k], tau_u[:,k])
            opti.subject_to(x[:,k+1] == next_x)
        
            tracking_error = x[0:2,k] - t_ref                           # x,y - target reference error at step k
            tracking_cost = tracking_error.T @ Q @ tracking_error      # transpose works without editor color?

            control_step = tau_u[:,k]                                   # tau_u at step k
            control_cost = control_step.T @ R @ control_step 

            objective_cost += (tracking_cost + control_cost)

        # control rate objective cost between each step 
        control_rate = tau_u[:,1:] - tau_u[:,:-1]                   # difference between current and previous tau
        control_rate_cost = ca.sumsqr(I @ control_rate)            # sum of squares for every row from current steo
        objective_cost += control_rate_cost
        

        # opti.minimize(objective_cost) 
        cost_func = ca.Function('objective_cost',
                        [x, tau_u, t_ref, R, Q],
                        [objective_cost])  

        opti.minimize(cost_func(x, tau_u, t_ref, R, Q))


        
    def solver(self):
        options = {
                    "ipopt": {
                        "print_level": 2,
                        "max_iter": 500,
                        "tol": 1e-6,
                        "acceptable_tol": 1e-4,
                        "linear_solver": "mumps",
                        "warm_start_init_point": "yes"
                    },
                    "print_time": False,
                    "expand": True
                }
        opti.solver("ipopt", options)





func = Otter3DOF()
NMPC = NMPCControl(func, 100, )

