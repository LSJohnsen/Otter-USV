import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Find alternative method


import casadi as ca
from model_3dof import Otter3DOF 
from casadi_control.lib import MPC_config

class NMPCControl:
    def __init__(self, f, N, sampleTime=0.02, solver="rk"):

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


       
    def function_integrator(self, f, sampleTime):

        x = ca.SX.sym('x', 6)       # [eta,nu]
        tau_u = ca.SX.sym('tau', 3)   # [X,0,N]
        #pc = ca.SX.sym('p', 3)     # currents (remove?)

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
                                    {'tf': sampleTime, 'simplify':True, 'number_finite_elements': 4})

        #Function takes variables [x, tau_u)], and creates function for next integration step 
        F = ca.Function('F', [x, tau_u], [integrator_options(x0=x, p=ca.vertcat(tau_u))['xf']]) 
        return F

    def control_specification(self):
 
        N = self.N

        opti = ca.Opti()

        x = opti.variable(6,N+1)        # States
        tau_u = opti.variable(3,N+1)    # Controls
        x0 = opti.parameter(6)          # initial state for every control step 
        t_ref = opti.parameter(2)       # target
        
        

        # Cost function parameters - ENDRE 
        Q = opti.parameter(2, 2) # tracking error weights
        R = opti.parameter(3, 3) # MV weights, control change magnitude penalty
        I = opti.parameter(3, 3) # rate of change penalty 

        # Initial conditions & bounds 

        # Controls (max 200NM at the same time - surge+yaw must be between +200, -~)
        u_min = ca.DM([-116,   0, -73])  # Change surge/yaw to correct force specifications
        u_max = ca.DM([ 150,   0,  73]) 
        u_min_H = ca.repmat(u_min, 1, N)
        u_max_H = ca.repmat(u_max, 1, N)
        opti.subject_to(opti.bounded(u_min_H, tau_u, u_max_H)) #[+-150, 0, +-50]


        # Weights
        opti.set_value(Q, self.Q_weight) 
        opti.set_value(R, self.R_weight)
        opti.set_value(I, self.I_weight)

    def solver(self):





func = Otter3DOF()
NMPC = NMPCControl(func, 100, )

