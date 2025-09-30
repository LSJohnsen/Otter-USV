

import casadi as ca
from model_3dof import Otter3DOF 

class NMPCControl:
    def __init__(self, f, N, sampleTime=0.02, solver="rk"):

        # Target states
        self.current_target = None
        self.previous_target = None
        self.target = None

        # MPC options
        self.N = N
        self.function = self.function_integrator(f, sampleTime) # make integration 


    def function_integrator(self, f, sampleTime):

        x = ca.SX.sym('x', 6) # [eta,nu]
        u = ca.SX.sym('tau', 3) # [X,0,N]
        pc = ca.SX.sym('p', 3) # currents (remove?)

        # Get 3DOF model from Otter3DOF func
        ode = f(x, u, p) 
        
        #xdot = ode (model rhs)
        dae = {'x': x, 'p': ca.vertcat(u, p), 'ode': ode} #force as input and current (remove currnent?)
        
        # Integrator options with runge kutta 4 integrator
        integrator_options = ca.integrator('integrator',
                                    'rk',
                                    dae,
                                    {'tf': sampleTime, 'simplify':True, 'number_finite_elements': 4})

        #Function takes variables [x, u, pc (remove pc)], and creates function for next integration step 
        F = ca.Function('F', [x, u, pc], [integrator_options(x0=x, p=ca.vertcat(u, pc))['xf']]) 
        return F



        
        self.N = N
        opti = ca.Opti()

        x = opti.variable(6,N+1) # States
        u = opti.variable(3,N+1) # Controls
        p = opti.parameter(6) 
        target = opti.parameter(2) # target state




        # Cost function parameters - ENDRE 
        Q = opti.parameter(2, 2) # Control variable weights (output (CV, y))
        R = opti.parameter(3, 3) # Manipulated variable weights (Input (MU, u)
        I = opti.parameter(3, 3) 




function = Otter3DOF()
NMPC = NMPCControl(function, 100, )

