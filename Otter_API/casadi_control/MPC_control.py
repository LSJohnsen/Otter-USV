

import casadi as ca
from model_3dof import Otter3DOF 

class NMPCControl:
    def __init__(self, usv_model, N, sampleTime=0.02, solver="rk"):


        # USV states
        self.usv_model = usv_model
        self.x = self.usv_model.x
        self.u = self.usv_model.u
        self.ode = self.usv_model.ode

        # Target states
        self.current_target = None
        self.previous_target = None
        self.target = None

        # MPC options
        self.N = N





        
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

    
    def control_step(f, sampleTime):

        x  = ca.SX.sym('x', 6)
        u = ca.SX.sym('u', 3)
        p  = ca.SX.sym('nu_c', 3)

        # DAE
        dae = {
        'x': x,                       # states (eta, nu)
        'p': ca.vertcat(u, p),        # parameters (inputs + currents) CHANGE TO ONLY CONTROLS OR ADD FILTER/ESTIMATION
        'ode': f(x, u, p)            # dynamics (x, u, p) -> f inherits the ODE: see 3DOF_model.py
        }    
        
        # Integrator
        integrator = ca.integrator(
        'integrator', 'rk', dae,
        {'tf': sampleTime, 'simplify': True, 'number_of_finite_elements': 4}
        )
        
        F = ca.Function('F_dt', [x, u, p], [integrator(x0=x, p=ca.vertcat(u, p))['xf']])
        return F