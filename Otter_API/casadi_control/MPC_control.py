

import casadi as ca


class NMPCControl:
    def __init__(self, model, N):


        
        self.N = N
        opti = ca.Opti()

        x = opti.variable(6,N+1)
        u = opti.vaiable(3,N+1) 

    
    def control_step(f, sampleTime):

        x  = ca.SX.sym('x', 6)
        u = ca.SX.sym('u', 3)
        p  = ca.SX.sym('nu_c', 3)

        # DAE
        dae = {
        'x': x,                       # states (eta, nu)
        'p': ca.vertcat(u, p),        # parameters (inputs + currents) CHANGE TO ONLY CONTROLS OR ADD FILTER/ESTIMATION
        'ode': f(x, u, p)            # dynamics (x, u, p): see 3DOF_model.py
        }    
        
        # Integrator
        integrator = ca.integrator(
        'integrator', 'rk', dae,
        {'tf': sampleTime, 'simplify': True, 'number_of_finite_elements': 4}
        )
        
        F = ca.Function('F_dt', [x, u, p], [integrator(x0=x, p=ca.vertcat(u, p))['xf']])
        return F