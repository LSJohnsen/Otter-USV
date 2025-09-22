
import casadi as ca
import numpy as np
from casadi_utils import *
from Otter_simulator import otter_simulator

'''
Otter USV 3-DOF CasADi model
'''

# Dict of required constant parameters from simulator object
def param_from_sim(sim):

    params = {
        'MRB6': ca.DM(sim.MRB),
        'MA6' : ca.DM(sim.MA),
        'D6'  : ca.DM(sim.D),
        'Ig'  : ca.DM(sim.Ig[:3,:3]),
        'H_rg': ca.DM(sim.H_rg),
        'm_total': float(sim.m_total),
        'L': float(sim.L), 'B': float(sim.B), 'T': float(sim.T)
    }
    return params

class Otter3DOF:
    def __init__(self, params, step):


        self.reduced = ca.DM([[1,0,0,0,0,0], # Selects 3DOF parameters from 6DOF matrix  
                        [0,1,0,0,0,0],         # reduced @ X_6DOF @ reduced.T = X_3DOF 
                        [0,0,0,0,0,1]])        # pick rows 1,2,6 @ X @ pick colums 1,2,6 


        # 6DOF mass & damping to 3DOF
        self.M3 = self.reduced @ (params['MRB6'] + params['MA6']) @ self.reduced.T 
        self.D3 = self.reduced @ params['D6'] @ self.reduced.T


        # Constants from sim
        self.Ig = params['Ig'] # inertia around CG
        self.H_rg = params['H_rg'] # CG to CO transfrom 
        self.m_total = params['m_total'] # mass
        self.MA3 = self.reduced @ params['MA6'] @ self.reduced.T # 
        self.dt = step # Integration step

        # ---
        eta = ca.SX.sym('eta', 3) # [x, y, psi]
        nu =  ca.SX.sym('nu', 3) # [u, v, r]
        tau =  ca.SX.sym('u', 3) # [X,Y,N]
        nu_c = ca.SX.sym('nu_c', 3) # Currents in body
        x = ca.vertcat(eta, nu) # 
        p    = nu_c

        J = B2N(eta[2]) #Body to NED transfrom 

        # Rigid-Body Coriolis to 3DOF 
        nu6  = ca.vertcat(nu[0],nu[1],0,0,0,nu[2])
        CRB6 = CRB6sx(nu6, self.m_total, self.Ig, self.H_rg)
        CRB3 = self.reduced @ CRB6 @ self.reduced.T 

        nu_r = nu - nu_c # Relative velocity
        
        CA3 = CA3sx(self.MA3, nu_r)
        C3 = CRB3 + CA3

        # Hydrodynamic linear damping + nonlinear yaw damping from vehicle sim
        tau_d = self.D3 @ nu_r
        tau_d[2] += 10 * self.D3[2,2] * ca.fabs(nu_r[2]) * nu_r[2]

        # cross-flow drag 
        tau_cfd = crossFlowDrag3(params['L'], params['B'], params['T'], nu_r) 



        # create state derivative for 3DOF model 
        '''
        x_dot = [eta_dot, nu_dot] 
        nu_dot using fossens equation:
        M*nu(dot)+C(nu)*nu+D(nu)nu+G(eta) -> ignoring bouyancy and solving for nu_dot
        nu_dot*M = (tau-C(nu)nu-tau_d(nu)-tau_cfd(nu)) using solver
        
        '''
        rhs = ca.vertcat(J @ nu,                    # compute the RHS forces from fossens eq               
                        ca.solve(self.M3,           
                        tau - C3 @ nu - tau_d - tau_cfd)  
                        ) 
        
        self.f_ct = ca.Function('f_ct', [x, tau, p], [rhs]) # continous time function ([state vector, control input, nu_c], rhs_function)

        #runge-kutta 4th order

        
        xk = ca.SX.sym('xk', 6)
        uk = ca.SX.sym('uk', 3)
        pk = ca.SX.sym('pk', 3)
        

        def rk4_step(x0, u0, p0):
            dt = self.dt
            k1 = self.f_ct(x0,          u0, p0)
            k2 = self.f_ct(x0 + dt/2*k1, u0, p0)
            k3 = self.f_ct(x0 + dt/2*k2, u0, p0)
            k4 = self.f_ct(x0 + dt   *k3, u0, p0)
            return x0 + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        
        self.F = ca.Function('F', [xk, uk, pk], [rk4_step(xk, uk, pk)])

sim = otter_simulator()
params = param_from_sim(sim)    
