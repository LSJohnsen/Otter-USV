import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import casadi as ca
import numpy as np
from casadi_control.lib.casadi_utils import B2N, CRB6sx, CA3sx, crossFlowDrag3
from casadi_control.lib.usv_params import usv_params_6dof

''' 
Otter USV 3-DOF CasADi model
'''

params = usv_params_6dof()

class Otter3DOF:
    def __init__(self, params, step):


        self.reduced = ca.DM([[1,0,0,0,0,0], # Selects 3DOF parameters from 6DOF matrix  
                        [0,1,0,0,0,0],         # reduced @ X_6DOF @ reduced.T = X_3DOF 
                        [0,0,0,0,0,1]])        # pick rows 1,2,6 @ X @ pick colums 1,2,6 


        # 6DOF mass & damping to 3DOF
        self.M3 = self.reduced @ (params['MRB6'] + params['MA6']) @ self.reduced.T ### Change matrix operations to ca.mtimes(m1, m2, m3) if i mix dm/sx etc.
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

        # Hydrodynamic linear damping + nonlinear yaw damping (from simulator)
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

        ode = ca.vertcat(J @ nu,                    # compute the RHS forces from fossens eq (ODE for 3DOF USV)              
                        ca.solve(self.M3,           
                        tau - C3 @ nu - tau_d - tau_cfd)  
                        ) 
        
        # DAE
        dae = {
        'x': x,                         # states (eta, nu)
        'p': ca.vertcat(tau, p),        # parameters (inputs + currents) CHANGE TO ONLY CONTROLS OR ADD FILTER/ESTIMATION
        'ode': ode                      # dynamics (x, u, p)
        }    
        
        # Integrator
        integrator = ca.integrator(
        'integrator', 'rk', dae,
        {'tf': self.dt, 'simplify': True, 'number_of_finite_elements': 4}
        )
     
        # Function F(xk,uk,pk)->F(k+1)
        self.F = ca.Function(
        'F',
        [x, tau, p],
        [integrator(x0=x, p=ca.vertcat(tau, p))['xf']]
        ) # continous time function ([state vector, control input, nu_c], rhs_function)

        


test = Otter3DOF(params, 1)

# Testing, x = [eta,nu] = [position, velocities]
x0   = ca.DM.zeros(6, 1)      # [x, y, psi, u, v, r]

tau  = ca.DM([100, 0, 50])    # [X, Y, N]
nu_c = ca.DM.zeros(3, 1)      # currents in body frame

x1 = test.F(x0, tau, nu_c)    # one integration step
x2 = x1 + test.F(x0, tau, nu_c)
x3 = x2 + test.F(x0, tau, nu_c)
