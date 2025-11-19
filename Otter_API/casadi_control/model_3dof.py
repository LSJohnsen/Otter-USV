import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Find alternative method

import casadi as ca
from casadi_control.lib.casadi_utils import B2N, CRB6sx, CA3sx, crossFlowDrag3
from casadi_control.lib.usv_params import usv_params_6dof


''' 
Otter USV 3-DOF CasADi model
    Constants and functions are taken from the sim to convert to casadi build instead of numpy
    see: Otter_simulator, casadi_control/lib/casadi_utils, ../../usv_params
'''

params = usv_params_6dof()

def Otter3DOF(params):
        reduced = ca.DM([[1,0,0,0,0,0], # Selects 3DOF parameters from sim 6DOF matrix 
                        [0,1,0,0,0,0],         # reduced @ X_6DOF @ reduced.T = X_3DOF 
                        [0,0,0,0,0,1]])        # rows 1,2,6 @ X @ pick colums 1,2,6 


        # 6DOF mass & damping to 3DOF
        M3 = reduced @ (params['MRB6'] + params['MA6']) @ reduced.T ### Change matrix operations to ca.mtimes(m1, m2, m3) if i mix dm/sx etc.
        D3 = reduced @ params['D6'] @ reduced.T


        # Constants from sim
        Ig = params['Ig']           # inertia around CG
        H_rg = params['H_rg']       # CG to CO transfrom 
        m_total = params['m_total'] # mass
        MA3 = reduced @ params['MA6'] @ reduced.T # 

        # USV states
        eta = ca.SX.sym('eta', 3)   # [x,y,psi]
        nu =  ca.SX.sym('nu', 3)    # [u,v,r]
        tau_u =  ca.SX.sym('u', 3)  # [X,0,N] 
        nu_c = ca.SX.sym('nu_c', 3) # Currents in body
        nu_c = ca.DM.zeros(3,1)     #override as static if not estimating currents
        x = ca.vertcat(eta, nu)     # [x,y,psi,u,v,r]

        J = B2N(eta[2]) #Body to NED transfrom 

        # Rigid-Body Coriolis to 3DOF 
        nu6  = ca.vertcat(nu[0],nu[1],0,0,0,nu[2])
        CRB6 = CRB6sx(nu6, m_total, Ig, H_rg)
        CRB3 = reduced @ CRB6 @ reduced.T 

        nu_r = nu - nu_c # Relative velocity
        
        CA3 = CA3sx(MA3, nu_r)
        C3 = CRB3 + CA3

        # Hydrodynamic linear damping + nonlinear yaw damping (from simulator)
        tau_d = D3 @ nu_r
        tau_d[2] += 10 * D3[2,2] * ca.fabs(nu_r[2]) * nu_r[2]

        # cross-flow drag 
        tau_cfd = crossFlowDrag3(params['L'], params['B'], params['T'], nu_r) 
        
        # create state derivative for 3DOF model 

        '''
        x_dot = [eta_dot, nu_dot] 
        nu_dot using fossens equation:
        M*nu(dot)+C(nu)*nu+D(nu)nu+G(eta) -> ignoring bouyancy and solving for nu_dot
        nu_dot*M = (tau-C(nu)nu-tau_d(nu)-tau_cfd(nu)) using solver
        '''

        ode = ca.vertcat(J @ nu,                                # compute the RHS forces from fossens eq (ODE for 3DOF USV)              
                        ca.solve(M3,           
                        tau_u - C3 @ nu - tau_d - tau_cfd)  
                        ) 
            

        # continous time function ([state vector, control input, nu_c], rhs_function)
        f = ca.Function('f', [x, tau_u], [ode]) # returns x_dot 
        #f = ca.Function('f_ct', [x, tau, p], [ode])
        return f
        
