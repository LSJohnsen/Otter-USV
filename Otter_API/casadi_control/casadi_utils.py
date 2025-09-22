
import casadi as ca
from lib.gnc import *

'''
Utils from otter_simulator and GNC for use with CasADi:
'''

def Smtrx(a):
    #S = Smtrx(a) computes the 3x3 vector skew-symmetric matrix S(a) = -S(a)'. (Fossen)
    return ca.vertcat(
        ca.horzcat(0,    -a[2],  a[1]),
        ca.horzcat(a[2],    0,  -a[0]),
        ca.horzcat(-a[1], a[0],   0)
    )   

def B2N(psi):
    # Rotation matrix from body to NED
    return ca.vertcat(  
        ca.horzcat(ca.cos(psi), -ca.sin(psi), 0),
        ca.horzcat(ca.sin(psi),  ca.cos(psi), 0),
        ca.horzcat(0,            0,           1)
    )

def CRB6sx(nu6, m_total, Ig, H_rg):
        #Rigid-body Coriolis/centripetal in CO (see otter_simulator/gnc) 
        omega = nu6[3:6]
        CRB_CG = ca.SX.zeros(6,6)
        CRB_CG[0:3,0:3] = m_total * Smtrx(omega)
        CRB_CG[3:6,3:6] = -Smtrx(Ig @ omega)
        return H_rg.T @ CRB_CG @ H_rg

def CA3sx(MA3, nu3):
     #Added-mass Coriolis for 3DOF (see otter_simulator/gnc)
    C = ca.SX.zeros(3,3)
    C[0,2] = -MA3[1,1]*nu3[1] - MA3[1,2]*nu3[2]
    C[1,2] =  MA3[0,0]*nu3[0]
    C[2,0] = -C[0,2]
    C[2,1] = -C[1,2]
    return C


def crossFlowDrag3(L, B, T, nu_r):
    """
    3-DOF cross-flow drag using strip theory (Fossen).
    Returns [X, Y, N] in body
    """
    rho = 1026.0
    n   = 20
    dx  = L / n

    Cd_2D = Hoerner(B, T) 

    Yh = 0
    Nh = 0
    xL = -L/2

    v_r = nu_r[1]     # sway 3dof
    r   = nu_r[2]     # yaw rate 3dof

    for _ in range(n+1):
        Ucf = ca.fabs(v_r + xL*r) * (v_r + xL*r)        
        Yh  = Yh - 0.5 * rho * T * Cd_2D * Ucf * dx     
        Nh  = Nh - 0.5 * rho * T * Cd_2D * xL * Ucf * dx
        xL += dx

    return -ca.vertcat(0, Yh, Nh) # return as negative to keep simple convention in solver


