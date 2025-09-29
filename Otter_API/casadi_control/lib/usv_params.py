
# lib/usv_params.py
import math
import numpy as np
import casadi as ca

'''
From Otter_simulator: returns dictionary of parameters with casadi conversion 
'''

def _Smtrx(a):
    ax, ay, az = a
    return np.array([[ 0, -az,  ay],
                     [ az,  0, -ax],
                     [-ay,  ax,  0 ]], dtype=float)

def _Hmtrx(r):
    # H = [[I3, 0],
    #      [S(r), I3]]
    S = _Smtrx(r)
    H = np.zeros((6,6), float)
    H[0:3,0:3] = np.eye(3)
    H[3:6,3:6] = np.eye(3)
    H[3:6,0:3] = S
    return H

def usv_params_6dof(starting_yaw_angle: float = 0.0) -> dict:
    # Constants
    D2R = math.pi / 180     # deg2rad
    g = 9.81                # acceleration of gravity (m/s^2)
    rho = 1026              # density of water (kg/m^3)

    beta_c = starting_yaw_angle * D2R

    # Initialize the Otter USV model
    T_n = 1.0               # propeller time constants (s)
    L = 2.0                 # Length (m)
    B = 1.08                # beam (m)
    nu = np.array([0, 0, 0, 0, 0, 0], float)   # velocity vector
    u_actual = np.array([0, 0], float)         # propeller revolution states

    controls = ["Left propeller shaft speed (rad/s)", "Right propeller shaft speed (rad/s)"]
    dimU = len(controls)

    # Vehicle parameters
    m = 55.0                                            # mass (kg)
    mp = 25.0                                           # Payload (kg)
    m_total = m + mp
    rp = np.array([0.05, 0, -0.35], float)             # location of payload (m)
    rg = np.array([0.2, 0, -0.2], float)               # CG for hull only (m)
    rg = (m * rg + mp * rp) / (m + mp)                 # CG corrected for payload
    S_rg = _Smtrx(rg)
    H_rg = _Hmtrx(rg)
    S_rp = _Smtrx(rp)

    R44 = 0.4 * B   # radii of gyration (m)
    R55 = 0.25 * L
    R66 = 0.25 * L
    T_yaw = 1.0     # time constant in yaw (s)
    Umax = 6 * 0.5144   # max forward speed (m/s)

    # Data for one pontoon
    B_pont = 0.25   # beam of one pontoon (m)
    y_pont = 0.395  # distance from centerline to waterline centroid (m)
    Cw_pont = 0.75  # waterline area coefficient (-)
    Cb_pont = 0.4   # block coefficient, computed from m = 55 kg

    # Inertia dyadic, volume displacement and draft
    nabla = (m + mp) / rho                      # volume
    T = nabla / (2 * Cb_pont * B_pont * L)      # draft
    Ig_CG = m * np.diag(np.array([R44 ** 2, R55 ** 2, R66 ** 2]))
    Ig = Ig_CG - m * S_rg @ S_rg - mp * S_rp @ S_rp

    # Experimental propeller data including lever arms
    l1 = -y_pont     # lever arm, left propeller (m)
    l2 =  y_pont     # lever arm, right propeller (m)
    k_pos = 0.02216 / 2  # Positive Bollard, one propeller
    k_neg = 0.01289 / 2  # Negative Bollard, one propeller
    n_max = math.sqrt((0.5 * 24.4 * g) / k_pos)   # max. prop. rev.
    n_min = -math.sqrt((0.5 * 13.6 * g) / k_neg)  # min. prop. rev.

    # MRB_CG = [ (m+mp) * I3   O3      (Fossen 2021, Chapter 3)
    #               O3        Ig ]
    MRB_CG = np.zeros((6, 6))
    MRB_CG[0:3, 0:3] = (m + mp) * np.identity(3)
    MRB_CG[3:6, 3:6] = Ig
    MRB = H_rg.T @ MRB_CG @ H_rg

    # Hydrodynamic added mass (best practice)
    Xudot = -0.1 * m
    Yvdot = -1.5 * m
    Zwdot = -1.0 * m
    Kpdot = -0.2 * Ig[0, 0]
    Mqdot = -0.8 * Ig[1, 1]
    Nrdot = -1.7 * Ig[2, 2]

    MA = -np.diag([Xudot, Yvdot, Zwdot, Kpdot, Mqdot, Nrdot])

    # System mass matrix
    M = MRB + MA
    Minv = np.linalg.inv(M)

    # Hydrostatic quantities (Fossen 2021, Chapter 4)
    Aw_pont = Cw_pont * L * B_pont  # waterline area, one pontoon
    I_T = (
        2
        * (1 / 12)
        * L
        * B_pont ** 3
        * (6 * Cw_pont ** 3 / ((1 + Cw_pont) * (1 + 2 * Cw_pont)))
        + 2 * Aw_pont * y_pont ** 2
    )
    I_L = 0.8 * 2 * (1 / 12) * B_pont * L ** 3
    KB = (1 / 3) * (5 * T / 2 - 0.5 * nabla / (L * B_pont))
    BM_T = I_T / nabla  # BM values
    BM_L = I_L / nabla
    KM_T = KB + BM_T    # KM values
    KM_L = KB + BM_L
    KG = T - rg[2]
    GM_T = KM_T - KG    # GM values
    GM_L = KM_L - KG

    G33 = rho * g * (2 * Aw_pont)  # spring stiffness
    G44 = rho * g * nabla * GM_T
    G55 = rho * g * nabla * GM_L
    G_CF = np.diag([0, 0, G33, G44, G55, 0])  # spring stiff. matrix in CF
    LCF = -0.2
    H_cf = _Hmtrx(np.array([LCF, 0.0, 0.0]))  # transform G_CF from CF to CO
    G = H_cf.T @ G_CF @ H_cf

    # Natural frequencies
    w3 = math.sqrt(G33 / M[2, 2]) #heave
    w4 = math.sqrt(G44 / M[3, 3]) #roll
    w5 = math.sqrt(G55 / M[4, 4]) #pitch

    # Linear damping terms (hydrodynamic derivatives)
    Xu = -24.4 * g / Umax  # specified using the maximum speed
    Yv = 0
    Zw = -2 * 0.3 * w3 * M[2, 2]  # specified using relative damping 
    Kp = -2 * 0.2 * w4 * M[3, 3]  # sqrt(g44/m44)  
    Mq = -2 * 0.4 * w5 * M[4, 4]
    Nr = -M[5, 5] / T_yaw  # specified by the time constant T_yaw

    D = -np.diag([Xu, Yv, Zw, Kp, Mq, Nr])

    mass = m + mp

    return {
        # scalars
        "D2R": D2R, "g": g, "rho": rho, "beta_c": beta_c,
        "T_n": T_n, "L": L, "B": B, "T": T,
        "controls": controls, "dimU": dimU,
        "m": m, "mp": mp, "m_total": m_total,
        "l1": l1, "l2": l2, "k_pos": k_pos, "k_neg": k_neg, "n_max": n_max, "n_min": n_min,
        "Umax": Umax, "T_yaw": T_yaw,

        # vectors (NumPy)
        "nu0": nu, "u_actual0": u_actual, "rp": rp, "rg": rg,

        # matrices (CasADi DM)
        "H_rg":  ca.DM(H_rg),
        "Ig":    ca.DM(Ig),
        "MRB6":  ca.DM(MRB),
        "MA6":   ca.DM(MA),
        "M6":    ca.DM(M),
        "Minv6": ca.DM(Minv),
        "D6":    ca.DM(D),
        "G6":    ca.DM(G),

        # convenience: 3DOF reduction (surge, sway, yaw)
        "R3": ca.DM([[1,0,0,0,0,0],
                     [0,1,0,0,0,0],
                     [0,0,0,0,0,1]]),
    }
