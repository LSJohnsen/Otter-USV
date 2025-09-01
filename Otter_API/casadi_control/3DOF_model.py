
import casadi as ca
import numpy as np

'''
Otter USV 3-DOF CasADi model
'''

# State vectors

eta = ca.SX.sym("eta",3)
nu = ca.SX.sym("nu",3)

u = ca.SX.sym("u",3) # Otter control [surge, sway (0), yaw]
x = ca.vertcat(eta, nu) # x = [eta, nu]^t = [x, y, psi, u, v, r]^t

# Mass, Coriolis & Damping matrices

