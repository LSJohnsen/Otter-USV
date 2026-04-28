
import numpy as np


class WindModel:
    def __init__(
        self,
        mean_speed=5.0,                    # mean wind speed
        mean_dir=0.0,                      # wind direction in world frame
        gust_std=0.5,                      # standard deviation of gust speed [m/s]
        gust_time_constant=5.0,            # gust time constant
        rho_air=1.225,                     # air density [kg/m^3]
        Cx=0.5,                            # surge wind coefficient
        Cy=0.875,                            # sway wind coefficient
        Cn=0.15,                           # yaw wind coefficient
        A_front=0.15,                      # frontal area [m^2]
        A_side=0.35,                       # lateral area [m^2]
        L_ref=1.0,                         # reference lever arm
        seed=None,
    ):
        self.mean_speed = mean_speed
        self.mean_dir = mean_dir
        self.gust_std = gust_std
        self.gust_time_constant = gust_time_constant

        self.rho_air = rho_air
        self.Cx = Cx
        self.Cy = Cy
        self.Cn = Cn
        self.A_front = A_front
        self.A_side = A_side
        self.L_ref = L_ref

        self.rng = np.random.default_rng(seed)
        self.v_gust = 0.0                  # scalar gust state [m/s]

    def reset(self):
        self.v_gust = 0.0

    def step_gust(self, dt):
        """
        First-order gust model:
            dv_g/dt = -(1/Tg) v_g + (Kw/Tg) n
        """
        if self.gust_time_constant <= 1e-6:
            self.v_gust = self.gust_std * self.rng.normal()
            return self.v_gust

        # Kw so stationary variation is roughly tied to gust_std
        Kw = self.gust_std * np.sqrt(2.0 * self.gust_time_constant)

        n = self.rng.normal()              # Gaussian excitation
        dv = (
            -(1.0 / self.gust_time_constant) * self.v_gust
            + (Kw / self.gust_time_constant) * n
        )
        self.v_gust += dt * dv
        return self.v_gust

    def get_tau_wind(self, dt, eta, nu):
        """
        wind forces [X_w, Y_w, 0, 0, 0, N_w]
        """
        psi = eta[5]                       # vessel heading
        u = nu[0]                          # surge velocity in body
        v = nu[1]                          # sway velocity in body

        # update gust
        gust = self.step_gust(dt)

        # total wind speed in world frame
        Vw = max(0.0, self.mean_speed + gust)

        # wind velocity in world frame
        Vwx_world = Vw * np.cos(self.mean_dir)
        Vwy_world = Vw * np.sin(self.mean_dir)

        # rotate wind into body frame
        c = np.cos(psi)
        s = np.sin(psi)

        Vwx_body = c * Vwx_world + s * Vwy_world
        Vwy_body = -s * Vwx_world + c * Vwy_world

        # relative wind in body frame
        u_rw = Vwx_body - u
        v_rw = Vwy_body - v

        # quadratic aerodynamic loads
        X_w = 0.5 * self.rho_air * self.Cx * self.A_front * u_rw * abs(u_rw)
        Y_w = 0.5 * self.rho_air * self.Cy * self.A_side  * v_rw * abs(v_rw)
        N_w = 0.5 * self.rho_air * self.Cn * self.A_side  * self.L_ref * v_rw * abs(v_rw)

        tau_wind = np.array([X_w, Y_w, 0.0, 0.0, 0.0, N_w], dtype=float)
        return tau_wind