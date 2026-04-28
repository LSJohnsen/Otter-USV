import numpy as np

"""
Wave model of a two-parameter Bretschneider spectrum based on fossen p.277
"""

import numpy as np


class WaveModel:
    def __init__(
        self,
        Hs=0.3,                         # significant wave height (one third of highesty wave)
                                        # (0-0.1 calm(rippled) 11% prob, 0.1-0.5smooth (wavelets) , 0.5-1.25 slight 31%)
        Tp=2.0,                         # peak wave period
        mean_dir=0.0,                   # mean wave direction in world frame
        N=12,                           # number of wave components
        g=9.81,
        seed=None,
        gain_X=20.0,                    # scaling from wave amplitude to surge force
        gain_Y=35.0,                    # scaling from wave amplitude to sway force
        gain_N=8.0,                     # scaling from wave amplitude to yaw moment
        spread_std=np.deg2rad(20.0),    # directional spread around mean wave direction
    ):

        self.Hs = Hs
        self.Tp = Tp
        self.mean_dir = mean_dir
        self.N = N
        self.g = g
        self.gain_X = gain_X
        self.gain_Y = gain_Y
        self.gain_N = gain_N
        self.spread_std = spread_std

        rng = np.random.default_rng(seed)

        wp = 2.0 * np.pi / Tp                                   # peak angular frequency

        w_min = 0.5 * wp                                        # lower sampled frequency
        w_max = 2.0 * wp                                        # upper sampled frequency
        self.omega = np.linspace(w_min, w_max, N)               # sampled wave frequencies
        dw = self.omega[1] - self.omega[0] if N > 1 else 0.1    # frequency spacing

        S = self.bretschneider_spectrum(self.omega, Hs, Tp)     # wave energy distribution
        self.A = np.sqrt(2.0 * S * dw)                          # wave amplitudes from spectrum
        self.phase = rng.uniform(0.0, 2.0 * np.pi, N)           # random phase for each component

        self.theta = rng.normal(mean_dir, spread_std, N)        # direction of each wave component
        self.k = (self.omega ** 2) / g                          # magnitude of wave number
        self.kx = self.k * np.cos(self.theta)
        self.ky = self.k * np.sin(self.theta)
        

        # force amplitude contributed by each wave component
        # projected into global x/y directions first
        self.X_amp = gain_X * self.A * np.cos(self.theta)       # surge-related force amplitude contribution
        self.Y_amp = gain_Y * self.A * np.sin(self.theta)       # sway-related force amplitude contribution
        self.N_amp = gain_N * self.A * np.sin(self.theta)       # yaw-related moment amplitude contribution

    @staticmethod
    def bretschneider_spectrum(omega, Hs, Tp):
        wp = 2.0 * np.pi / Tp                                   # peak angular frequency
        omega = np.maximum(omega, 1e-6)                         # avoid divide-by-zero

        '''
        p.277 (10.52)
        '''
        S = (
            1.25                                                # Bretschneider coefficient
            * (wp**4 / omega**5)                                # spectral shape
            * (Hs**2)                                           # total energy scaling
            * np.exp(-1.25 * (wp / omega) ** 4)                 # decay away from peak
        )
        return S

    def get_tau_wave(self, t, eta, nu):
        x = eta[0]                                                 # current vessel x-position
        y = eta[1]                                                 # current vessel y-position
        psi_body = eta[5]                                          # vessel heading

        psi = self.kx * x + self.ky * y - self.omega * t + self.phase
        wave_signal = np.cos(psi)

        X_world = np.sum(self.X_amp * wave_signal)                 # wave force in world x
        Y_world = np.sum(self.Y_amp * wave_signal)                 # wave force in world y
        N_wave = np.sum(self.N_amp * wave_signal)                  # wave yaw moment

        c = np.cos(psi_body)
        s = np.sin(psi_body)

        X_body = c * X_world + s * Y_world                         # wave force in surge
        Y_body = -s * X_world + c * Y_world                        # wave force in sway

        tau_wave = np.array([X_body, Y_body, 0.0, 0.0, 0.0, N_wave], dtype=float)
        return tau_wave