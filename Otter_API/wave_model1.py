import numpy as np

class WaveModel:
    def __init__(
        self,
        Hs=0.3,                         # significant wave height
        Tp=2.0,                         # peak wave period
        mean_dir=0.0,                   # mean wave direction in world frame
        N=12,                           # number of wave components
        g=9.81,
        seed=None,
        gain_X=20.0,                    # scaling from wave amplitude to surge force
        gain_Y=35.0,                    # scaling from wave amplitude to sway force
        gain_N=8.0,                     # scaling from force imbalance to yaw moment
        spread_std=np.deg2rad(20.0),    # directional spread around mean wave direction
        L_eff=1.0,                      # half-distance bow-stern
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
        self.L_eff = L_eff

        rng = np.random.default_rng(seed)

        wp = 2.0 * np.pi / Tp                                   # peak angular frequency

        w_min = 0.5 * wp                                        # min frequency half of peak
        w_max = 2.0 * wp                                        # max frequency twice peak
        self.omega = np.linspace(w_min, w_max, N)               # wave frequencies distrubuted over N spectrum
        dw = self.omega[1] - self.omega[0] if N > 1 else 0.1    # frequency dt

        S = self.bretschneider_spectrum(self.omega, Hs, Tp)     

        self.A = np.sqrt(2.0 * S * dw)                          # amplitude
        self.phase = rng.uniform(0.0, 2.0 * np.pi, N)           # randomize phases for components

        self.theta = rng.normal(mean_dir, spread_std, N)        # direction of each wave component
        self.k = (self.omega ** 2) / g                          # magnitude of wave number

        self.kx = self.k * np.cos(self.theta)                   # wave vector x
        self.ky = self.k * np.sin(self.theta)                   # wave vector y

        # force in world based on gains and amplitude 
        self.X_amp = gain_X * self.A * np.cos(self.theta)       # world x wave force 
        self.Y_amp = gain_Y * self.A * np.sin(self.theta)       # world y wave force 

    @staticmethod
    def bretschneider_spectrum(omega, Hs, Tp):
        
        '''
        fossen:
        p. 277
        '''

        wp = 2.0 * np.pi / Tp                                   # peak angular frequency
        omega = np.maximum(omega, 1e-6)                         # avoid divide-by-zero
        S = (
            1.25                                                # Bretschneider coefficient
            * (wp**4 / omega**5)                                # spectral shape
            * (Hs**2)                                           # total energy scaling
            * np.exp(-1.25 * (wp / omega) ** 4)                 # decay away from peak
        )
        return S

    def _world_force_at_point(self, x, y, t):
        psi = self.kx * x + self.ky * y - self.omega * t + self.phase  # phase at this point
        wave_signal = np.cos(psi)                                       # component oscillation at this point

        X_world = np.sum(self.X_amp * wave_signal)                      # world x wave force at point
        Y_world = np.sum(self.Y_amp * wave_signal)                      # world y wave force at point
        return X_world, Y_world
    
    def get_tau_wave(self, t, eta, nu):
        x_c = eta[0]                                                    # vessel center x-position
        y_c = eta[1]                                                    # vessel center y-position
        psi_body = eta[5]                                               # vessel heading

        c = np.cos(psi_body)
        s = np.sin(psi_body)

        # bow and stern sample points in world frame
        x_bow = x_c + self.L_eff * c                                    # bow x-position
        y_bow = y_c + self.L_eff * s                                    # bow y-position
        x_stern = x_c - self.L_eff * c                                  # stern x-position
        y_stern = y_c - self.L_eff * s                                  # stern y-position

        # wave force at bow and stern
        Xw_bow, Yw_bow = self._world_force_at_point(x_bow, y_bow, t)    # world force at bow
        Xw_stern, Yw_stern = self._world_force_at_point(x_stern, y_stern, t)  # world force at stern

        # average force gives net translation
        Xw_world = 0.5 * (Xw_bow + Xw_stern)                            # net world x force
        Yw_world = 0.5 * (Yw_bow + Yw_stern)                            # net world y force

        # rotate net force into body frame
        X_body = c * Xw_world + s * Yw_world                            # wave force in surge
        Y_body = -s * Xw_world + c * Yw_world                           # wave force in sway

        # force difference between bow and stern gives yaw moment
        Fx_diff = Xw_bow - Xw_stern                                     # x-force imbalance
        Fy_diff = Yw_bow - Yw_stern                                     # y-force imbalance

        # 2D moment from lever arm ±L_eff along body x-axis
        N_wave = self.gain_N * self.L_eff * (-s * Fx_diff + c * Fy_diff)  # yaw moment from imbalance

        tau_wave = np.array([X_body, Y_body, 0.0, 0.0, 0.0, N_wave], dtype=float)
        return tau_wave