import math
import numpy as np


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class LOSCascadePIDController:
    """
    LOS guidance + cascade PID controller

    Outer loop:
        position error for desired heading psi_des and desired surge speed u_des

    Inner loop:
        surge speed error -> tau_X
        heading error + yaw-rate damping -> tau_N

        tau_X, tau_N = controller.calculate_control(
            target_pos=np.array([target_north, target_east]),
            eta=eta,
            nu=nu,
            dt=sample_time
        )
    """

    def __init__(
        self,
        
        kp_u=80.0,
        ki_u=5.0,
        kd_u=20.0,

        
        kp_psi=80.0,
        ki_psi=2.0,
        kd_r=25.0,

        # Outer-loop guidance
        u_max=1.0,
        d_slow=5.0,
        hold_radius=1.0,
        position_tolerance=0.2,

        # force limits
        tau_X_max=150.0,
        tau_X_min=-116.0,
        tau_N_max=110.0,

        # I limits
        Imax_u=2.0,
        Imax_psi=1.0,

        # anti-windup / behavior
        use_integral=True,
        reset_integral_inside_hold=True,
        allow_reverse=False,
        hold_heading=None,
    ):
        # Inner-loop gains
        self.kp_u = kp_u
        self.ki_u = ki_u
        self.kd_u = kd_u

        self.kp_psi = kp_psi
        self.ki_psi = ki_psi
        self.kd_r = kd_r

        # Guidance parameters
        self.u_max = u_max
        self.d_slow = d_slow
        self.hold_radius = hold_radius
        self.position_tolerance = position_tolerance

        # Limits
        self.tau_X_max = tau_X_max
        self.tau_X_min = tau_X_min
        self.tau_N_max = tau_N_max

        # Integrator limits
        self.Imax_u = Imax_u
        self.Imin_u = -Imax_u
        self.Imax_psi = Imax_psi
        self.Imin_psi = -Imax_psi

        # Options
        self.use_integral = use_integral
        self.reset_integral_inside_hold = reset_integral_inside_hold
        self.allow_reverse = allow_reverse

        # If None, heading is initialized from first LOS heading
        self.hold_heading = hold_heading

        # Internal states
        self.integral_u = 0.0
        self.integral_psi = 0.0
        self.prev_u_error = 0.0
        self.prev_distance = None

        self.last_tau_X = 0.0
        self.last_tau_N = 0.0
        self.last_u_des = 0.0
        self.last_psi_des = 0.0
        self.last_distance = 0.0
        self.last_heading_error = 0.0
        self.last_mode = "approach"

    def reset(self, hold_heading=None):
        self.integral_u = 0.0
        self.integral_psi = 0.0
        self.prev_u_error = 0.0
        self.prev_distance = None

        self.last_tau_X = 0.0
        self.last_tau_N = 0.0
        self.last_u_des = 0.0
        self.last_psi_des = 0.0
        self.last_distance = 0.0
        self.last_heading_error = 0.0
        self.last_mode = "approach"

        if hold_heading is not None:
            self.hold_heading = hold_heading

    def calculate_control(self, target_pos, eta, nu, dt):
        """
        Parameters
        ----------
        target_pos : array-like
            Target position. Use same coordinate convention as eta.
            Expected here: [north, east] or [x, y], as long as it matches eta.
        eta : array-like
            Vessel pose/state. Expected eta[0], eta[1], eta[5] = position and yaw.
        nu : array-like
            Vessel velocities. Expected nu[0], nu[5] = surge velocity and yaw rate.
        dt : float
            Simulation/controller timestep.

        Returns
        -------
        tau_X : float
            Desired surge force [N]
        tau_N : float
            Desired yaw moment [Nm]
        """

        dt = max(float(dt), 1e-6)

        # State
        x = float(eta[0])
        y = float(eta[1])
        psi = float(eta[5])

        u = float(nu[0])
        r = float(nu[5])

        target_x = float(target_pos[0])
        target_y = float(target_pos[1])

        # position error
        e_x = target_x - x
        e_y = target_y - y
        distance = math.hypot(e_x, e_y)

        # LOS heading
        psi_los = math.atan2(e_y, e_x)

        # initialize hold heading if not provided
        if self.hold_heading is None:
            self.hold_heading = psi_los

        # use LOS far away, fixed hold heading close to target
        if distance > self.hold_radius:
            psi_des = psi_los
            self.last_mode = "approach"
        else:
            psi_des = self.hold_heading
            self.last_mode = "hold"

        psi_error = wrap_to_pi(psi_des - psi)

        # Outer loop: desired surge speed
        # smooth decrease near target
        u_des = self.u_max * math.tanh(distance / max(self.d_slow, 1e-6))

        # stop commanding surge inside tolerance
        if distance < self.position_tolerance:
            u_des = 0.0

        # reduce forward motion if not pointing toward desired direction
        heading_gate = max(0.0, math.cos(psi_error))
        u_des *= heading_gate
  
        # keep False for safer station keeping.
        if self.allow_reverse:
            if abs(psi_error) > math.pi / 2.0:
                u_des = -u_des

        # inner surge-speed PID
        u_error = u_des - u

        if self.use_integral:
            self.integral_u += u_error * dt
            self.integral_u = np.clip(self.integral_u, self.Imin_u, self.Imax_u)

        du_error = (u_error - self.prev_u_error) / dt
        self.prev_u_error = u_error

        tau_X = (
            self.kp_u * u_error
            + self.ki_u * self.integral_u
            + self.kd_u * du_error
        )

        # inner yaw PID with measured yaw-rate damping
        if self.use_integral:
            self.integral_psi += psi_error * dt
            self.integral_psi = np.clip(self.integral_psi, self.Imin_psi, self.Imax_psi)

        tau_N = (
            self.kp_psi * psi_error
            + self.ki_psi * self.integral_psi
            - self.kd_r * r
        )

        # integrator reset near target to reduce windup
        if self.reset_integral_inside_hold and distance < self.position_tolerance:
            self.integral_u = 0.0

            if abs(psi_error) < np.deg2rad(2.0):
                self.integral_psi = 0.0

        # saturate command
        tau_X_unsat = tau_X
        tau_N_unsat = tau_N

        tau_X = float(np.clip(tau_X, self.tau_X_min, self.tau_X_max))
        tau_N = float(np.clip(tau_N, -self.tau_N_max, self.tau_N_max))

        # anti-windup
        if self.use_integral:
            if tau_X != tau_X_unsat:
                self.integral_u *= 0.95

            if tau_N != tau_N_unsat:
                self.integral_psi *= 0.95

        # Store debug values
        self.last_tau_X = tau_X
        self.last_tau_N = tau_N
        self.last_u_des = u_des
        self.last_psi_des = psi_des
        self.last_distance = distance
        self.last_heading_error = psi_error

        return tau_X, tau_N

    def get_debug_info(self):
        return {
            "mode": self.last_mode,
            "tau_X": self.last_tau_X,
            "tau_N": self.last_tau_N,
            "u_des": self.last_u_des,
            "psi_des": self.last_psi_des,
            "distance": self.last_distance,
            "heading_error": self.last_heading_error,
            "integral_u": self.integral_u,
            "integral_psi": self.integral_psi,
        }