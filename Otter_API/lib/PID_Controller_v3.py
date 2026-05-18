import time
import math


"""
based on v2, cleaner implementation
"""
class PIDController:
    def __init__(self,
                 kp_surge, ki_surge, kd_surge,
                 kp_yaw,   ki_yaw,   kd_yaw,
                 Imax_surge=1000.0,
                 Imax_yaw=40.0,
                 dp_reverse_radius=3.0,
                 moving_target=True):

        self.kp_surge = kp_surge
        self.ki_surge = ki_surge
        self.kd_surge = kd_surge
        self.kp_yaw = kp_yaw
        self.ki_yaw = ki_yaw
        self.kd_yaw = kd_yaw

        self.integral_surge = 0.0
        self.prev_error_surge = 0.0
        self.prev_time_surge = None
        self.integral_yaw = 0.0
        self.prev_error_yaw = 0.0
        self.prev_time_yaw = None

        self.Imax_surge = Imax_surge
        self.Imin_surge = -Imax_surge
        self.Imax_yaw = Imax_yaw
        self.Imin_yaw = -Imax_yaw

        self.moving_target = moving_target
        self.dp_reverse_radius = dp_reverse_radius

    def calculate_surge(self, surge_radius, distance_to_target,
                        yaw_setpoint, yaw_measured):

        current_time = time.time()
        dt = (current_time - self.prev_time_surge) if self.prev_time_surge else 0.0
        self.prev_time_surge = current_time

        error = distance_to_target - surge_radius
        yaw_error = (yaw_setpoint - yaw_measured + math.pi) % (2 * math.pi) - math.pi

        target_is_behind = abs(yaw_error) > (math.pi / 2)

        if target_is_behind:
            error = -error

        # Integral
        if dt > 0.0:
            if self.moving_target:
                # Target tracking: always accumulate (original behaviour)
                self.integral_surge += error * dt
            else:
                # DP: only accumulate integral close to target to close final gap
                if distance_to_target < self.dp_reverse_radius:
                    self.integral_surge += error * dt
                else:
                    self.integral_surge = 0.0  # flush during approach, no windup
            self.integral_surge = max(min(self.integral_surge, self.Imax_surge), self.Imin_surge)

        # Derivative
        derivative = ((error - self.prev_error_surge) / dt) if dt > 0.0 else 0.0
        self.prev_error_surge = error

        output = self.kp_surge * error + self.ki_surge * self.integral_surge + self.kd_surge * derivative

        if self.moving_target:
            # Original target tracking: suppress surge proportionally to yaw error
            yaw_error_scale = yaw_error / math.pi
            output = output * (1 - abs(yaw_error_scale))
        else:
            # DP: reverse zone logic
            in_reverse_zone = target_is_behind and (distance_to_target < self.dp_reverse_radius)
            if in_reverse_zone:
                angle_blend = (abs(yaw_error) - math.pi / 2) / (math.pi / 2)
                angle_blend = angle_blend ** 2
                output = output * angle_blend * 0.5
            else:
                yaw_error_scale = yaw_error / math.pi
                output = output * (1 - abs(yaw_error_scale))

        return output

    def calculate_yaw(self, setpoint, measured_value,
                      surge_radius, distance_to_target):

        current_time = time.time()
        dt = (current_time - self.prev_time_yaw) if self.prev_time_yaw else 0.0
        self.prev_time_yaw = current_time

        error = (setpoint - measured_value + math.pi) % (2 * math.pi) - math.pi

        if abs(error) < 0.017 and distance_to_target < 6.0:
            self.integral_yaw = 0.0
            error = 0.0

        if not self.moving_target:
            # DP only: mute yaw in reverse zone
            target_is_behind = abs(error) > (math.pi / 2)
            in_reverse_zone = target_is_behind and (distance_to_target < self.dp_reverse_radius)
            if in_reverse_zone:
                self.integral_yaw = 0.0
                return 0.0

        # Integral
        if dt > 0.0:
            self.integral_yaw += error * dt
            self.integral_yaw = max(min(self.integral_yaw, self.Imax_yaw), self.Imin_yaw)

        derivative = ((error - self.prev_error_yaw) / dt) if dt > 0.0 else 0.0
        self.prev_error_yaw = error

        return (self.kp_yaw * error
              + self.ki_yaw * self.integral_yaw
              + self.kd_yaw * derivative)
    

class SurgePIDAdapter:
    def __init__(self, pid_obj):
        self.pid = pid_obj

    def calculate_surge(self, surge_radius, distance_to_target, yaw_setpoint, yaw_measured):
        return self.pid.calculate_surge(surge_radius, distance_to_target, yaw_setpoint, yaw_measured)


class YawPIDAdapter:
    def __init__(self, pid_obj):
        self.pid = pid_obj

    def calculate_yaw(self, setpoint, measured_value, surge_radius, distance_to_target):
        return self.pid.calculate_yaw(setpoint, measured_value, surge_radius, distance_to_target)   
