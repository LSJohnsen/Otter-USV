import time
import math


"""
based on v2, cleaner implementation
"""
class PIDController:

    def __init__(self,
                 kp_surge, ki_surge, kd_surge,
                 kp_yaw,   ki_yaw,   kd_yaw,
                 Imax_surge=10.0,
                 Imax_yaw=40.0):

        # gains
        self.kp_surge = kp_surge
        self.ki_surge = ki_surge
        self.kd_surge = kd_surge
        self.kp_yaw = kp_yaw
        self.ki_yaw = ki_yaw
        self.kd_yaw = kd_yaw

        # Sstates
        self.integral_surge = 0.0
        self.prev_error_surge = 0.0
        self.prev_time_surge = None
        self.integral_yaw = 0.0
        self.prev_error_yaw = 0.0
        self.prev_time_yaw = None

        # Integrator limits
        self.Imax_surge = Imax_surge
        self.Imin_surge = -Imax_surge
        self.Imax_yaw = Imax_yaw
        self.Imin_yaw = -Imax_yaw


    def calculate_surge(self, surge_radius, distance_to_target,
                        yaw_setpoint, yaw_measured):
            
        current_time = time.time()
        if self.prev_time_surge is None:
            dt = 0.0
        else:
            dt = current_time - self.prev_time_surge
        self.prev_time_surge = current_time

        
        error = distance_to_target - surge_radius
        yaw_error = (yaw_setpoint - yaw_measured + math.pi) % (2 * math.pi) - math.pi

        if (yaw_error > (math.pi/2) or yaw_error < -(math.pi/2)):       # Allows the thrusters to go in reverse if the target is passed
            error = -error

        # integral
        if dt > 0.0:
            self.integral_surge += error * dt
            self.integral_surge = max(min(self.integral_surge, self.Imax_surge), self.Imin_surge)

        # derivative
        if dt > 0.0:
            derivative = (error - self.prev_error_surge) / dt
        else:
            derivative = 0.0
        self.prev_error_surge = error

        # Final pid
        output = self.kp_surge * error + self.ki_surge * self.integral_surge + self.kd_surge * derivative
        yaw_error_scale = yaw_error / math.pi                                                              
        output = output * (1 - abs(yaw_error_scale))                                                        # zero surge if large yaw error

        return output
    
    def calculate_yaw(self, setpoint, measured_value,
                      surge_radius, distance_to_target):
        
        current_time = time.time()
        if self.prev_time_yaw is None:
            dt = 0.0
        else:
            dt = current_time - self.prev_time_yaw
        self.prev_time_yaw = current_time

        error = (setpoint - measured_value + math.pi) % (2 * math.pi) - math.pi

        # Resets the integral when the target is reached. test others?
        if abs(error) < 0.017 and distance_to_target < 6.0: 
            self.integral_yaw = 0.0
            error = 0.0

        # int
        if dt > 0.0:
            self.integral_yaw += error * dt
            self.integral_yaw = max(min(self.integral_yaw, self.Imax_yaw), self.Imin_yaw)

        # derivative
        if dt > 0.0:
            derivative = (error - self.prev_error_yaw) / dt
        else:
            derivative = 0.0
        self.prev_error_yaw = error

        # Final pid
        output = (self.kp_yaw * error
            + self.ki_yaw * self.integral_yaw
            + self.kd_yaw * derivative)

        return output
    

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
