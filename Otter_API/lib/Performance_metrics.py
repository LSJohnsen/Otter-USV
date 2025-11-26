
import numpy as np


class PerformanceMetrics:
    def __init__(self):
        self.IAE_dist = 0.0
        self.IAE_head = 0.0
        self.ISU_sum = 0.0    # raw integral of u^2
        self.time = 0.0       # total simulation time

    def reset(self):
        self.IAE_dist = 0.0
        self.IAE_head = 0.0
        self.ISU_sum = 0.0
        self.IAU_sum = 0.0
        self.time = 0.0

    def update(self, distance_to_target, heading_error, u1, u2, dt):
        # Integral of Absolute Error
        self.IAE_dist += abs(distance_to_target) * dt
        self.IAE_head += abs(heading_error) * dt
        
        # Integral of Squared Control Effort
        self.ISU_sum += (u1**2 + u2**2) * dt
        
        # Integral of absolute control effort 
        self.IAU += (abs(u1) + abs(u2)) * dt
        # Track elapsed time for normalization
        self.time += dt

    def get_IAE(self):
        return self.IAE_dist, self.IAE_head

    def get_IAU(self):
        return self.IAU

    def get_ISU(self):
        return self.ISU_sum

    def get_ISU_normalized(self):
        if self.time == 0:
            return 0.0
        return self.ISU_sum / self.time
