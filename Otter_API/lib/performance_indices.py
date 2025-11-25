
import numpy as np

def IAE(IAE_dist_prev, IAE_head_prev, distance_to_target, heading_error, sampleTime):
        
        IAE_dist = IAE_dist_prev + abs(distance_to_target) * sampleTime
        IAE_head = IAE_head_prev + abs(heading_error) * sampleTime
        return IAE_dist, IAE_head