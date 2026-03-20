       
'''
    d, u, v, r = distance_to_target, nu[0], nu[1], nu[5] 
        # d_dot = (prev_distance - d) / self.sampletime

        # change based on actual UOWC system performance 
        d_opt = 0.1 # optimal tracking radius
        d_acc = 1.0 # acceptable tracking radius

        # inside acceptable range
        in_range = np.clip((d_acc - d) / d_acc, 0.0, 1.0)   # normalized to 1 when exactly above
        outside_range = 1.0 - in_range                      # 0 when close, 1 at boundary+
        w = outside_range**2   # weight decreases when closer to target     

        # Move toward target when outside range
        reward += 1.0 * outside_range * (prev_distance - d)

        # Prefer being inside acceptable range
        reward += 0.5 * in_range

        # Prefer the optimal distance (0.1)
        reward -= 2.0 * ((d - d_opt) / d_acc)**2

        # Prevent overshoot when close
        reward -= 0.6 * w * abs(d_dot)

        # slow and stable when close
        reward -= 0.2 * w * abs(u)
        reward -= 0.2 * w * abs(v)
        reward -= 0.15 * w * abs(r)

        # weak heading guidance when far away
        reward += 0.05 * outside_range * np.cos(heading_error)

        self.last_distance = float(distance_to_target)
'''

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import time

def append_reward_training_progress(csv_path: str, reward_callback):
    rewards, lengths = reward_callback.return_log()                         # get logged rewards and episode lengths
    n = min(len(rewards), len(lengths))                                     # number of finished episodes logged
    if n == 0:
        print(f"Reward CSV: No episodes logged yet; nothing to write to {csv_path}")
        return

    start_episode = 1                                                       # default starting episode index
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r") as f:
                rows = list(csv.reader(f))
                if len(rows) > 1:
                    start_episode = int(rows[-1][1]) + 1                    # continue after last saved episode
        except Exception:
            start_episode = 1

    file_exists = os.path.exists(csv_path)                                  # check whether file already exists
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["unix_time", "episode", "reward", "length"])        # write header once

        for i in range(n):
            w.writerow([
                int(time.time()),                                           # timestamp
                start_episode + i,                                          # episode number
                float(rewards[i]),                                          # episode reward
                int(lengths[i]),                                            # episode length
            ])

    print(f"Reward CSV: Appended {n} episodes to {csv_path} (starting from {start_episode})")