
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