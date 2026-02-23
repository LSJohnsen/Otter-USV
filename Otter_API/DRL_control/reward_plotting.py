       
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

# ---- parameters ----
d_opt = 0.1
d_acc = 1.0

# fixed test values
prev_distance = 1.2
heading_error = 0.0
u = 0.2
v = 0.1
r = 0.2
d_dot_test = 0.3

def compute_reward(d, d_dot, u, v, r, heading_error):
    reward = 0.0

    in_range = np.clip((d_acc - d) / d_acc, 0.0, 1.0)
    outside_range = 1.0 - in_range
    w = outside_range**2   # far weight

    # progress term
    reward += 1.0 * outside_range * (prev_distance - d)

    # in-range preference
    reward += 0.5 * in_range

    # optimal distance shaping
    reward -= 2.0 * ((d - d_opt) / d_acc)**2

    # overshoot penalty
    reward -= 0.6 * w * abs(d_dot)

    # velocity damping
    reward -= 0.2 * w * abs(u)
    reward -= 0.2 * w * abs(v)
    reward -= 0.15 * w * abs(r)

    # heading guidance far away
    reward += 0.05 * outside_range * np.cos(heading_error)

    return reward

# -------------------------------
# 1. reward vs distance
# -------------------------------
d_vals = np.linspace(0, 2.0, 200)
rewards = [compute_reward(d, d_dot_test, u, v, r, heading_error) for d in d_vals]

plt.figure()
plt.plot(d_vals, rewards)
plt.axvline(d_opt, linestyle="--", label="optimal")
plt.axvline(d_acc, linestyle=":", label="acceptable")
plt.title("Reward vs Distance")
plt.xlabel("distance")
plt.ylabel("reward")
plt.legend()
plt.grid()

# -------------------------------
# 2. reward vs closing speed
# -------------------------------
d = 0.3  # inside acceptable range
d_dot_vals = np.linspace(-1.0, 1.0, 200)
rewards_ddot = [compute_reward(d, dd, u, v, r, heading_error) for dd in d_dot_vals]

plt.figure()
plt.plot(d_dot_vals, rewards_ddot)
plt.title("Reward vs d_dot (closing speed)")
plt.xlabel("d_dot")
plt.ylabel("reward")
plt.grid()

# -------------------------------
# 3. reward vs velocity magnitude
# -------------------------------
vel_vals = np.linspace(0, 1.0, 200)
rewards_vel = [compute_reward(d, d_dot_test, vv, vv, vv, heading_error) for vv in vel_vals]

plt.figure()
plt.plot(vel_vals, rewards_vel)
plt.title("Reward vs velocity magnitude")
plt.xlabel("velocity magnitude")
plt.ylabel("reward")
plt.grid()

# -------------------------------
# 4. heatmap: distance vs d_dot
# -------------------------------
d_grid = np.linspace(0, 2.0, 100)
dd_grid = np.linspace(-1.0, 1.0, 100)

R = np.zeros((len(d_grid), len(dd_grid)))

for i, d in enumerate(d_grid):
    for j, dd in enumerate(dd_grid):
        R[i, j] = compute_reward(d, dd, u, v, r, heading_error)

plt.figure()
plt.imshow(R, extent=[dd_grid[0], dd_grid[-1], d_grid[-1], d_grid[0]], aspect="auto")
plt.colorbar(label="reward")
plt.xlabel("d_dot")
plt.ylabel("distance")
plt.title("Reward heatmap: distance vs d_dot")

plt.show()
