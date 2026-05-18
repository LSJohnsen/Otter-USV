import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "reward_straight_dist.csv")
csv_path_iae = os.path.join(BASE_DIR, "iae_straight_dist.csv")

def plot_reward_history(csv_path):
    df = pd.read_csv(csv_path)

    if "episode" not in df.columns or "reward" not in df.columns:
        raise ValueError("CSV must contain 'episode' and 'reward' columns")

    episodes = df["episode"]
    rewards = df["reward"]

    plt.figure()
    plt.plot(episodes, rewards, linewidth=0.8, label="Episode reward")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward History")
    plt.grid(True)
    plt.legend()
    plt.savefig("reward_history.pdf", bbox_inches="tight")
    plt.show()


def plot_iae_distance_history(csv_path):
    df = pd.read_csv(csv_path)

    if "episode" not in df.columns or "IAE_distance" not in df.columns:
        raise ValueError("CSV must contain 'episode' and 'IAE_distance' columns")

    episodes = df["episode"]
    iae_dist = df["IAE_distance"]

    plt.figure()
    plt.plot(episodes, iae_dist, linewidth=0.8, label="IAE Distance")

    plt.xlabel("Episode")
    plt.ylabel("IAE Distance")
    plt.title("IAE Distance During Training")
    plt.grid(True)
    plt.legend()
    plt.savefig("iae_distance_history.pdf", bbox_inches="tight")
    plt.show()  


def plot_reward_history_smoothed(csv_path, window=20):
    df = pd.read_csv(csv_path)

    rewards = df["reward"]
    smoothed = rewards.rolling(window=window).mean()

    plt.figure()
    plt.plot(df["episode"], rewards, alpha=0.3, label="Raw")
    plt.plot(df["episode"], smoothed, linewidth=2, label=f"Smoothed ({window})")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward History (Smoothed)")
    plt.grid(True)
    plt.legend()
    plt.savefig("reward_history_smoothed.pdf", bbox_inches="tight")
    plt.show()

def plot_reward_per_time_smoothed(csv_path, window=20):
    df = pd.read_csv(csv_path)

    if not all(col in df.columns for col in ["episode", "reward", "length"]):
        raise ValueError("CSV must contain 'episode', 'reward', and 'length' columns")

    # reward per step (time-normalized)
    reward_per_step = df["reward"] / df["length"]

    # smoothing
    smoothed = reward_per_step.rolling(window=window, min_periods=1).mean()

    plt.figure()
    plt.plot(df["episode"], reward_per_step, alpha=0.3, label="Raw (per step)")
    plt.plot(df["episode"], smoothed, linewidth=2, label=f"Smoothed ({window})")

    plt.xlabel("Episode")
    plt.ylabel("Reward per Step")
    plt.title("Training Reward per Step (Time-Normalized)")
    plt.grid(True)
    plt.legend()

    plt.margins(x=0)
    plt.savefig("reward_per_time_smoothed.pdf", bbox_inches="tight")
    plt.show()

def plot_iae_distance_history_smoothed(csv_path, window=20, percentile=98):
    df = pd.read_csv(csv_path)

    if "episode" not in df.columns or "IAE_distance" not in df.columns:
        raise ValueError("CSV must contain 'episode' and 'IAE_distance' columns")

    episodes = df["episode"]
    iae = df["IAE_distance"]

    # smoothing
    smoothed = iae.rolling(window=window).mean()

    # compute y-limit based on percentile (ignore extreme outliers)
    ymax = np.percentile(iae, percentile)

    plt.figure()
    plt.plot(episodes, iae, alpha=0.3, label="Raw IAE Distance")
    plt.plot(episodes, smoothed, linewidth=2, label=f"Smoothed ({window})")
    plt.margins(x=0)

    plt.ylim(0, ymax * 1.1)  # small margin above percentile

    plt.xlabel("Episode")
    plt.ylabel("IAE Distance")
    plt.title(f"IAE Distance During Training (Smoothed, {percentile}th percentile clipped)")
    plt.grid(True)
    plt.legend()
    plt.savefig("iae_distance_history_smoothed.pdf", bbox_inches="tight")
    plt.show()

plot_reward_history(csv_path)
plot_reward_history_smoothed(csv_path)
plot_reward_per_time_smoothed(csv_path)
plot_iae_distance_history(csv_path_iae)
plot_iae_distance_history_smoothed(csv_path_iae, window=20)