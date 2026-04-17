import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

csv_IAE = r".\DRL_control\ppo_saves\iae_training_progress.csv"
csv_mc = "monte_carlo_results.csv"
df = pd.read_csv(csv_mc)

def plot_histogram(df, key="IAE_distance", bins=30):
    plt.figure()
    plt.hist(df[key], bins=bins, density=True)
    plt.xlabel(key.replace("_", " "))
    plt.ylabel("Probability density")
    plt.title(f"Monte-Carlo Histogram of {key}")
    plt.grid(True)
    plt.show()

def plot_cdf(df, key="IAE_distance"):
    values = np.sort(df[key].to_numpy())
    cdf = np.arange(1, len(values) + 1) / len(values)

    plt.figure()
    plt.plot(values, cdf)
    plt.xlabel(key.replace("_", " "))
    plt.ylabel("CDF")
    plt.title(f"Monte-Carlo CDF of distance IAE")
    plt.grid(True)
    plt.show()



# gpt generated plot
def plot_iae_from_csv(csv_path, use_global_episode=False, plot_heading=False):
    """
    Reads a CSV log and plots IAE vs episode.

    Parameters
    ----------
    csv_path : str
        Path to CSV file.
    use_global_episode : bool
        If True, ignores episode column and uses continuous index.
    plot_heading : bool
        If True, also plots IAE_heading.
    """

    # Read CSV safely
    df = pd.read_csv(csv_path, header=0)

    # Extract columns by index
    episode_col = 1
    iae_dist_col = 3
    iae_head_col = 4 if df.shape[1] > 4 else None

    # Convert to numeric
    df["episode"] = pd.to_numeric(df.iloc[:, episode_col], errors="coerce")
    df["IAE_distance"] = pd.to_numeric(df.iloc[:, iae_dist_col], errors="coerce")

    if iae_head_col is not None:
        df["IAE_heading"] = pd.to_numeric(df.iloc[:, iae_head_col], errors="coerce")

    # Drop bad rows
    df = df.dropna(subset=["episode", "IAE_distance"])

    # Episode axis
    if use_global_episode:
        x = range(1, len(df) + 1)
        xlabel = "Global Episode"
    else:
        x = df["episode"]
        xlabel = "Episode"

    # Plot distance IAE
    plt.figure(figsize=(8, 4))
    plt.plot(x, df["IAE_distance"], linewidth=1, label="IAE Distance")

    # Optional heading plot
    if plot_heading and "IAE_heading" in df.columns:
        plt.plot(x, df["IAE_heading"], linewidth=1, label="IAE Heading")

    plt.xlabel(xlabel)
    plt.ylabel("IAE")
    plt.title("IAE per Episode")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

#plot_histogram(df, key="IAE_distance")
#plot_cdf(df, key="IAE_distance")


plot_iae_from_csv(csv_IAE)