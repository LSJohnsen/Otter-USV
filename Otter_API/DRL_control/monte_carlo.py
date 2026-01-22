import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


csv_file = "monte_carlo_results.csv"
df = pd.read_csv(csv_file)

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
    plt.title(f"Monte-Carlo CDF of {key}")
    plt.grid(True)
    plt.show()

plot_histogram(df, key="IAE_distance")
plot_cdf(df, key="IAE_distance")