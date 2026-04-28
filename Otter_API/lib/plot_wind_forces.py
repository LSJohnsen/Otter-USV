import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from wind_model import WindModel


def plot_wind_forces(model, T=50.0, dt=0.05):
    """
    Wind forces over time with fixed vessel position/body velocities.
    """

    eta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi/4])
    nu = np.zeros(6)

    t_vals = np.arange(0, T, dt)

    X_vals = []
    Y_vals = []
    N_vals = []

    model.reset()

    for _ in t_vals:
        tau = model.get_tau_wind(dt, eta, nu)

        X_vals.append(tau[0])
        Y_vals.append(tau[1])
        N_vals.append(tau[5])

    # --- FIX: match figure style ---
    plt.figure(figsize=(10, 6))

    plt.plot(t_vals, X_vals, label="Surge force X")
    plt.plot(t_vals, Y_vals, label="Sway force Y")
    plt.plot(t_vals, N_vals, label="Yaw moment N")

    plt.xlabel("Time [s]")
    plt.ylabel("Wind load [N]")  # fixed label consistency
    plt.title("Wind loads vs time (45° heading)")

    plt.grid(True)
    plt.legend()

    plt.xlim(t_vals[0], t_vals[-1])  # ensures no edge clipping

    plt.savefig("wind_vs_time.pdf", bbox_inches="tight")
    plt.show()


def plot_wind_forces_vs_time(model, T=50.0, dt=0.05):
    """
    Wind forces over time at a fixed vessel heading.
    Matches style of heading plot.
    """

    # Fixed vessel state (45 degrees heading)
    psi = np.pi / 4
    eta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, psi], dtype=float)
    nu = np.zeros(6)

    t_vals = np.arange(0, T, dt)

    X_vals, Y_vals, N_vals = [], [], []

    model.reset()

    for t in t_vals:
        tau = model.get_tau_wind(t, eta, nu)

        X_vals.append(tau[0])
        Y_vals.append(tau[1])
        N_vals.append(tau[5])

    # --- Plot (matching style) ---
    plt.figure(figsize=(10, 6))

    plt.plot(t_vals, X_vals, label="Surge force X")
    plt.plot(t_vals, Y_vals, label="Sway force Y")
    plt.plot(t_vals, N_vals, label="Yaw moment N")

    plt.xlabel("Time [s]")
    plt.ylabel("Wind load [N]")
    plt.title("Wind loads vs time (45° heading)")

    plt.grid(True)
    plt.legend()

    plt.xlim(t_vals[0], t_vals[-1])  # ensures no clipping

    plt.savefig("wind_vs_time.pdf", bbox_inches="tight")
    plt.show()


def plot_wind_response_vs_heading(model, T=40.0, dt=0.05):
    headings = np.linspace(0, 2*np.pi, 181)

    rms_X = []
    rms_Y = []
    rms_N = []

    nu = np.zeros(6)
    t_vals = np.arange(0, T, dt)

    for psi in headings:
        eta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, psi], dtype=float)

        X_hist = []
        Y_hist = []
        N_hist = []

        model.reset()

        for _ in t_vals:
            tau = model.get_tau_wind(dt, eta, nu)  # keep your original behavior
            X_hist.append(tau[0])
            Y_hist.append(tau[1])
            N_hist.append(tau[5])

        X_hist = np.asarray(X_hist)
        Y_hist = np.asarray(Y_hist)
        N_hist = np.asarray(N_hist)

        rms_X.append(np.sqrt(np.mean(X_hist**2)))
        rms_Y.append(np.sqrt(np.mean(Y_hist**2)))
        rms_N.append(np.sqrt(np.mean(N_hist**2)))

    headings_deg = np.rad2deg(headings)

    fig, ax = plt.subplots(figsize=(10, 6))  # same figure size as first plot
    ax.plot(headings_deg, rms_X, label="RMS surge force X")
    ax.plot(headings_deg, rms_Y, label="RMS sway force Y")
    ax.plot(headings_deg, rms_N, label="RMS yaw moment N")

    ax.set_xlabel("Vessel heading [deg]")
    ax.set_ylabel("RMS wind load [N]")
    ax.set_title("Wind loads vs heading")
    ax.grid(True)
    ax.legend()
    ax.set_xlim(0, 360)

    fig.tight_layout()  # consistent layout
    fig.savefig("wind_vs_heading_rms.pdf")  # no bbox_inches='tight'
    plt.show()

def plot_wind_force_vs_heading(model, t_eval=0.0):
    headings = np.linspace(0, 2*np.pi, 361)
    nu = np.zeros(6)

    X_vals = []
    Y_vals = []
    N_vals = []

    for psi in headings:
        eta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, psi], dtype=float)

        tau = model.get_tau_wind(t_eval, eta, nu)

        X_vals.append(tau[0])
        Y_vals.append(tau[1])
        N_vals.append(tau[5])

    headings_deg = np.rad2deg(headings)

    plt.figure(figsize=(10,6))
    plt.plot(headings_deg, X_vals, label="Surge force X")
    plt.plot(headings_deg, Y_vals, label="Sway force Y")
    plt.plot(headings_deg, N_vals, label="Yaw moment N")

    plt.xlabel("Vessel heading [deg]")
    plt.ylabel("Wind load [N]")
    plt.title(f"Wind loads vs heading")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 360)
    plt.savefig("wind_vs_heading.pdf", bbox_inches="tight")
    plt.show()

wind = WindModel(
    mean_speed=5.0,
    mean_dir=0.0,
    gust_std=0.5,
    gust_time_constant=5.0,
    seed=1
)

plot_wind_forces(wind, T=20)
#plot_wind_forces_vs_time(wind, T=20)
#plot_wind_response_vs_heading(wind, T=20)
#plot_wind_forces_for_headings(wind)
#plot_wind_force_vs_heading(wind)