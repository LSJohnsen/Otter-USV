import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from wave_model1 import WaveModel

def plot_wave_forces(model, T=50.0, dt=0.05):
    eta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi])
    nu = np.zeros(6)

    t_vals = np.arange(0, T, dt)

    X_vals, Y_vals, N_vals = [], [], []

    for t in t_vals:
        tau = model.get_tau_wave(t, eta, nu)
        X_vals.append(tau[0])
        Y_vals.append(tau[1])
        N_vals.append(tau[5])

    plt.figure(figsize=(10,6))
    plt.plot(t_vals, X_vals, label="X")
    plt.plot(t_vals, Y_vals, label="Y")
    plt.plot(t_vals, N_vals, label="N")

    plt.xlabel("Time [s]")
    plt.ylabel("Wave load [N]")
    plt.title("Wave loads vs time")
    plt.grid(True)
    plt.legend()
    plt.xlim(t_vals[0], t_vals[-1])   # ensures full span
    plt.savefig("wave_vs_time.pdf", bbox_inches="tight")
    plt.show()


def plot_wave_response_vs_heading(model, T=40.0, dt=0.05):
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

        for t in t_vals:
            tau = model.get_tau_wave(t, eta, nu)
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

    plt.figure()
    plt.plot(headings_deg, rms_X, label="RMS surge force X")
    plt.plot(headings_deg, rms_Y, label="RMS sway force Y")
    plt.plot(headings_deg, rms_N, label="RMS yaw moment N")

    plt.xlabel("Vessel heading [deg]")
    plt.ylabel("RMS wave load")
    plt.title("Wave-induced loads vs vessel heading")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_wave_forces_for_headings(model, headings_deg=(0, 45, 90, 180), T=20.0, dt=0.05):
    t_vals = np.arange(0, T, dt)
    nu = np.zeros(6)

    for hdg_deg in headings_deg:
        psi = np.deg2rad(hdg_deg)
        eta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, psi], dtype=float)

        X_hist, Y_hist, N_hist = [], [], []

        for t in t_vals:
            tau = model.get_tau_wave(t, eta, nu)
            X_hist.append(tau[0])
            Y_hist.append(tau[1])
            N_hist.append(tau[5])

        plt.figure()
        plt.plot(t_vals, X_hist, label="X")
        plt.plot(t_vals, Y_hist, label="Y")
        plt.plot(t_vals, N_hist, label="N")
        plt.xlabel("Time [s]")
        plt.ylabel("Wave load [N]")
        plt.title(f"Wave loads at heading {hdg_deg} deg")
        plt.grid(True)
        plt.legend()
        plt.show()

def plot_wave_vs_heading(model, t_eval=0.0):
    headings_deg = np.linspace(0, 720, 721)
    nu = np.zeros(6)

    X_vals, Y_vals, N_vals = [], [], []

    for hdg_deg in headings_deg:
        psi = np.deg2rad(hdg_deg)
        eta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, psi], dtype=float)

        tau = model.get_tau_wave(t_eval, eta, nu)

        X_vals.append(tau[0])
        Y_vals.append(tau[1])
        N_vals.append(tau[5])

    plt.figure(figsize=(10, 6))
    plt.plot(headings_deg, X_vals, label="X")
    plt.plot(headings_deg, Y_vals, label="Y")
    plt.plot(headings_deg, N_vals, label="N")

    plt.xlabel("Heading [deg]")
    plt.ylabel("Wave load [N]")
    plt.title("Wave loads vs heading")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 720)
    plt.savefig("wave_vs_heading.pdf", bbox_inches="tight")
    plt.show()

wave = WaveModel(
    Hs=0.3,
    Tp=2.0,
    mean_dir=0.0,
    seed=1
)

plot_wave_forces(wave, T=20)
#plot_wave_response_vs_heading(wave, T=20)
#plot_wave_forces_for_headings(wave)
#plot_wave_vs_heading(wave)