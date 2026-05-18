import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
UOWC optical propagation plots and optical communication scoring.
Based on plots @Ivar Saksvik.

    1. Plot 2D/3D UOWC propagation fields
    2. Compute optical received power score from simulation
    3. Compute optical received power score from experiment
    4. Compute experiment tracking/control metrics directly from the CSV log

    received_energy_J       = integral of received power over time [J]
    mean_P_rec_W_time       = time-average received power [W]
    mean_P_rec_dBm_time     = time-average received power converted to dBm
    integrated_H_s          = integral of optical channel transfer over time
    mean_H_time             = time-average optical channel transfer

tracking/control metrics
    IAE_distance            = integral of distance-to-target over time [m s]
    avg_distance_tracking   = time-average distance-to-target [m]
    IAU                     = integral of absolute actuator/control use over time

    dBm is logarithmic, so average received power is computed in watts first:
        mean_P_rec_W_time = integral(P_rec_W dt) / duration
        mean_P_rec_dBm_time = 10 log10(mean_P_rec_W_time / 1e-3)
"""



# USER SETTINGS

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent

LABEL = "DRL"

# -------------------------------------------------------------------------
# Simulation CSV
# Expected simulation columns:
#     simTime
#     simData_0, simData_1, ...
#     targetData_0, targetData_1, ...
# -------------------------------------------------------------------------
SIM_CSV_PATH = (
    _PROJECT_ROOT
    / "logs"
    / "sim_logs"
    / "drl_sim_straight_nodist"
    / "drl_straight_nodist.csv"
)

# Experiment CSV
# Expected experiment columns:
#     current_time or cycle_time
#     north_from_observer
#     east_from_observer
#     target_north_from_observer

EXP_CSV_PATH = (
    _PROJECT_ROOT
    / "logs"
    / "experiment_logs"
    / "PID_stationary_fin.csv"
)

OUTPUT_DIR = (
    _PROJECT_ROOT
    / "logs"
    / "optical_scores"
)

# What to run
COMPUTE_SIM_SCORE = False
COMPUTE_EXPERIMENT_SCORE = True

MAKE_2D_PROPAGATION_PLOTS = False
MAKE_3D_PROPAGATION_PLOTS = False
SHOW_PLOTS = True

# Experiment time window. Use None for full range.
EXP_START_TIME_S = 1.0
EXP_END_TIME_S = 150.0

# Optical parameters
theta_deg = 90.0      # semi-angle at half power [deg]
P_total = 20.0        # transmitted optical power per LED [W]
Adet = 1e-4           # detector physical area [m^2]
Ts = 1.0
index = 1.5
FOV_deg = 100.0

# Vertical separation / target depth
height_score = 5.0

# Receiver sensitivity threshold for outage calculation
power_threshold_dBm = -40.0

# Propagation plot settings
heights_for_plots = [1, 5, 10, 30]
lx, ly = 20, 20
Nx, Ny = lx * 30, ly * 30
XT, YT = 0.0, 0.0

# Tracking metric filters
MAX_REASONABLE_DISTANCE_M = 500.0


# =============================================================================
# HELPERS
# =============================================================================

def safe_label(text):
    return str(text).replace(" ", "_").replace("/", "_").replace("\\", "_")


def clean_simtime(series):
    return (
        series.astype(str)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .astype(float)
        .to_numpy()
    )


def sorted_prefixed_columns(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    return sorted(cols, key=lambda s: int(s.split("_")[1]))


def lambertian_order(theta_deg):
    theta_eff = min(theta_deg, 89.999)
    return -np.log(2) / np.log(np.cos(np.deg2rad(theta_eff)))


def watts_to_dBm(power_w):
    power_w = np.asarray(power_w, dtype=float)

    out = np.full_like(power_w, -np.inf, dtype=float)
    positive = power_w > 0.0
    out[positive] = 10.0 * np.log10(power_w[positive] / 1e-3)

    return out


def scalar_watts_to_dBm(power_w):
    if power_w > 0.0:
        return 10.0 * np.log10(power_w / 1e-3)
    return -np.inf


def clean_distance_signal(t, distance, max_reasonable_distance=MAX_REASONABLE_DISTANCE_M):
    """
    Removes impossible distance spikes, e.g. startup values around 1e6.
    Returns filtered time and distance arrays with time reset to zero.
    """
    t = np.asarray(t, dtype=float)
    distance = np.asarray(distance, dtype=float)

    valid = (
        np.isfinite(t)
        & np.isfinite(distance)
        & (distance >= 0.0)
        & (distance < max_reasonable_distance)
    )

    t_clean = t[valid]
    distance_clean = distance[valid]

    if len(t_clean) >= 2:
        t_clean = t_clean - t_clean[0]

    return t_clean, distance_clean


def _get_first_existing_column(df, candidate_names):
    for name in candidate_names:
        if name in df.columns:
            values = df[name].to_numpy(dtype=float)
            if np.isfinite(values).any():
                return values, name
    return None, None


def compute_tracking_metrics_from_experiment_df(df):
    """
    Computes tracking/control metrics directly from experiment CSV data.

    IAE_distance:
        Integral of distance-to-target over time [m s].

    avg_distance:
        Time-average distance-to-target [m].

    IAU:
        Integral of absolute actuator/control use over time.

        Preferred formula if tau_X and tau_N exist:
            IAU = integral(|tau_X| + |tau_N|) dt

        If only controller_X_cmd/controller_N_cmd exist, those columns are used.
        If the commands appear normalized in [-1, 1], they are scaled using
        tau_X = 150 N and tau_N = 116 Nm before integration.
    """
    t = df["t"].to_numpy(dtype=float)

    if "distance_to_target" in df.columns:
        distance = df["distance_to_target"].to_numpy(dtype=float)
    else:
        n_err = df["target_north_from_observer"] - df["north_from_observer"]
        e_err = df["target_east_from_observer"] - df["east_from_observer"]
        distance = np.hypot(n_err, e_err)

    t_dist, distance_clean = clean_distance_signal(
        t,
        distance,
        max_reasonable_distance=MAX_REASONABLE_DISTANCE_M,
    )

    if len(t_dist) >= 2:
        duration = max(t_dist[-1] - t_dist[0], 1e-9)
        IAE_distance = float(np.trapezoid(np.abs(distance_clean), t_dist))
        avg_distance = float(np.trapezoid(distance_clean, t_dist) / duration)
    else:
        IAE_distance = np.nan
        avg_distance = np.nan

    # IAU from actuator/control columns
    tau_X, tau_X_col = _get_first_existing_column(
        df,
        [
            "tau_X",
            "applied_tau_X",
            "controller_X_cmd",
            "surge_force_command",
            "X_force",
        ],
    )
    tau_N, tau_N_col = _get_first_existing_column(
        df,
        [
            "tau_N",
            "applied_tau_N",
            "controller_N_cmd",
            "yaw_moment_command",
            "N_moment",
        ],
    )

    IAU = np.nan
    IAU_source = ""

    if tau_X is not None and tau_N is not None:
        tau_X = np.asarray(tau_X, dtype=float)
        tau_N = np.asarray(tau_N, dtype=float)

        # If controller commands are normalized, scale to physical units.
        if tau_X_col == "controller_X_cmd" and np.nanmax(np.abs(tau_X)) <= 1.5:
            tau_X = tau_X * 150.0

        if tau_N_col == "controller_N_cmd" and np.nanmax(np.abs(tau_N)) <= 1.5:
            tau_N = tau_N * 116.0

        valid_u = (
            np.isfinite(t)
            & np.isfinite(tau_X)
            & np.isfinite(tau_N)
        )

        t_u = t[valid_u]
        tau_X_u = tau_X[valid_u]
        tau_N_u = tau_N[valid_u]

        if len(t_u) >= 2:
            t_u = t_u - t_u[0]
            IAU = float(
                np.trapezoid(
                    np.abs(tau_X_u) + np.abs(tau_N_u),
                    t_u,
                )
            )
            IAU_source = f"{tau_X_col}, {tau_N_col}"

    return {
        "IAE_distance": IAE_distance,
        "avg_distance": avg_distance,
        "IAU": IAU,
        "IAU_source": IAU_source,
    }


# =============================================================================
# CORE OPTICAL MODEL
# =============================================================================

def optical_transfer_from_distance(
    horizontal_distance,
    *,
    height=5.0,
    theta_deg=90.0,
    P_total=20.0,
    Adet=1e-4,
    Ts=1.0,
    index=1.5,
    FOV_deg=100.0,
):
    """
    Computes optical channel transfer and received power from horizontal distance.
    """
    horizontal_distance = np.asarray(horizontal_distance, dtype=float)

    D = np.sqrt(horizontal_distance**2 + height**2)

    m = lambertian_order(theta_deg)

    FOV = np.deg2rad(FOV_deg)
    G_Con = (index**2) / (np.sin(FOV)**2)

    cos_phi = height / D
    cos_psi = height / D

    psi = np.arccos(np.clip(cos_psi, -1.0, 1.0))
    inside_fov = psi <= FOV

    H = (
        ((m + 1.0) * Adet)
        / (2.0 * np.pi * D**2)
    ) * (cos_phi**m) * cos_psi * Ts * G_Con

    H[~inside_fov] = 0.0

    P_rec_W = P_total * H
    P_rec_dBm = watts_to_dBm(P_rec_W)

    return H, P_rec_W, P_rec_dBm, D, inside_fov


def summarize_optical_timeseries(
    t,
    horizontal_distance,
    H,
    P_rec_W,
    P_rec_dBm,
    inside_fov,
    *,
    label="simulation",
    source="simulation",
    csv_path=None,
    height=5.0,
    theta_deg=90.0,
    FOV_deg=100.0,
    P_total=20.0,
    power_threshold_dBm=-40.0,
    start_time_s=None,
    end_time_s=None,
):
    """
    Summarizes optical communication quality over one simulation/experiment.
    """
    t = np.asarray(t, dtype=float)
    horizontal_distance = np.asarray(horizontal_distance, dtype=float)
    H = np.asarray(H, dtype=float)
    P_rec_W = np.asarray(P_rec_W, dtype=float)
    P_rec_dBm = np.asarray(P_rec_dBm, dtype=float)
    inside_fov = np.asarray(inside_fov, dtype=bool)

    if len(t) < 2:
        raise ValueError("Need at least two samples for optical summary.")

    duration_s = t[-1] - t[0]

    if duration_s <= 0:
        raise ValueError("Time vector must have positive duration.")

    integrated_H_s = np.trapezoid(H, t)
    received_energy_J = np.trapezoid(P_rec_W, t)

    mean_H_time = integrated_H_s / duration_s
    mean_P_rec_W_time = received_energy_J / duration_s
    mean_P_rec_dBm_time = scalar_watts_to_dBm(mean_P_rec_W_time)

    finite_dBm = np.isfinite(P_rec_dBm)
    outage_mask = P_rec_dBm < power_threshold_dBm
    outage_percent = 100.0 * np.mean(outage_mask)

    summary_df = pd.DataFrame([{
        "label": label,
        "source": source,
        "csv_path": str(csv_path) if csv_path is not None else "",
        "start_time_s": start_time_s,
        "end_time_s": end_time_s,
        "duration_s": duration_s,

        "height_m": height,
        "theta_deg": theta_deg,
        "FOV_deg": FOV_deg,
        "P_total_W": P_total,

        "received_energy_J": received_energy_J,
        "mean_P_rec_W_time": mean_P_rec_W_time,
        "mean_P_rec_dBm_time": mean_P_rec_dBm_time,

        "integrated_H_s": integrated_H_s,
        "mean_H_time": mean_H_time,

        "max_P_rec_W": np.max(P_rec_W),
        "min_P_rec_W": np.min(P_rec_W),
        "max_P_rec_dBm": np.max(P_rec_dBm[finite_dBm]) if np.any(finite_dBm) else -np.inf,
        "min_P_rec_dBm": np.min(P_rec_dBm[finite_dBm]) if np.any(finite_dBm) else -np.inf,

        "max_H": np.max(H),
        "min_H": np.min(H),

        "power_threshold_dBm": power_threshold_dBm,
        "outage_percent": outage_percent,
        "inside_fov_percent": 100.0 * np.mean(inside_fov),

        "mean_horizontal_distance_m": np.mean(horizontal_distance),
        "rms_horizontal_distance_m": np.sqrt(np.mean(horizontal_distance**2)),
        "min_horizontal_distance_m": np.min(horizontal_distance),
        "max_horizontal_distance_m": np.max(horizontal_distance),
        "final_horizontal_distance_m": horizontal_distance[-1],
    }])

    return summary_df


# =============================================================================
# SIMULATION CSV LOADING AND SCORING
# =============================================================================

def load_simulation_log(csv_path):
    """
    Loads simulation CSV.

    Expected columns:
        simTime
        simData_0, simData_1, ...
        targetData_0, targetData_1, ...
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find simulation CSV file: {csv_path}")

    df = pd.read_csv(csv_path)

    if "simTime" not in df.columns:
        raise ValueError("Simulation CSV must contain a 'simTime' column.")

    sim_cols = sorted_prefixed_columns(df, "simData_")
    target_cols = sorted_prefixed_columns(df, "targetData_")

    if len(sim_cols) < 2:
        raise ValueError("Simulation CSV must contain at least simData_0 and simData_1.")

    if len(target_cols) < 2:
        raise ValueError("Simulation CSV must contain at least targetData_0 and targetData_1.")

    simTime = clean_simtime(df["simTime"])
    simData = df[sim_cols].to_numpy(dtype=float)
    targetData = df[target_cols].to_numpy(dtype=float)

    n = min(len(simTime), len(simData), len(targetData))

    return simTime[:n], simData[:n], targetData[:n]


def compute_optical_score_from_sim_arrays(
    simTime,
    simData,
    targetData,
    *,
    height=5.0,
    theta_deg=90.0,
    P_total=20.0,
    Adet=1e-4,
    Ts=1.0,
    index=1.5,
    FOV_deg=100.0,
    power_threshold_dBm=-40.0,
    label="simulation",
    source="simulation",
    csv_path=None,
):
    """
    Computes optical received-power score from already loaded simulation arrays.
    """
    simTime = np.asarray(simTime, dtype=float).ravel()
    simData = np.asarray(simData, dtype=float)
    targetData = np.asarray(targetData, dtype=float)

    n = min(len(simTime), len(simData), len(targetData))

    if n < 2:
        raise ValueError("Need at least two samples to compute optical transfer.")

    t = simTime[:n]

    usv_n = simData[:n, 0]
    usv_e = simData[:n, 1]

    tar_n = targetData[:n, 0]
    tar_e = targetData[:n, 1]

    horizontal_distance = np.sqrt((tar_n - usv_n)**2 + (tar_e - usv_e)**2)

    H, P_rec_W, P_rec_dBm, D, inside_fov = optical_transfer_from_distance(
        horizontal_distance,
        height=height,
        theta_deg=theta_deg,
        P_total=P_total,
        Adet=Adet,
        Ts=Ts,
        index=index,
        FOV_deg=FOV_deg,
    )

    optical_df = pd.DataFrame({
        "simTime": t,
        "usv_north_m": usv_n,
        "usv_east_m": usv_e,
        "target_north_m": tar_n,
        "target_east_m": tar_e,
        "horizontal_distance_m": horizontal_distance,
        "height_m": height,
        "distance_3d_m": D,
        "inside_fov": inside_fov,
        "channel_transfer_H": H,
        "P_rec_W": P_rec_W,
        "P_rec_dBm": P_rec_dBm,
    })

    summary_df = summarize_optical_timeseries(
        t,
        horizontal_distance,
        H,
        P_rec_W,
        P_rec_dBm,
        inside_fov,
        label=label,
        source=source,
        csv_path=csv_path,
        height=height,
        theta_deg=theta_deg,
        FOV_deg=FOV_deg,
        P_total=P_total,
        power_threshold_dBm=power_threshold_dBm,
    )

    return optical_df, summary_df


def compute_score_from_simulation_csv(
    csv_path,
    *,
    output_dir=None,
    label="simulation",
    height=5.0,
    theta_deg=90.0,
    P_total=20.0,
    Adet=1e-4,
    Ts=1.0,
    index=1.5,
    FOV_deg=100.0,
    power_threshold_dBm=-40.0,
    save_csv=True,
):
    """
    Loads simulation CSV and computes optical received-power score.
    """
    csv_path = Path(csv_path)

    simTime, simData, targetData = load_simulation_log(csv_path)

    optical_df, summary_df = compute_optical_score_from_sim_arrays(
        simTime,
        simData,
        targetData,
        height=height,
        theta_deg=theta_deg,
        P_total=P_total,
        Adet=Adet,
        Ts=Ts,
        index=index,
        FOV_deg=FOV_deg,
        power_threshold_dBm=power_threshold_dBm,
        label=label,
        source="simulation",
        csv_path=csv_path,
    )

    if save_csv:
        save_optical_outputs(
            optical_df,
            summary_df,
            output_dir=output_dir,
            label=label,
            prefix="simulation",
            fallback_dir=csv_path.parent,
        )

    return optical_df, summary_df



# EXPERIMENT CSV LOADING AND SCORING


def load_experiment_log_for_optical_score(
    csv_path,
    *,
    start_time_s=None,
    end_time_s=None,
):
    """
    Loads experiment CSV logs for optical transfer scoring.

    Expected experiment columns:
        north_from_observer
        east_from_observer
        target_north_from_observer
        target_east_from_observer
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find experiment CSV file: {csv_path}")

    df = pd.read_csv(csv_path, sep=";")

    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "timestamp_string"})

    for col in df.columns:
        if col != "timestamp_string":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "current_time" in df.columns and df["current_time"].notna().any():
        t = df["current_time"].to_numpy(dtype=float)
        finite_t = t[np.isfinite(t)]

        if len(finite_t) == 0:
            raise ValueError("current_time exists but contains no finite values.")

        t = t - finite_t[0]

    elif "cycle_time" in df.columns and df["cycle_time"].notna().any():
        t = np.cumsum(df["cycle_time"].fillna(0.0).to_numpy(dtype=float))

    else:
        t = np.arange(len(df), dtype=float)

    df["t_raw"] = t

    mask = np.isfinite(df["t_raw"].to_numpy(dtype=float))

    if start_time_s is not None:
        mask &= df["t_raw"].to_numpy(dtype=float) >= float(start_time_s)

    if end_time_s is not None:
        mask &= df["t_raw"].to_numpy(dtype=float) <= float(end_time_s)

    df = df.loc[mask].copy()

    if len(df) < 2:
        raise ValueError("Selected experiment time window contains fewer than two valid samples.")

    df["t"] = df["t_raw"] - df["t_raw"].iloc[0]

    required_cols = [
        "north_from_observer",
        "east_from_observer",
        "target_north_from_observer",
        "target_east_from_observer",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required experiment columns: {missing}")

    df = df.dropna(subset=required_cols).copy()

    if len(df) < 2:
        raise ValueError("Experiment log has fewer than two valid position samples after dropping NaNs.")

    return df.reset_index(drop=True)


def compute_score_from_experiment_csv(
    csv_path,
    *,
    output_dir=None,
    label="experiment",
    start_time_s=None,
    end_time_s=None,
    height=5.0,
    theta_deg=90.0,
    P_total=20.0,
    Adet=1e-4,
    Ts=1.0,
    index=1.5,
    FOV_deg=100.0,
    power_threshold_dBm=-40.0,
    save_csv=True,
):
    """
    Computes optical received-power score and tracking/control metrics directly from an experiment CSV log.
    """
    csv_path = Path(csv_path)

    df = load_experiment_log_for_optical_score(
        csv_path,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
    )

    t = df["t"].to_numpy(dtype=float)

    usv_n = df["north_from_observer"].to_numpy(dtype=float)
    usv_e = df["east_from_observer"].to_numpy(dtype=float)

    tar_n = df["target_north_from_observer"].to_numpy(dtype=float)
    tar_e = df["target_east_from_observer"].to_numpy(dtype=float)

    horizontal_distance = np.sqrt((tar_n - usv_n)**2 + (tar_e - usv_e)**2)

    # Filter impossible distance spikes for optical score as well.
    t_opt, horizontal_distance_opt = clean_distance_signal(
        t,
        horizontal_distance,
        max_reasonable_distance=MAX_REASONABLE_DISTANCE_M,
    )

    if len(t_opt) < 2:
        raise ValueError("Not enough valid distance samples after filtering impossible spikes.")

    H, P_rec_W, P_rec_dBm, D, inside_fov = optical_transfer_from_distance(
        horizontal_distance_opt,
        height=height,
        theta_deg=theta_deg,
        P_total=P_total,
        Adet=Adet,
        Ts=Ts,
        index=index,
        FOV_deg=FOV_deg,
    )

    optical_df = pd.DataFrame({
        "simTime": t_opt,
        "experiment_time_s": t_opt,
        "horizontal_distance_m": horizontal_distance_opt,
        "height_m": height,
        "distance_3d_m": D,
        "inside_fov": inside_fov,
        "channel_transfer_H": H,
        "P_rec_W": P_rec_W,
        "P_rec_dBm": P_rec_dBm,
    })

    summary_df = summarize_optical_timeseries(
        t_opt,
        horizontal_distance_opt,
        H,
        P_rec_W,
        P_rec_dBm,
        inside_fov,
        label=label,
        source="experiment",
        csv_path=csv_path,
        height=height,
        theta_deg=theta_deg,
        FOV_deg=FOV_deg,
        P_total=P_total,
        power_threshold_dBm=power_threshold_dBm,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
    )

    tracking_metrics = compute_tracking_metrics_from_experiment_df(df)

    summary_df["IAE_distance"] = tracking_metrics["IAE_distance"]
    summary_df["avg_distance_tracking"] = tracking_metrics["avg_distance"]
    summary_df["IAU"] = tracking_metrics["IAU"]
    summary_df["IAU_source"] = tracking_metrics["IAU_source"]

    if save_csv:
        save_optical_outputs(
            optical_df,
            summary_df,
            output_dir=output_dir,
            label=label,
            prefix="experiment",
            fallback_dir=csv_path.parent,
        )

    return optical_df, summary_df


# =============================================================================
# OUTPUT SAVING
# =============================================================================

def save_optical_outputs(
    optical_df,
    summary_df,
    *,
    output_dir=None,
    label="simulation",
    prefix="simulation",
    fallback_dir=None,
):
    """
    Saves optical time series and summary CSV files.
    """
    if output_dir is None:
        if fallback_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(fallback_dir)
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    timeseries_path = output_dir / f"{safe_label(label)}_{prefix}_received_power_timeseries.csv"
    summary_path = output_dir / f"{safe_label(label)}_{prefix}_received_power_summary.csv"

    optical_df.to_csv(timeseries_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved optical time series to: {timeseries_path}")
    print(f"Saved optical summary to: {summary_path}")


# =============================================================================
# PROPAGATION PLOTS
# =============================================================================

def compute_propagation_grid(
    *,
    height,
    theta_deg=90.0,
    P_total=20.0,
    Adet=1e-4,
    Ts=1.0,
    index=1.5,
    FOV_deg=100.0,
    lx=20,
    ly=20,
    Nx=600,
    Ny=600,
    XT=0.0,
    YT=0.0,
):
    """
    Computes received power over a 2D grid for one vertical separation.
    """
    x = np.linspace(-lx / 2.0, lx / 2.0, Nx)
    y = np.linspace(-ly / 2.0, ly / 2.0, Ny)
    XR, YR = np.meshgrid(x, y)

    horizontal_distance = np.sqrt((XR - XT)**2 + (YR - YT)**2)

    _, P_rec_W, _, _, inside_fov = optical_transfer_from_distance(
        horizontal_distance,
        height=height,
        theta_deg=theta_deg,
        P_total=P_total,
        Adet=Adet,
        Ts=Ts,
        index=index,
        FOV_deg=FOV_deg,
    )

    P_rec_W = np.maximum(P_rec_W, 1e-15)
    P_rec_dBm = 10.0 * np.log10(P_rec_W / 1e-3)
    P_rec_dBm[~inside_fov] = 10.0 * np.log10(1e-15 / 1e-3)

    return XR, YR, P_rec_dBm


def plot_2d_propagation(
    *,
    heights=(1, 5, 10, 30),
    theta_deg=90.0,
    P_total=20.0,
    Adet=1e-4,
    Ts=1.0,
    index=1.5,
    FOV_deg=100.0,
    lx=20,
    ly=20,
    Nx=600,
    Ny=600,
    XT=0.0,
    YT=0.0,
    show=True,
):
    """
    Makes the 2D contour propagation plots.
    """
    fig2d, axes2d = plt.subplots(2, 2, figsize=(12, 10))

    for i, h in enumerate(heights):
        XR, YR, P_rec_dBm = compute_propagation_grid(
            height=h,
            theta_deg=theta_deg,
            P_total=P_total,
            Adet=Adet,
            Ts=Ts,
            index=index,
            FOV_deg=FOV_deg,
            lx=lx,
            ly=ly,
            Nx=Nx,
            Ny=Ny,
            XT=XT,
            YT=YT,
        )

        ax = axes2d.flat[i]
        contour = ax.contourf(XR, YR, P_rec_dBm, levels=100, cmap="viridis")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"2D received power, h = {h} m")
        ax.set_xlim([-lx / 2.0, lx / 2.0])
        ax.set_ylim([-ly / 2.0, ly / 2.0])
        ax.set_aspect("equal")

        fig2d.colorbar(contour, ax=ax, label="Received power (dBm)")

    plt.tight_layout()

    if show:
        plt.show()

    return fig2d


def plot_3d_propagation(
    *,
    heights=(1, 5, 10, 30),
    theta_deg=90.0,
    P_total=20.0,
    Adet=1e-4,
    Ts=1.0,
    index=1.5,
    FOV_deg=100.0,
    lx=20,
    ly=20,
    Nx=600,
    Ny=600,
    XT=0.0,
    YT=0.0,
    show=True,
):
    """
    Makes the 3D surface propagation plots.
    """
    fig3d = plt.figure(figsize=(14, 11))

    for i, h in enumerate(heights):
        XR, YR, P_rec_dBm = compute_propagation_grid(
            height=h,
            theta_deg=theta_deg,
            P_total=P_total,
            Adet=Adet,
            Ts=Ts,
            index=index,
            FOV_deg=FOV_deg,
            lx=lx,
            ly=ly,
            Nx=Nx,
            Ny=Ny,
            XT=XT,
            YT=YT,
        )

        ax = fig3d.add_subplot(2, 2, i + 1, projection="3d")

        surf = ax.plot_surface(
            XR,
            YR,
            P_rec_dBm,
            cmap="viridis",
            linewidth=0,
            antialiased=True,
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Received power (dBm)")
        ax.set_title(f"3D received power, h = {h} m")

        ax.set_xlim([-lx / 2.0, lx / 2.0])
        ax.set_ylim([-ly / 2.0, ly / 2.0])

        fig3d.colorbar(surf, ax=ax, shrink=0.6, label="Received power (dBm)")

    plt.tight_layout()

    if show:
        plt.show()

    return fig3d


# =============================================================================
# COMPARISON HELPER
# =============================================================================

def compare_optical_summaries(summary_list, output_path=None):
    """
    Combines several one-row summary DataFrames into one comparison table.
    """
    comparison_df = pd.concat(summary_list, ignore_index=True)

    columns_first = [
        "label",
        "source",
        "duration_s",
        "IAE_distance",
        "avg_distance_tracking",
        "IAU",
        "IAU_source",
        "received_energy_J",
        "mean_P_rec_W_time",
        "mean_P_rec_dBm_time",
        "integrated_H_s",
        "mean_H_time",
        "min_P_rec_dBm",
        "max_P_rec_dBm",
        "outage_percent",
        "rms_horizontal_distance_m",
        "max_horizontal_distance_m",
        "inside_fov_percent",
    ]

    columns_first = [c for c in columns_first if c in comparison_df.columns]
    remaining = [c for c in comparison_df.columns if c not in columns_first]
    comparison_df = comparison_df[columns_first + remaining]

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_path, index=False)
        print(f"Saved optical comparison to: {output_path}")

    return comparison_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    m = lambertian_order(theta_deg)
    print(f"Lambertian order m = {m:.4f}")
    print(f"Output directory: {OUTPUT_DIR}")

    summaries = []

    if COMPUTE_SIM_SCORE:
        print("\nComputing optical received-power score from simulation CSV:")
        print(SIM_CSV_PATH)

        sim_optical_df, sim_summary_df = compute_score_from_simulation_csv(
            SIM_CSV_PATH,
            output_dir=OUTPUT_DIR,
            label=LABEL,
            height=height_score,
            theta_deg=theta_deg,
            P_total=P_total,
            Adet=Adet,
            Ts=Ts,
            index=index,
            FOV_deg=FOV_deg,
            power_threshold_dBm=power_threshold_dBm,
            save_csv=True,
        )

        summaries.append(sim_summary_df)

        print("\nSimulation optical summary:")
        print(sim_summary_df.T)

        e_rec = sim_summary_df.loc[0, "received_energy_J"]
        e_rec_df = pd.DataFrame([{
            "label": LABEL,
            "E_rec_J": e_rec,
        }])

        e_rec_path = OUTPUT_DIR / f"{safe_label(LABEL)}_E_rec.csv"
        e_rec_df.to_csv(e_rec_path, index=False)

        print(f"\nE_rec = {e_rec:.6e} J")
        print(f"Saved E_rec to: {e_rec_path}")

    if COMPUTE_EXPERIMENT_SCORE:
        print("\nComputing optical received-power score from experiment CSV:")
        print(EXP_CSV_PATH)

        exp_optical_df, exp_summary_df = compute_score_from_experiment_csv(
            EXP_CSV_PATH,
            output_dir=OUTPUT_DIR,
            label=f"{LABEL}_experiment",
            start_time_s=EXP_START_TIME_S,
            end_time_s=EXP_END_TIME_S,
            height=height_score,
            theta_deg=theta_deg,
            P_total=P_total,
            Adet=Adet,
            Ts=Ts,
            index=index,
            FOV_deg=FOV_deg,
            power_threshold_dBm=power_threshold_dBm,
            save_csv=True,
        )

        summaries.append(exp_summary_df)

        print("\nExperiment optical summary:")
        print(exp_summary_df.T)

        print("\nExperiment tracking/control metrics from CSV:")
        print(f"IAE distance:     {exp_summary_df.loc[0, 'IAE_distance']:.6f} m s")
        print(f"Average distance: {exp_summary_df.loc[0, 'avg_distance_tracking']:.6f} m")
        print(f"IAU:              {exp_summary_df.loc[0, 'IAU']:.6f}")
        print(f"IAU source:       {exp_summary_df.loc[0, 'IAU_source']}")

    if len(summaries) > 1:
        comparison_path = OUTPUT_DIR / f"{safe_label(LABEL)}_optical_comparison.csv"

        comparison_df = compare_optical_summaries(
            summaries,
            output_path=comparison_path,
        )

        print("\nOptical comparison:")
        print(comparison_df)

    if MAKE_2D_PROPAGATION_PLOTS:
        plot_2d_propagation(
            heights=heights_for_plots,
            theta_deg=theta_deg,
            P_total=P_total,
            Adet=Adet,
            Ts=Ts,
            index=index,
            FOV_deg=FOV_deg,
            lx=lx,
            ly=ly,
            Nx=Nx,
            Ny=Ny,
            XT=XT,
            YT=YT,
            show=SHOW_PLOTS,
        )

    if MAKE_3D_PROPAGATION_PLOTS:
        plot_3d_propagation(
            heights=heights_for_plots,
            theta_deg=theta_deg,
            P_total=P_total,
            Adet=Adet,
            Ts=Ts,
            index=index,
            FOV_deg=FOV_deg,
            lx=lx,
            ly=ly,
            Nx=Nx,
            Ny=Ny,
            XT=XT,
            YT=YT,
            show=SHOW_PLOTS,
        )


if __name__ == "__main__":
    main()
