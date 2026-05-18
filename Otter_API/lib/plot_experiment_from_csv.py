# -*- coding: utf-8 -*-


import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

# Use TkAgg when running locally with interactive plots.
# Comment this out if running headless on Linux.
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize



_THIS_DIR = Path(__file__).resolve().parent

# Change these to your experiment CSV and output folder.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent

CSV_PATH = _PROJECT_ROOT / "logs" / "experiment_logs" / "mpc_station_final_R0075.csv"
PLOT_DIR = _PROJECT_ROOT / "logs" / "experiment_plots" / "mpc_station"

LABEL = "NMPC"

# Select plotting interval in seconds from the CSV start.
# Use None to keep the full range.
START_TIME_S = 1.0
END_TIME_S = 150      

#frame:
#   "initial_body" -> x = starboard, y = forward based on initial USV heading
#   "ned"          -> x = east, y = north
#   "target_aligned" x = target path direction, y = cross-track, target starts at [10, 0]

FRAME_MODE = "initial_body"
#FRAME_MODE = "target_aligned"

SAVE_PLOTS = True
SHOW_PLOTS = True
SAVE_PDF = True
SAVE_EPS = False

PLOTS_TO_MAKE = ["all"]

# Figure style, copied from the simulation plotting style
legendSize = 9
titleSize = 14
labelSize = 12
tickSize = 10
lineWidth = 0.9
gridAlpha = 0.65

figSizeSingle = [18, 10]      # cm
figSizeMulti = [25, 15]       # cm
figSizePath = [18, 14]        # cm
dpiValue = 150

CONTROL_LABELS_TAU = [
    "Surge force command",
    "Yaw moment command",
]



# HELPERS

def transform_ne_to_target_aligned(
    usv_north,
    usv_east,
    target_north,
    target_east,
    yaw,
    desired_target_start_y=10.0,
):
    """
    Transform N/E trajectories to a normalized target-path plot frame.

    Output frame:
        Target initial position -> [0, 10]
        Target path             -> positive x-axis
        USV initial position    -> [0, 0]

    This is a comparison/visualization frame, not a rigid N/E rotation.
    """

    usv_north = np.asarray(usv_north, dtype=float)
    usv_east = np.asarray(usv_east, dtype=float)
    target_north = np.asarray(target_north, dtype=float)
    target_east = np.asarray(target_east, dtype=float)
    yaw = np.asarray(yaw, dtype=float)


    usv0 = np.array([usv_north[0], usv_east[0]], dtype=float)
    target0 = np.array([target_north[0], target_east[0]], dtype=float)



    d_target = np.array([
        target_north[-1] - target_north[0],
        target_east[-1] - target_east[0],
    ], dtype=float)

    if np.linalg.norm(d_target) < 1e-8:
        # Stationary fallback: use initial USV-to-target direction
        d_target = target0 - usv0

    if np.linalg.norm(d_target) < 1e-8:
        # Final fallback
        d_target = np.array([np.cos(yaw[0]), np.sin(yaw[0])], dtype=float)

    t_hat = d_target / np.linalg.norm(d_target)


    n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)


    initial_offset = target0 - usv0
    if np.dot(initial_offset, n_hat) < 0.0:
        n_hat = -n_hat



    target_disp = np.column_stack([
        target_north - target_north[0],
        target_east - target_east[0],
    ])

    target_progress = target_disp @ t_hat
    tar_x = target_progress
    tar_y = desired_target_start_y * np.ones_like(tar_x)


    usv_disp = np.column_stack([
        usv_north - usv_north[0],
        usv_east - usv_east[0],
    ])

    usv_x = usv_disp @ t_hat
    usv_y = usv_disp @ n_hat


    heading_vec = np.column_stack([
        np.cos(yaw),
        np.sin(yaw),
    ])

    heading_x = heading_vec @ t_hat
    heading_y = heading_vec @ n_hat

    yaw_plot = np.arctan2(heading_y, heading_x)

    return usv_x, usv_y, tar_x, tar_y, yaw_plot

def despike_signal(t, x, max_rate, window=5):
    """
    Removes single-sample spikes based on a maximum physically plausible rate of change.
    max_rate is in signal units per second.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float).copy()

    for i in range(1, len(x) - 1):
        dt_prev = max(t[i] - t[i - 1], 1e-6)
        dt_next = max(t[i + 1] - t[i], 1e-6)

        rate_prev = abs((x[i] - x[i - 1]) / dt_prev)
        rate_next = abs((x[i + 1] - x[i]) / dt_next)

        # Spike if it jumps up/down too fast and immediately returns
        if rate_prev > max_rate and rate_next > max_rate:
            x[i] = 0.5 * (x[i - 1] + x[i + 1])

    return pd.Series(x).rolling(
        window=window,
        center=True,
        min_periods=1
    ).mean().to_numpy()

def cm2inch(value):
    return value / 2.54


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def R2D(value):
    return value * 180.0 / math.pi


def safe_label(text):
    return str(text).replace(" ", "_").replace("/", "_").replace("\\", "_")


def apply_axis_style(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=labelSize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=labelSize)
    if title:
        ax.set_title(title, fontsize=titleSize)

    ax.tick_params(axis="both", which="major", labelsize=tickSize)
    ax.grid(True, alpha=gridAlpha)


def finalize_figure(fig, plot_name, rect=(0.03, 0.03, 0.98, 0.94)):
    fig.tight_layout(rect=rect)

    if SAVE_PLOTS:
        os.makedirs(PLOT_DIR, exist_ok=True)
        base = f"{safe_label(LABEL)}_{plot_name}"

        png_out = os.path.join(PLOT_DIR, base + ".png")
        fig.savefig(png_out, bbox_inches="tight", dpi=dpiValue, pad_inches=0.08)
        print(f"Saving plot to: {png_out}")

        if SAVE_PDF:
            pdf_out = os.path.join(PLOT_DIR, base + ".pdf")
            fig.savefig(pdf_out, format="pdf", bbox_inches="tight", pad_inches=0.08)
            print(f"Saving plot to: {pdf_out}")

        if SAVE_EPS:
            eps_out = os.path.join(PLOT_DIR, base + ".eps")
            fig.savefig(eps_out, format="eps", bbox_inches="tight", pad_inches=0.08)
            print(f"Saving plot to: {eps_out}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def choose_legend_location(x1, y1, x2, y2):
    xs = np.concatenate([np.asarray(x1), np.asarray(x2)])
    ys = np.concatenate([np.asarray(y1), np.asarray(y2)])

    xmin, xmax = np.nanmin(xs), np.nanmax(xs)
    ymin, ymax = np.nanmin(ys), np.nanmax(ys)

    xspan = max(xmax - xmin, 1e-9)
    yspan = max(ymax - ymin, 1e-9)

    xn = (xs - xmin) / xspan
    yn = (ys - ymin) / yspan

    boxes = {
        "upper left":  ((0.00, 0.38), (0.62, 1.00)),
        "upper right": ((0.38, 1.00), (0.62, 1.00)),
        "lower left":  ((0.00, 0.38), (0.00, 0.38)),
        "lower right": ((0.38, 1.00), (0.00, 0.38)),
    }

    scores = {}
    for loc, ((xa, xb), (ya, yb)) in boxes.items():
        inside = ((xn >= xa) & (xn <= xb) & (yn >= ya) & (yn <= yb))
        scores[loc] = np.count_nonzero(inside)

    preference = ["lower right", "upper right", "lower left", "upper left"]
    best_score = min(scores.values())
    candidates = [loc for loc, score in scores.items() if score == best_score]

    for loc in preference:
        if loc in candidates:
            return loc

    return "lower right"


# =============================================================================
# CSV LOADING AND BODY-FRAME TRANSFORM
# =============================================================================

def load_experiment_csv(csv_path, start_time_s=None, end_time_s=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV file: {csv_path}")

    df = pd.read_csv(csv_path, sep=";")

    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "timestamp_string"})

    for col in df.columns:
        if col != "timestamp_string":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "current_time" in df.columns and df["current_time"].notna().any():
        t = df["current_time"].to_numpy(dtype=float)
        t = t - t[np.isfinite(t)][0]
    elif "cycle_time" in df.columns:
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
        raise ValueError("Time window contains fewer than two valid samples.")

    # Reset plotted time to start at zero for the selected interval.
    df["t"] = df["t_raw"] - df["t_raw"].iloc[0]

    return df.reset_index(drop=True)


def get_yaw_rad(df):
    if "yaw_rad" in df.columns and df["yaw_rad"].notna().any():
        return wrap_to_pi(df["yaw_rad"].to_numpy(dtype=float))

    if "current_orientation_3" in df.columns and df["current_orientation_3"].notna().any():
        return wrap_to_pi(np.deg2rad(df["current_orientation_3"].to_numpy(dtype=float)))

    return np.zeros(len(df), dtype=float)


def transform_ne_to_initial_body(north, east, n0, e0, psi0):
    """
    Transform N/E coordinates to initial-body plot coordinates.

    Returns
    -------
    starboard, forward
        x-axis = starboard/right of initial USV heading
        y-axis = forward/ahead of initial USV heading
    """
    dn = np.asarray(north, dtype=float) - float(n0)
    de = np.asarray(east, dtype=float) - float(e0)

    forward = np.cos(psi0) * dn + np.sin(psi0) * de
    starboard = -np.sin(psi0) * dn + np.cos(psi0) * de

    return starboard, forward

def build_plot_arrays(df, frame_mode="initial_body"):
    required = [
        "north_from_observer",
        "east_from_observer",
        "target_north_from_observer",
        "target_east_from_observer",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required position columns: {missing}")

    data = df.dropna(subset=required).copy()
    if len(data) < 2:
        raise ValueError("Not enough valid position samples after dropping NaNs.")

    t = data["t"].to_numpy(dtype=float)

    usv_n = data["north_from_observer"].to_numpy(dtype=float)
    usv_e = data["east_from_observer"].to_numpy(dtype=float)

    tar_n = data["target_north_from_observer"].to_numpy(dtype=float)
    tar_e = data["target_east_from_observer"].to_numpy(dtype=float)

    yaw = get_yaw_rad(data)

    if frame_mode == "target_aligned":
        usv_x, usv_y, tar_x, tar_y, yaw_plot = transform_ne_to_target_aligned(
            usv_north=usv_n,
            usv_east=usv_e,
            target_north=tar_n,
            target_east=tar_e,
            yaw=yaw,
        )

        xlabel = "Along target path (m)"
        ylabel = "Cross-track position (m)"

    elif frame_mode == "initial_body":
        n0 = usv_n[0]
        e0 = usv_e[0]
        psi0 = yaw[0]

        usv_x, usv_y = transform_ne_to_initial_body(usv_n, usv_e, n0, e0, psi0)
        tar_x, tar_y = transform_ne_to_initial_body(tar_n, tar_e, n0, e0, psi0)

        yaw_plot = wrap_to_pi(yaw - psi0)

        xlabel = "Starboard from USV start (m)"
        ylabel = "Forward from USV start (m)"

    elif frame_mode == "ned":
        usv_x = usv_e
        usv_y = usv_n

        tar_x = tar_e
        tar_y = tar_n

        yaw_plot = yaw

        xlabel = "East (m)"
        ylabel = "North (m)"

    else:
        raise ValueError(
            "frame_mode must be 'target_aligned', 'initial_body', or 'ned'."
        )

    return data, t, usv_x, usv_y, tar_x, tar_y, yaw_plot, xlabel, ylabel


def plot_usv_tracking_from_experiment(df, figNo=1, n_marks=8):
    data, t, usv_x, usv_y, tar_x, tar_y, yaw_plot, xlabel, ylabel = build_plot_arrays(
        df,
        frame_mode=FRAME_MODE,
    )

    n = min(len(t), len(usv_x), len(tar_x))
    t = t[:n]
    usv_x = usv_x[:n]
    usv_y = usv_y[:n]
    tar_x = tar_x[:n]
    tar_y = tar_y[:n]
    yaw_plot = yaw_plot[:n]

    target_is_stationary = max(np.ptp(tar_x), np.ptp(tar_y)) <= 1e-6

    n_marks = max(2, int(n_marks))
    mark_idx = np.linspace(0, n - 1, n_marks, dtype=int)

    if t[-1] > t[0]:
        t_norm = (t - t[0]) / (t[-1] - t[0])
    else:
        t_norm = np.zeros_like(t)

    mark_cols = t_norm[mark_idx]
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=1.0)

    all_x = np.concatenate([tar_x, usv_x])
    all_y = np.concatenate([tar_y, usv_y])
    span = max(np.ptp(all_x), np.ptp(all_y), 1.0)

    circle_radius = np.clip(0.020 * span, 0.25, 0.75)
    usv_length = 1.8 * circle_radius
    usv_width = 0.90 * circle_radius

    fig = plt.figure(
        figNo,
        figsize=(cm2inch(figSizePath[0]), cm2inch(figSizePath[1])),
        dpi=dpiValue,
    )
    fig.clf()
    ax = fig.add_subplot(111)

    line_handles = []

    if not target_is_stationary:
        target_line, = ax.plot(
            tar_x,
            tar_y,
            linestyle="--",
            linewidth=lineWidth,
            color="C0",
            label="Target path",
            zorder=1,
        )
        line_handles.append(target_line)

    usv_line, = ax.plot(
        usv_x,
        usv_y,
        linestyle="-",
        linewidth=lineWidth,
        color="C1",
        label="USV path",
        zorder=2,
    )
    line_handles.append(usv_line)

    # Target markers
    if target_is_stationary:
        target_patches = [Circle((tar_x[0], tar_y[0]), radius=circle_radius)]
        target_facecolors = [cmap(0.5)]
    else:
        target_patches = [Circle((tar_x[k], tar_y[k]), radius=circle_radius) for k in mark_idx]
        target_facecolors = cmap(norm(mark_cols))

    pc_target = PatchCollection(
        target_patches,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    pc_target.set_facecolor(target_facecolors)
    ax.add_collection(pc_target)

  
    # USV heading triangles in plot frame
    L = usv_length
    W = usv_width

    # Triangle points for heading zero along +x in target-aligned mode
    if FRAME_MODE == "target_aligned":
        pts_body = np.array([
            [L, 0.0],
            [-0.55 * L, -W],
            [-0.55 * L, W],
        ])
    else:
        # Original style: heading zero along +y
        pts_body = np.array([
            [0.0, L],
            [-W, -0.55 * L],
            [W, -0.55 * L],
        ])

    usv_patches = []

    for k in mark_idx:
        psi = yaw_plot[k]
        c, s = np.cos(psi), np.sin(psi)

        if FRAME_MODE == "target_aligned":
            # heading zero points along +x
            R_xy = np.array([
                [c, -s],
                [s,  c],
            ])
        else:
            # heading zero points along +y, original style
            R_xy = np.array([
                [c,  s],
                [-s, c],
            ])

        pts = pts_body @ R_xy.T
        pts[:, 0] += usv_x[k]
        pts[:, 1] += usv_y[k]

        usv_patches.append(Polygon(pts, closed=True))

    pc_usv = PatchCollection(
        usv_patches,
        facecolor=cmap(norm(mark_cols)),
        edgecolor="black",
        linewidth=0.45,
        zorder=4,
    )
    ax.add_collection(pc_usv)

    ax.set_xlabel(xlabel, fontsize=labelSize)
    ax.set_ylabel(ylabel, fontsize=labelSize)
    ax.set_title(f"{LABEL} station-keeping" if target_is_stationary else f"{LABEL} target tracking", fontsize=titleSize)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", which="major", labelsize=tickSize)
    ax.grid(True, alpha=gridAlpha)

    if target_is_stationary:
        x_center = 0.5 * (np.nanmin(all_x) + np.nanmax(all_x))
        y_center = 0.5 * (np.nanmin(all_y) + np.nanmax(all_y))

        x_span = max(np.nanmax(all_x) - np.nanmin(all_x), 1.0)
        y_span = max(np.nanmax(all_y) - np.nanmin(all_y), 1.0)

        desired_x_span = max(x_span, 0.85 * y_span)

        x_pad = 0.30 * desired_x_span
        y_pad = 0.16 * y_span

        ax.set_xlim(x_center - 0.5 * desired_x_span - x_pad,
                    x_center + 0.5 * desired_x_span + x_pad)
        ax.set_ylim(np.nanmin(all_y) - y_pad,
                    np.nanmax(all_y) + y_pad)
    else:
        pad = 0.16 * span
        ax.set_xlim(np.nanmin(all_x) - pad, np.nanmax(all_x) + pad)
        ax.set_ylim(np.nanmin(all_y) - pad, np.nanmax(all_y) + pad)

    target_proxy = Line2D(
        [], [], linestyle="None", marker="o", markersize=7,
        markerfacecolor=cmap(0.5), markeredgecolor="black", markeredgewidth=0.6,
        label="Target position",
    )

    usv_proxy = Line2D(
        [], [], linestyle="None", marker=(3, 0, 0), markersize=8,
        markerfacecolor=cmap(0.5), markeredgecolor="black", markeredgewidth=0.6,
        label="USV position and heading",
    )

    handles = line_handles + [target_proxy, usv_proxy]

    if target_is_stationary:
        ax.legend(
            handles=handles,
            loc="lower right",
            bbox_to_anchor=(0.98, 0.04),
            fontsize=legendSize,
            frameon=True,
            framealpha=0.9,
            facecolor="white",
            edgecolor="0.5",
        )
    else:
        ax.legend(
            handles=handles,
            loc=choose_legend_location(usv_x, usv_y, tar_x, tar_y),
            fontsize=legendSize,
            frameon=True,
            framealpha=0.9,
            facecolor="white",
            edgecolor="0.5",
        )

    finalize_figure(fig, "path_body_frame", rect=(0.04, 0.04, 0.98, 0.94))


def plot_distance(df, figNo=2):
    t = df["t"].to_numpy(dtype=float)

    if "distance_to_target" in df.columns:
        distance = df["distance_to_target"].to_numpy(dtype=float)
    else:
        n_err = df["target_north_from_observer"] - df["north_from_observer"]
        e_err = df["target_east_from_observer"] - df["east_from_observer"]
        distance = np.hypot(n_err, e_err)


    distance = np.asarray(distance, dtype=float)

    valid = (
        np.isfinite(t)
        & np.isfinite(distance)
        & (distance >= 0.0)
        & (distance < 500.0)   # reject impossible startup spikes
    )

    t_plot = t[valid]
    distance_plot = distance[valid]

    if len(t_plot) < 2:
        print("Skipping distance plot: not enough valid distance samples.")
        return

    # Reset plotted time after filtering
    t_plot = t_plot - t_plot[0]

    # Optional light smoothing for single-sample jumps
    distance_plot = (
        pd.Series(distance_plot)
        .interpolate(limit_direction="both")
        .rolling(window=3, center=True, min_periods=1)
        .median()
        .to_numpy()
    )

    fig, ax = plt.subplots(
        num=figNo,
        figsize=(cm2inch(figSizeSingle[0]), cm2inch(figSizeSingle[1])),
        dpi=dpiValue,
    )

    ax.plot(t_plot, distance_plot, linewidth=lineWidth, label="Distance to target")

    apply_axis_style(
        ax,
        xlabel="Time (s)",
        ylabel="Distance (m)",
        title=f"{LABEL} distance to target",
    )

    ax.legend(fontsize=legendSize)

    finalize_figure(fig, "distance")


def plot_heading_error(df, figNo=3):
    t = df["t"].to_numpy(dtype=float)

    fig, ax = plt.subplots(
        num=figNo,
        figsize=(cm2inch(figSizeSingle[0]), cm2inch(figSizeSingle[1])),
        dpi=dpiValue,
    )

    plotted = False

    if "current_angle_deg" in df.columns:
        ax.plot(t, df["current_angle_deg"], linewidth=lineWidth, label="Yaw")
        plotted = True
    elif "yaw_rad" in df.columns:
        ax.plot(t, R2D(wrap_to_pi(df["yaw_rad"].to_numpy(dtype=float))), linewidth=lineWidth, label="Yaw")
        plotted = True

    if "yaw_setpoint_deg" in df.columns:
        ax.plot(t, df["yaw_setpoint_deg"], "--", linewidth=lineWidth, label="Yaw setpoint")
        plotted = True
    elif "yaw_setpoint" in df.columns:
        ax.plot(t, R2D(wrap_to_pi(df["yaw_setpoint"].to_numpy(dtype=float))), "--", linewidth=lineWidth, label="Yaw setpoint")
        plotted = True

    if "heading_error_deg" in df.columns:
        ax.plot(t, df["heading_error_deg"], linewidth=lineWidth, label="Heading error")
        plotted = True
    elif "heading_error" in df.columns:
        ax.plot(t, R2D(wrap_to_pi(df["heading_error"].to_numpy(dtype=float))), linewidth=lineWidth, label="Heading error")
        plotted = True

    if not plotted:
        print("Skipping heading plot: missing heading columns.")
        plt.close(fig)
        return

    apply_axis_style(ax, xlabel="Time (s)", ylabel="Angle (deg)", title=f"{LABEL} heading")
    ax.legend(fontsize=legendSize)

    finalize_figure(fig, "heading")

OTTER_MAX_SPEED = 4.5 * 0.5144  # 4.5 kn -> m/s


def smooth_signal(x, window=9):
    """
    Smooth signal before differentiation.
    """
    x = np.asarray(x, dtype=float)

    return (
        pd.Series(x)
        .interpolate(limit_direction="both")
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def despike_speed(t, U, max_speed=OTTER_MAX_SPEED, window=7):
    """
    Replace physically impossible speed spikes with a local smoothed estimate.
    """
    U = np.asarray(U, dtype=float).copy()

    if np.count_nonzero(np.isfinite(U)) < 3:
        return U

    U_smooth = (
        pd.Series(U)
        .interpolate(limit_direction="both")
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

    spike_mask = np.abs(U) > 1.25 * max_speed
    U[spike_mask] = U_smooth[spike_mask]

    return U


def get_position_velocity_ne(df):
    """
    Compute inertial North/East velocity from smoothed logged position.
    """
    t_full = df["t"].to_numpy(dtype=float)
    n_full = df["north_from_observer"].to_numpy(dtype=float)
    e_full = df["east_from_observer"].to_numpy(dtype=float)

    valid = (
        np.isfinite(t_full)
        & np.isfinite(n_full)
        & np.isfinite(e_full)
    )

    v_n = np.zeros(len(df), dtype=float)
    v_e = np.zeros(len(df), dtype=float)

    if np.count_nonzero(valid) < 3:
        return v_n, v_e

    valid_indices = np.where(valid)[0]

    t = t_full[valid]
    n = n_full[valid]
    e = e_full[valid]

    order = np.argsort(t)
    t = t[order]
    n = n[order]
    e = e[order]
    valid_indices = valid_indices[order]

    unique_t, unique_idx = np.unique(t, return_index=True)

    t = unique_t
    n = n[unique_idx]
    e = e[unique_idx]
    valid_indices = valid_indices[unique_idx]

    if len(t) < 3:
        return v_n, v_e

    n_smooth = smooth_signal(n, window=9)
    e_smooth = smooth_signal(e, window=9)

    v_n_valid = np.gradient(n_smooth, t)
    v_e_valid = np.gradient(e_smooth, t)

    v_n[valid_indices] = v_n_valid
    v_e[valid_indices] = v_e_valid

    v_n = (
        pd.Series(v_n)
        .replace(0.0, np.nan)
        .interpolate(limit_direction="both")
        .fillna(0.0)
        .to_numpy()
    )

    v_e = (
        pd.Series(v_e)
        .replace(0.0, np.nan)
        .interpolate(limit_direction="both")
        .fillna(0.0)
        .to_numpy()
    )

    return v_n, v_e


def get_experiment_surge(df):
    """
    Body-frame surge velocity from position-derived inertial velocity.
    """
    v_n, v_e = get_position_velocity_ne(df)
    psi = get_yaw_rad(df)

    return np.cos(psi) * v_n + np.sin(psi) * v_e


def get_experiment_sway(df):
    """
    Body-frame sway velocity from position-derived inertial velocity.
    """
    v_n, v_e = get_position_velocity_ne(df)
    psi = get_yaw_rad(df)

    return -np.sin(psi) * v_n + np.cos(psi) * v_e


def get_experiment_total_speed(df):
    """
    Total horizontal speed from position-derived inertial velocity.
    """
    t = df["t"].to_numpy(dtype=float)

    v_n, v_e = get_position_velocity_ne(df)
    U = np.hypot(v_n, v_e)

    U = despike_speed(
        t,
        U,
        max_speed=OTTER_MAX_SPEED,
        window=7,
    )

    return U


def get_experiment_yaw_rate(df):
    if "current_rotational_velocities_3" in df.columns:
        return np.deg2rad(df["current_rotational_velocities_3"].to_numpy(dtype=float))

    yaw = get_yaw_rad(df)
    t = df["t"].to_numpy(dtype=float)

    if len(t) < 2:
        return np.zeros(len(df), dtype=float)

    return np.gradient(np.unwrap(yaw), t)

    return np.gradient(np.unwrap(yaw), t)

def plotSpeed(df, figNo=4):
    t = df["t"].to_numpy(dtype=float)
    U = get_experiment_total_speed(df)

    # Otter speed should not change by several m/s in one sample.
    U = despike_signal(t, U, max_rate=2.0, window=5)

    fig, ax = plt.subplots(
        num=figNo,
        figsize=(cm2inch(figSizeSingle[0]), cm2inch(figSizeSingle[1])),
        dpi=dpiValue,
    )

    ax.plot(t, U, linewidth=lineWidth, label="Total velocity")
    apply_axis_style(
        ax,
        xlabel="Time (s)",
        ylabel="Velocity (m/s)",
        title=f"{LABEL} total velocity",
    )
    ax.legend(fontsize=legendSize)

    finalize_figure(fig, "speed")


def plotSurge(df, figNo=5):
    t = df["t"].to_numpy(dtype=float)
    u = get_experiment_surge(df)

    # Remove physically unrealistic single-sample spikes for plotting
    u = despike_signal(t, u, max_rate=2.0, window=5)

    fig, ax = plt.subplots(
        num=figNo,
        figsize=(cm2inch(figSizeSingle[0]), cm2inch(figSizeSingle[1])),
        dpi=dpiValue,
    )

    ax.plot(t, u, linewidth=lineWidth, label="Surge velocity")
    apply_axis_style(
        ax,
        xlabel="Time (s)",
        ylabel="u (m/s)",
        title=f"{LABEL} surge velocity",
    )
    ax.legend(fontsize=legendSize)

    finalize_figure(fig, "surge")


def plotSway(df, figNo=6):
    t = df["t"].to_numpy(dtype=float)
    v = get_experiment_sway(df)

    # Remove physically unrealistic single-sample spikes for plotting
    v = despike_signal(t, v, max_rate=2.0, window=5)

    fig, ax = plt.subplots(
        num=figNo,
        figsize=(cm2inch(figSizeSingle[0]), cm2inch(figSizeSingle[1])),
        dpi=dpiValue,
    )

    ax.plot(t, v, linewidth=lineWidth, label="Sway velocity")
    apply_axis_style(
        ax,
        xlabel="Time (s)",
        ylabel="v (m/s)",
        title=f"{LABEL} sway velocity",
    )
    ax.legend(fontsize=legendSize)

    finalize_figure(fig, "sway")

    ax.plot(t, v, linewidth=lineWidth, label="Sway velocity")
    apply_axis_style(
        ax,
        xlabel="Time (s)",
        ylabel="v (m/s)",
        title=f"{LABEL} sway velocity",
    )
    ax.legend(fontsize=legendSize)

    finalize_figure(fig, "sway")


def plotYaw(df, figNo=7):
    t = df["t"].to_numpy(dtype=float)
    r = get_experiment_yaw_rate(df)

    fig, ax = plt.subplots(
        num=figNo,
        figsize=(cm2inch(figSizeSingle[0]), cm2inch(figSizeSingle[1])),
        dpi=dpiValue,
    )

    ax.plot(t, r, linewidth=lineWidth, label="Yaw rate")
    apply_axis_style(
        ax,
        xlabel="Time (s)",
        ylabel="r (rad/s)",
        title=f"{LABEL} yaw rate",
    )
    ax.legend(fontsize=legendSize)

    finalize_figure(fig, "yaw_rate")


def plot_controls_tau(df, figNo=5):
    t = df["t"].to_numpy(dtype=float)

    # Live logs are generalized forces/moments, not shaft speeds.
    cmd_cols = [
        ("tau_X", "Applied tau_X"),
        ("tau_N", "Applied tau_N"),
    ]

    alt_cmd_cols = [
        ("controller_X_cmd", "Controller X command"),
        ("controller_N_cmd", "Controller N command"),
    ]

    fig, axs = plt.subplots(
        1, 2,
        num=figNo,
        figsize=(cm2inch(figSizeMulti[0]), cm2inch(figSizeMulti[1])),
        dpi=dpiValue,
        sharex=True,
    )
    axs = np.atleast_1d(axs).ravel()

    for i, ax in enumerate(axs):
        plotted = False

        col, label = cmd_cols[i]
        if col in df.columns:
            ax.plot(t, df[col], linewidth=lineWidth, label=label)
            plotted = True

        alt_col, alt_label = alt_cmd_cols[i]
        if alt_col in df.columns:
            ax.plot(t, df[alt_col], "--", linewidth=lineWidth, label=alt_label)
            plotted = True

        if not plotted:
            ax.text(0.5, 0.5, "Missing control data", transform=ax.transAxes,
                    ha="center", va="center")

        ylabel = "Force (N)" if i == 0 else "Moment (Nm)"
        apply_axis_style(ax, xlabel="Time (s)", ylabel=ylabel, title=CONTROL_LABELS_TAU[i])
        ax.legend(loc="upper right", fontsize=legendSize, frameon=True)

    fig.suptitle(f"{LABEL} control response", fontsize=titleSize + 2, y=0.96)
    finalize_figure(fig, "controls_tau", rect=(0.04, 0.04, 0.98, 0.90))

def allocate_tau_to_rads(tau_X, tau_N):
    """
    Same allocation principle as Control.py:
        tau = B * u_alloc
        u_alloc = abs(n) * n
        n = sign(u_alloc) * sqrt(abs(u_alloc))

    Returns left/right shaft speed in rad/s.
    """

    # Same constants as Control.py
    y_pont = 0.395
    l1 = -y_pont
    l2 = y_pont
    k_pos = 0.02216 / 2.0

    B = k_pos * np.array([
        [1.0, 1.0],
        [-l1, -l2],
    ])

    Binv = np.linalg.inv(B)

    # Use same command limits as live DRL / Otter control
    tau_X = np.clip(tau_X, -150.0, 150.0)
    tau_N = np.clip(tau_N, -116.0, 116.0)

    tau = np.array([tau_X, tau_N], dtype=float)
    u_alloc = Binv @ tau

    n_left = np.sign(u_alloc[0]) * np.sqrt(abs(u_alloc[0]))
    n_right = np.sign(u_alloc[1]) * np.sqrt(abs(u_alloc[1]))

    return n_left, n_right


def allocate_series_to_rads(tau_X_series, tau_N_series):
    n_left = np.zeros(len(tau_X_series))
    n_right = np.zeros(len(tau_X_series))

    for i, (tx, tn) in enumerate(zip(tau_X_series, tau_N_series)):
        n_left[i], n_right[i] = allocate_tau_to_rads(tx, tn)

    return n_left, n_right


def plot_controls_allocated(df, figNo=5):
    t = df["t"].to_numpy(dtype=float)

    if "tau_X" not in df.columns or "tau_N" not in df.columns:
        print("Skipping allocated control plot: missing tau_X or tau_N.")
        return

    # Applied generalized command sent to PMARMAN
    tau_X = df["tau_X"].to_numpy(dtype=float)
    tau_N = df["tau_N"].to_numpy(dtype=float)

    n_left_applied, n_right_applied = allocate_series_to_rads(tau_X, tau_N)

    has_controller_cmd = (
        "controller_X_cmd" in df.columns
        and "controller_N_cmd" in df.columns
    )

    if has_controller_cmd:
        # If controller_X_cmd/controller_N_cmd are normalized DRL actions,
        # scale them to generalized force/moment first.
        controller_X = df["controller_X_cmd"].to_numpy(dtype=float)
        controller_N = df["controller_N_cmd"].to_numpy(dtype=float)

        # Detect normalized commands automatically
        if np.nanmax(np.abs(controller_X)) <= 1.5:
            controller_X = controller_X * 150.0

        if np.nanmax(np.abs(controller_N)) <= 1.5:
            controller_N = controller_N * 116.0

        n_left_controller, n_right_controller = allocate_series_to_rads(
            controller_X,
            controller_N,
        )

    fig, axs = plt.subplots(
        1, 2,
        num=figNo,
        figsize=(cm2inch(figSizeMulti[0]), cm2inch(figSizeMulti[1])),
        dpi=dpiValue,
        sharex=True,
    )

    axs = np.atleast_1d(axs).ravel()

    axs[0].plot(t, n_left_applied, linewidth=lineWidth, label="Applied command")
    if has_controller_cmd:
        axs[0].plot(
            t,
            n_left_controller,
            "--",
            linewidth=lineWidth,
            label="Controller command",
        )

    apply_axis_style(
        axs[0],
        xlabel="Time (s)",
        ylabel="Shaft speed (rad/s)",
        title="Left propeller",
    )
    axs[0].legend(loc="upper right", fontsize=legendSize, frameon=True)

    axs[1].plot(t, n_right_applied, linewidth=lineWidth, label="Applied command")
    if has_controller_cmd:
        axs[1].plot(
            t,
            n_right_controller,
            "--",
            linewidth=lineWidth,
            label="Controller command",
        )

    apply_axis_style(
        axs[1],
        xlabel="Time (s)",
        ylabel="Shaft speed (rad/s)",
        title="Right propeller",
    )
    axs[1].legend(loc="upper right", fontsize=legendSize, frameon=True)

    fig.suptitle(f"{LABEL} actuator response", fontsize=titleSize + 2, y=0.96)
    finalize_figure(fig, "controls_allocated", rect=(0.04, 0.04, 0.98, 0.90))



def plot_position_components_body(df, figNo=6):
    data, t, usv_x, usv_y, tar_x, tar_y, _, xlabel, ylabel = build_plot_arrays(
        df,
        frame_mode=FRAME_MODE,
    )

    fig, ax = plt.subplots(
        num=figNo,
        figsize=(cm2inch(figSizeSingle[0]), cm2inch(figSizeSingle[1])),
        dpi=dpiValue,
    )

    ax.plot(t, usv_y, linewidth=lineWidth, label="USV forward")
    ax.plot(t, tar_y, "--", linewidth=lineWidth, label="Target forward")
    ax.plot(t, usv_x, linewidth=lineWidth, label="USV starboard")
    ax.plot(t, tar_x, "--", linewidth=lineWidth, label="Target starboard")

    apply_axis_style(ax, xlabel="Time (s)", ylabel="Position (m)", title=f"{LABEL} position components")
    ax.legend(fontsize=legendSize)

    finalize_figure(fig, "position_components_body")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Running file: {__file__}")
    print(f"LABEL used by script: {LABEL}")
    print(f"CSV_PATH: {CSV_PATH}")
    print(f"PLOT_DIR: {PLOT_DIR}")
    print(f"Frame mode: {FRAME_MODE}")
    print(f"Time window: {START_TIME_S} to {END_TIME_S} s")

    df = load_experiment_csv(
        CSV_PATH,
        start_time_s=START_TIME_S,
        end_time_s=END_TIME_S,
    )

    print(f"Loaded samples: {len(df)}")
    print(f"Time range after crop: {df['t'].iloc[0]:.3f} to {df['t'].iloc[-1]:.3f} s")

    plots = PLOTS_TO_MAKE
    if "all" in plots:
        plots = [
            "path",
            "distance",
            "heading",
            "speed",
            "surge",
            "sway",
            "yaw",
            "controls",
            "position",
        ]

    fig_no = 1

    if "path" in plots:
        plot_usv_tracking_from_experiment(df, fig_no)
        fig_no += 1

    if "distance" in plots:
        plot_distance(df, fig_no)
        fig_no += 1

    if "heading" in plots:
        plot_heading_error(df, fig_no)
        fig_no += 1


    if "speed" in plots:
        plotSpeed(df, fig_no)
        fig_no += 1

    if "surge" in plots:
        plotSurge(df, fig_no)
        fig_no += 1

    if "sway" in plots:
        plotSway(df, fig_no)
        fig_no += 1

    if "yaw" in plots:
        plotYaw(df, fig_no)
        fig_no += 1

    if "controls" in plots:
        plot_controls_allocated(df, fig_no)
        fig_no += 1

    if "position" in plots:
        plot_position_components_body(df, fig_no)
        fig_no += 1

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
