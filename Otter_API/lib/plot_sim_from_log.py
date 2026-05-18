import os
import math
import numpy as np
import pandas as pd
import matplotlib

# Use TkAgg if running locally with plots
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, Polygon
from matplotlib.colors import Normalize
import os


# =============================================================================
# USER SETTINGS
# =============================================================================

from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent

# _THIS_DIR:
# C:/Repos/Master_Thesis/Code/Otter-USV/Otter_API/lib
#
# _PROJECT_ROOT:
# C:/Repos/Master_Thesis/Code/Otter-USV/Otter_API
_PROJECT_ROOT = _THIS_DIR.parent

LABEL = "NMPC"

CSV_PATH = (
    _PROJECT_ROOT
    / "logs"
    / "sim_logs"
    / "mpc_sim_station_dist"
    / "sim_log_nmpc.csv"
)

PLOT_DIR = (
    _PROJECT_ROOT
    / "logs"
    / "sim_plots_pid_track_dist"
)

CSV_PATH = str(CSV_PATH)
PLOT_DIR = str(PLOT_DIR)

SAVE_PLOTS = True
SHOW_PLOTS = True
SAVE_EPS = False
SAVE_PDF = True

PLOTS_TO_MAKE = ["all"]
SKIP_FIRST = 0

# Figure style
legendSize = 9
titleSize = 14
labelSize = 12
tickSize = 10
lineWidth = 0.9
gridAlpha = 0.65

figSizeSingle = [18, 10]      # cm, single-axis plots
figSizeMulti = [25, 15]       # cm, multi-axis plots
figSizePath = [18, 14]        # cm, path plot
dpiValue = 150

CONTROL_LABELS = [
    "Left propeller",
    "Right propeller",
]

def choose_legend_location(x1, y1, x2, y2):
    """
    Choose the least crowded legend corner based on path samples.

    Returns one of:
        'upper left', 'upper right', 'lower left', 'lower right'
    """

    xs = np.concatenate([np.asarray(x1), np.asarray(x2)])
    ys = np.concatenate([np.asarray(y1), np.asarray(y2)])

    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)

    xspan = max(xmax - xmin, 1e-9)
    yspan = max(ymax - ymin, 1e-9)

    # Normalize to [0, 1]
    xn = (xs - xmin) / xspan
    yn = (ys - ymin) / yspan

    # Corner boxes: slightly larger boxes to reflect legend footprint
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

    # Prefer lower right slightly when scores are equal
    preference = ["lower right", "upper right", "lower left", "upper left"]

    best_score = min(scores.values())
    candidates = [loc for loc, score in scores.items() if score == best_score]

    for loc in preference:
        if loc in candidates:
            return loc

    return "lower right"

# =============================================================================
# HELPERS
# =============================================================================

def R2D(value):
    return value * 180.0 / math.pi


def cm2inch(value):
    return value / 2.54


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def ssa(angle):
    return wrap_to_pi(angle)


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


def load_log(path, skip_first=0):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find CSV file: {path}")

    df = pd.read_csv(path)

    if "simTime" not in df.columns:
        raise ValueError("CSV must contain a 'simTime' column.")

    sim_cols = sorted_prefixed_columns(df, "simData_")
    target_cols = sorted_prefixed_columns(df, "targetData_")

    if len(sim_cols) == 0:
        raise ValueError("No simData_* columns found in CSV.")

    if len(target_cols) == 0:
        raise ValueError("No targetData_* columns found in CSV.")

    simTime = clean_simtime(df["simTime"])
    simData = df[sim_cols].to_numpy(dtype=float)
    targetData = df[target_cols].to_numpy(dtype=float)

    n = min(len(simTime), len(simData), len(targetData))
    simTime = simTime[:n]
    simData = simData[:n, :]
    targetData = targetData[:n, :]

    if skip_first > 0:
        simTime = simTime[skip_first:]
        simData = simData[skip_first:, :]
        targetData = targetData[skip_first:, :]

    return simTime, simData, targetData


def get_col(simData, idx, default=0.0):
    if simData.shape[1] > idx:
        return simData[:, idx]
    return np.full(simData.shape[0], default, dtype=float)


def compute_tracking_errors(simData, targetData):
    usv_north = get_col(simData, 0)
    usv_east = get_col(simData, 1)
    yaw = get_col(simData, 5)

    target_north = targetData[:, 0]
    target_east = targetData[:, 1]

    north_error = target_north - usv_north
    east_error = target_east - usv_east

    distance = np.sqrt(north_error**2 + east_error**2)

    yaw_setpoint = np.arctan2(east_error, north_error)
    heading_error = wrap_to_pi(yaw_setpoint - yaw)

    return distance, heading_error, yaw_setpoint


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
    """
    Applies consistent tight layout and saves the figure.
    rect reserves space for suptitle/shared legend.
    """
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


def new_single_figure(figNo):
    return plt.figure(
        figNo,
        figsize=(cm2inch(figSizeSingle[0]), cm2inch(figSizeSingle[1])),
        dpi=dpiValue,
    )


def new_multi_figure(figNo):
    return plt.figure(
        figNo,
        figsize=(cm2inch(figSizeMulti[0]), cm2inch(figSizeMulti[1])),
        dpi=dpiValue,
    )


# =============================================================================
# PLOTS
# =============================================================================

def plotVehicleStates(simTime, simData, figNo=1):
    t = simTime

    x = get_col(simData, 0)
    y = get_col(simData, 1)
    z = get_col(simData, 2)

    phi = R2D(ssa(get_col(simData, 3)))
    theta = R2D(ssa(get_col(simData, 4)))
    psi = R2D(ssa(get_col(simData, 5)))

    u = get_col(simData, 6)
    v = get_col(simData, 7)
    w = get_col(simData, 8)

    p = R2D(get_col(simData, 9))
    q = R2D(get_col(simData, 10))
    r = R2D(get_col(simData, 11))

    U = np.sqrt(u**2 + v**2 + w**2)

    beta_c = R2D(ssa(np.arctan2(v, u)))
    alpha_c = R2D(ssa(np.arctan2(w, u)))
    chi = R2D(ssa(get_col(simData, 5) + np.arctan2(v, u)))

    fig, axs = plt.subplots(
        3,
        3,
        num=figNo,
        figsize=(cm2inch(figSizeMulti[0]), cm2inch(figSizeMulti[1])),
        dpi=dpiValue,
    )

    axs = axs.ravel()

    axs[0].plot(y, x, linewidth=lineWidth)
    apply_axis_style(axs[0], xlabel="East (m)", ylabel="North (m)", title="North-East position")

    axs[1].plot(t, z, linewidth=lineWidth)
    apply_axis_style(axs[1], xlabel="Time (s)", ylabel="Depth (m)", title="Depth")

    axs[2].plot(t, phi, linewidth=lineWidth, label="Roll")
    axs[2].plot(t, theta, linewidth=lineWidth, label="Pitch")
    apply_axis_style(axs[2], xlabel="Time (s)", ylabel="Angle (deg)", title="Roll and pitch")
    axs[2].legend(fontsize=legendSize)

    axs[3].plot(t, U, linewidth=lineWidth)
    apply_axis_style(axs[3], xlabel="Time (s)", ylabel="Velocity (m/s)", title="Total velocity")

    axs[4].plot(t, chi, linewidth=lineWidth)
    apply_axis_style(axs[4], xlabel="Time (s)", ylabel="Course (deg)", title="Course angle")

    axs[5].plot(t, theta, linewidth=lineWidth, label="Pitch")
    axs[5].plot(t, alpha_c, linewidth=lineWidth, label="Flight path")
    apply_axis_style(axs[5], xlabel="Time (s)", ylabel="Angle (deg)", title="Pitch and flight path")
    axs[5].legend(fontsize=legendSize)

    axs[6].plot(t, u, linewidth=lineWidth, label="Surge")
    axs[6].plot(t, v, linewidth=lineWidth, label="Sway")
    axs[6].plot(t, w, linewidth=lineWidth, label="Heave")
    apply_axis_style(axs[6], xlabel="Time (s)", ylabel="Velocity (m/s)", title="Linear velocities")
    axs[6].legend(fontsize=legendSize)

    axs[7].plot(t, p, linewidth=lineWidth, label="Roll rate")
    axs[7].plot(t, q, linewidth=lineWidth, label="Pitch rate")
    axs[7].plot(t, r, linewidth=lineWidth, label="Yaw rate")
    apply_axis_style(axs[7], xlabel="Time (s)", ylabel="Rate (deg/s)", title="Angular rates")
    axs[7].legend(fontsize=legendSize)

    axs[8].plot(t, psi, linewidth=lineWidth, label="Yaw")
    axs[8].plot(t, beta_c, linewidth=lineWidth, label="Crab angle")
    apply_axis_style(axs[8], xlabel="Time (s)", ylabel="Angle (deg)", title="Yaw and crab angle")
    axs[8].legend(fontsize=legendSize)

    fig.suptitle(f"{LABEL} vehicle states", fontsize=titleSize + 2)
    finalize_figure(fig, "vehicle_states")


def plotControls(simTime, simData, figNo=2):
    """
    Assumes simData layout:
        eta       = simData[:, 0:6]
        nu        = simData[:, 6:12]
        u_control = simData[:, 12:14]
        u_actual  = simData[:, 14:16]
    """
    DOF = 6
    dimU = len(CONTROL_LABELS)
    t = simTime

    if simData.shape[1] < 2 * DOF + dimU:
        print("Not enough simData columns to plot controls.")
        print(f"simData shape: {simData.shape}")
        return

    u_cmd_all = simData[:, 2 * DOF : 2 * DOF + dimU]

    if simData.shape[1] >= 2 * DOF + 2 * dimU:
        u_act_all = simData[:, 2 * DOF + dimU : 2 * DOF + 2 * dimU]
        has_actual = True
    else:
        u_act_all = np.full_like(u_cmd_all, np.nan)
        has_actual = False

    values_for_limits = [u_cmd_all]
    if has_actual:
        values_for_limits.append(u_act_all)

    all_values = np.hstack(values_for_limits)
    global_min = np.nanmin(all_values)
    global_max = np.nanmax(all_values)

    span = global_max - global_min if global_max > global_min else 1.0
    pad = 0.12 * span

    col = 2
    row = int(math.ceil(dimU / col))

    fig, axs = plt.subplots(
        row,
        col,
        num=figNo,
        figsize=(cm2inch(figSizeMulti[0]), cm2inch(figSizeMulti[1])),
        dpi=dpiValue,
        sharex=True,
    )

    axs = np.atleast_1d(axs).ravel()

    legend_handles = []
    legend_labels = []

    for i in range(dimU):
        ax = axs[i]

        u_control = u_cmd_all[:, i].copy()
        u_actual = u_act_all[:, i].copy()

        ax.plot(
            t,
            u_control,
            linewidth=lineWidth,
            label="Command",
        )

        if has_actual:
            ax.plot(
                t,
                u_actual,
                linewidth=lineWidth,
                label="Actual",
            )

        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(global_min - pad, global_max + pad)

        apply_axis_style(
            ax,
            xlabel="Time (s)",
            ylabel="Shaft speed (rad/s)",
            title=CONTROL_LABELS[i],
        )

        ax.legend(
            loc="upper right",
            fontsize=legendSize,
            frameon=True,
        )

    for j in range(dimU, len(axs)):
        axs[j].set_visible(False)

    fig.suptitle(
        f"{LABEL} actuator response",
        fontsize=titleSize + 2,
        y=0.96,
    )

    finalize_figure(fig, "controls", rect=(0.04, 0.04, 0.98, 0.90))


def plotPosTar2(simTime, simData, targetData, figNo=3, n_marks=4, legend_loc="best"):
    """
    Plot USV and target path.

    Parameters
    ----------
    n_marks : int
        Number of USV/target position markers along the trajectory.
        Example: n_marks=5 gives fewer symbols, n_marks=20 gives more.
    legend_loc : str
        Matplotlib legend location inside the plot.
        Examples: "best", "upper right", "lower left", "upper left".
    """

    n_common = min(len(simData), len(targetData), len(simTime))
    simTime = simTime[:n_common]
    simData = simData[:n_common, :]
    targetData = targetData[:n_common, :]

    usv_north = simData[:, 0]
    usv_east = simData[:, 1]
    yaw = simData[:, 5]

    tar_north = targetData[:, 0]
    tar_east = targetData[:, 1]

    if simTime[-1] > simTime[0]:
        t_norm = ((simTime - simTime[0]) / (simTime[-1] - simTime[0])).ravel()
    else:
        t_norm = np.zeros_like(simTime)

    # Number of marker indices
    n_marks = max(1, int(n_marks))
    mark_idx = np.linspace(0, n_common - 1, n_marks + 1, dtype=int)
    mark_cols = t_norm[mark_idx]
    psi_marks = yaw[mark_idx]

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0.0, vmax=1.0)

    fig, ax = plt.subplots(
        num=figNo,
        figsize=(cm2inch(figSizePath[0]), cm2inch(figSizePath[1])),
        dpi=dpiValue,
    )

    ax.plot(
        tar_east,
        tar_north,
        color="C0",
        lw=lineWidth,
        linestyle="--",
        label="Target path",
    )

    ax.plot(
        usv_east,
        usv_north,
        color="C1",
        lw=lineWidth,
        label="USV path",
    )

    # Target circles
    target_patches = []
    circle_radius = 1.0

    for x, y in zip(tar_east[mark_idx], tar_north[mark_idx]):
        target_patches.append(Circle((x, y), radius=circle_radius))

    pc_target = PatchCollection(
        target_patches,
        facecolor=cmap(norm(mark_cols)),
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )

    # USV triangles
    L, W = 2.0, 1.08
    patches = []

    for k, angle in zip(mark_idx, psi_marks):
        pts_body = np.array([
            [L, 0.0],
            [-L / 2, -W],
            [-L / 2, W],
        ])

        c, s = np.cos(angle), np.sin(angle)
        R_ne = np.array([
            [c, -s],
            [s, c],
        ])

        pts_ne = pts_body @ R_ne.T
        pts_plot = np.column_stack((pts_ne[:, 1], pts_ne[:, 0]))
        pts_plot[:, 0] += usv_east[k]
        pts_plot[:, 1] += usv_north[k]

        patches.append(Polygon(pts_plot))

    pc = PatchCollection(
        patches,
        edgecolor="black",
        linewidth=0.6,
        zorder=4,
    )
    pc.set_facecolor(cmap(norm(mark_cols)))

    ax.add_collection(pc_target)
    ax.add_collection(pc)

    apply_axis_style(
        ax,
        xlabel="East (m)",
        ylabel="North (m)",
        title=f"{LABEL} path",
    )

    ax.set_aspect("equal", adjustable="box")

    all_east = np.concatenate([tar_east, usv_east])
    all_north = np.concatenate([tar_north, usv_north])

    min_e, max_e = all_east.min(), all_east.max()
    min_n, max_n = all_north.min(), all_north.max()

    width = max_e - min_e
    height = max_n - min_n
    pad = 0.2 * max(width, height, 1.0)

    ax.set_xlim(min_e - pad, max_e + pad)
    ax.set_ylim(min_n - pad, max_n + pad)

    line_handles, _ = ax.get_legend_handles_labels()

    target_proxy = Line2D(
        [],
        [],
        linestyle="None",
        marker="o",
        markersize=8,
        markerfacecolor="0.6",
        markeredgecolor="0.3",
        label="Target position",
    )

    usv_proxy = Line2D(
        [],
        [],
        linestyle="None",
        marker=(3, 0, 0),
        markersize=10,
        markerfacecolor="0.6",
        markeredgecolor="0.3",
        label="USV position and heading",
    )

    ax.legend(
        handles=line_handles + [target_proxy, usv_proxy],
        fontsize=legendSize,
        loc=legend_loc,
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="0.5",
    )

    finalize_figure(fig, "path", rect=(0.04, 0.04, 0.98, 0.94))


"""
IEEE-publication-style plot for USV target-tracking simulation results.

Dependencies: matplotlib, numpy
"""




IEEE_RC = {
    # Font
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":            9,
    "axes.titlesize":       9,
    "axes.labelsize":       9,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      8,
    # Line widths
    "axes.linewidth":       0.8,
    "grid.linewidth":       0.5,
    "lines.linewidth":      1.0,
    # Grid
    "axes.grid":            True,
    "grid.alpha":           0.4,
    "grid.linestyle":       "--",
    # Ticks
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.major.size":     3.5,
    "ytick.major.size":     3.5,
    "xtick.minor.visible":  False,
    "ytick.minor.visible":  False,
    # Figure
    "figure.dpi":           300,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    # Legend
    "legend.framealpha":    0.9,
    "legend.edgecolor":     "0.7",
    "legend.handlelength":  1.8,
}

# Single-column IEEE figure width ≈ 8.9 cm; double-column ≈ 18.4 cm
_CM2IN = 1 / 2.54


def _cm2in(cm):
    return cm * _CM2IN



def plot_usv_tracking(
    sim_time: np.ndarray,
    sim_data: np.ndarray,
    target_data: np.ndarray,
    *,
    n_marks: int = 8,
    usv_length: float | None = None,
    usv_width: float | None = None,
    circle_radius: float | None = None,
    cmap_name: str = "viridis",
    fig_width_cm: float = 18.0,
    fig_height_cm: float = 14.0,
    save_path: str | None = None,
    fig_number: int | None = None,
    show: bool = True,
    legend_loc: str = "upper left",
    show_annotations: bool = True,
    station_tol: float = 1e-6,
) -> plt.Figure:
    """
    Plot USV and target trajectories using the same style as the other plots.

    sim_data columns:
        [0] north (m), [1] east (m), [5] yaw (rad)

    target_data columns:
        [0] north (m), [1] east (m)
    """

    # ------------------------------------------------------------------
    # Data alignment
    # ------------------------------------------------------------------
    sim_time = np.asarray(sim_time).ravel()
    sim_data = np.asarray(sim_data)
    target_data = np.asarray(target_data)

    if target_data.ndim == 2 and len(target_data) > len(sim_data):
        extra = len(target_data) - len(sim_data)
        if extra == 2:
            target_data = target_data[1:-1]

    n = min(len(sim_time), len(sim_data), len(target_data))

    if n < 2:
        raise ValueError("Need at least two samples to plot USV tracking.")

    t = sim_time[:n]

    usv_n = sim_data[:n, 0]
    usv_e = sim_data[:n, 1]
    yaw = sim_data[:n, 5]

    tar_n = target_data[:n, 0]
    tar_e = target_data[:n, 1]

    # ------------------------------------------------------------------
    # Determine whether target is stationary
    # ------------------------------------------------------------------
    target_span_e = np.ptp(tar_e)
    target_span_n = np.ptp(tar_n)
    target_is_stationary = max(target_span_e, target_span_n) <= station_tol

    # ------------------------------------------------------------------
    # Marker positions and colors
    # ------------------------------------------------------------------
    n_marks = max(2, int(n_marks))
    mark_idx = np.linspace(0, n - 1, n_marks, dtype=int)

    if t[-1] > t[0]:
        t_norm = (t - t[0]) / (t[-1] - t[0])
    else:
        t_norm = np.zeros_like(t)

    mark_cols = t_norm[mark_idx]

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=0.0, vmax=1.0)

    # ------------------------------------------------------------------
    # Marker scaling
    # ------------------------------------------------------------------
    all_e = np.concatenate([tar_e, usv_e])
    all_n = np.concatenate([tar_n, usv_n])

    span = max(np.ptp(all_e), np.ptp(all_n), 1.0)

    if circle_radius is None:
        circle_radius = np.clip(0.020 * span, 0.25, 0.75)

    if usv_length is None:
        usv_length = 1.8 * circle_radius

    if usv_width is None:
        usv_width = 0.90 * circle_radius

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig = plt.figure(
        fig_number,
        figsize=(cm2inch(fig_width_cm), cm2inch(fig_height_cm)),
        dpi=dpiValue,
    )
    fig.clf()
    ax = fig.add_subplot(111)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    line_handles = []

    if not target_is_stationary:
        target_line, = ax.plot(
            tar_e,
            tar_n,
            linestyle="--",
            linewidth=lineWidth,
            color="C0",
            label="Target path",
            zorder=1,
        )
        line_handles.append(target_line)

    usv_line, = ax.plot(
        usv_e,
        usv_n,
        linestyle="-",
        linewidth=lineWidth,
        color="C1",
        label="USV path",
        zorder=2,
    )
    line_handles.append(usv_line)

    # ------------------------------------------------------------------
    # Target circles
    # ------------------------------------------------------------------
    if target_is_stationary:
        # Show one target position marker only
        target_patches = [
            Circle((tar_e[0], tar_n[0]), radius=circle_radius)
        ]
        target_facecolors = [cmap(0.5)]
    else:
        target_patches = [
            Circle((tar_e[k], tar_n[k]), radius=circle_radius)
            for k in mark_idx
        ]
        target_facecolors = cmap(norm(mark_cols))

    pc_target = PatchCollection(
        target_patches,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    pc_target.set_facecolor(target_facecolors)
    ax.add_collection(pc_target)

    # ------------------------------------------------------------------
    # USV heading triangles
    # ------------------------------------------------------------------
    L = usv_length
    W = usv_width

    pts_body = np.array([
        [L, 0.0],
        [-0.55 * L, -W],
        [-0.55 * L,  W],
    ])

    usv_patches = []

    for k in mark_idx:
        psi = yaw[k]

        c, s = np.cos(psi), np.sin(psi)

        R_ne = np.array([
            [c, -s],
            [s,  c],
        ])

        pts_ne = pts_body @ R_ne.T

        # Convert from North-East to plot coordinates East-North
        pts_plot = np.column_stack((
            pts_ne[:, 1] + usv_e[k],
            pts_ne[:, 0] + usv_n[k],
        ))

        usv_patches.append(Polygon(pts_plot, closed=True))

    pc_usv = PatchCollection(
        usv_patches,
        facecolor=cmap(norm(mark_cols)),
        edgecolor="black",
        linewidth=0.45,
        zorder=4,
    )

    ax.add_collection(pc_usv)

    # ------------------------------------------------------------------
    # Optional annotations
    # ------------------------------------------------------------------
    if show_annotations:
        ax.annotate(
            "Start",
            xy=(usv_e[0], usv_n[0]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=tickSize,
            color="0.25",
        )

        if not target_is_stationary:
            ax.annotate(
                "End",
                xy=(tar_e[-1], tar_n[-1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=tickSize,
                color="0.25",
            )

    # ------------------------------------------------------------------
    # Axes formatting
    # ------------------------------------------------------------------
    ax.set_xlabel("East (m)", fontsize=labelSize)
    ax.set_ylabel("North (m)", fontsize=labelSize)

    if target_is_stationary:
        ax.set_title(f"{LABEL} station-keeping", fontsize=titleSize)
    else:
        ax.set_title(f"{LABEL} target tracking", fontsize=titleSize)

    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", which="major", labelsize=tickSize)
    ax.grid(True, alpha=gridAlpha)

    # ------------------------------------------------------------------
    # Axis limits
    # ------------------------------------------------------------------
    if target_is_stationary:
        # Station-keeping plots are often narrow in East.
        # Widen the East axis to create natural empty space for the legend.
        min_e = np.min(all_e)
        max_e = np.max(all_e)
        min_n = np.min(all_n)
        max_n = np.max(all_n)

        e_center = 0.5 * (min_e + max_e)
        n_center = 0.5 * (min_n + max_n)

        e_span = max(max_e - min_e, 1.0)
        n_span = max(max_n - min_n, 1.0)

        # Make East span at least comparable to North span.
        # This creates horizontal room without changing the equal-axis scaling.
        desired_e_span = max(e_span, 0.85 * n_span)

        e_pad = 0.30 * desired_e_span
        n_pad = 0.16 * n_span

        ax.set_xlim(
            e_center - 0.5 * desired_e_span - e_pad,
            e_center + 0.5 * desired_e_span + e_pad,
        )
        ax.set_ylim(
            min_n - n_pad,
            max_n + n_pad,
        )

    else:
        pad = 0.16 * span

        ax.set_xlim(np.min(all_e) - pad, np.max(all_e) + pad)
        ax.set_ylim(np.min(all_n) - pad, np.max(all_n) + pad)

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    target_proxy = Line2D(
        [],
        [],
        linestyle="None",
        marker="o",
        markersize=7,
        markerfacecolor=cmap(0.5),
        markeredgecolor="black",
        markeredgewidth=0.6,
        label="Target position",
    )

    usv_proxy = Line2D(
        [],
        [],
        linestyle="None",
        marker=(3, 0, 0),
        markersize=8,
        markerfacecolor=cmap(0.5),
        markeredgecolor="black",
        markeredgewidth=0.6,
        label="USV position and heading",
    )

    legend_handles = line_handles + [target_proxy, usv_proxy]

    if target_is_stationary:
        ax.legend(
            handles=legend_handles,
            loc="lower right",
            bbox_to_anchor=(0.98, 0.04),
            fontsize=legendSize,
            frameon=True,
            framealpha=0.9,
            facecolor="white",
            edgecolor="0.5",
        )
    else:
        chosen_legend_loc = choose_legend_location(usv_e, usv_n, tar_e, tar_n)

        ax.legend(
            handles=legend_handles,
            loc=chosen_legend_loc,
            fontsize=legendSize,
            frameon=True,
            framealpha=0.9,
            facecolor="white",
            edgecolor="0.5",
        )



    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(
            save_path,
            bbox_inches="tight",
            dpi=dpiValue,
            pad_inches=0.08,
        )
        print(f"[plot_usv_tracking] Saved → {save_path}")

    if show:
        plt.show()

    return fig

def plotSpeed(simTime, simData, figNo=4):
    u = get_col(simData, 6)
    v = get_col(simData, 7)
    w = get_col(simData, 8)
    U = np.sqrt(u**2 + v**2 + w**2)

    fig, ax = plt.subplots(
        num=figNo,
        figsize=(cm2inch(figSizeSingle[0]), cm2inch(figSizeSingle[1])),
        dpi=dpiValue,
    )

    ax.plot(simTime, U, linewidth=lineWidth, label="Total velocity")
    apply_axis_style(ax, xlabel="Time (s)", ylabel="Velocity (m/s)", title=f"{LABEL} total velocity")
    ax.legend(fontsize=legendSize)

    finalize_figure(fig, "speed")


def plotSurge(simTime, simData, figNo=5):
    u = get_col(simData, 6)

    fig, ax = plt.subplots(
        num=figNo,
        figsize=(cm2inch(figSizeSingle[0]), cm2inch(figSizeSingle[1])),
        dpi=dpiValue,
    )

    ax.plot(simTime, u, linewidth=lineWidth, label="Surge velocity")
    apply_axis_style(ax, xlabel="Time (s)", ylabel="u (m/s)", title=f"{LABEL} surge velocity")
    ax.legend(fontsize=legendSize)

    finalize_figure(fig, "surge")


def plotSway(simTime, simData, figNo=6):
    v = get_col(simData, 7)

    fig, ax = plt.subplots(
        num=figNo,
        figsize=(cm2inch(figSizeSingle[0]), cm2inch(figSizeSingle[1])),
        dpi=dpiValue,
    )

    ax.plot(simTime, v, linewidth=lineWidth, label="Sway velocity")
    apply_axis_style(ax, xlabel="Time (s)", ylabel="v (m/s)", title=f"{LABEL} sway velocity")
    ax.legend(fontsize=legendSize)

    finalize_figure(fig, "sway")


def plotYaw(simTime, simData, figNo=7):
    r = get_col(simData, 11)

    fig, ax = plt.subplots(
        num=figNo,
        figsize=(cm2inch(figSizeSingle[0]), cm2inch(figSizeSingle[1])),
        dpi=dpiValue,
    )

    ax.plot(simTime, r, linewidth=lineWidth, label="Yaw rate")
    apply_axis_style(ax, xlabel="Time (s)", ylabel="r (rad/s)", title=f"{LABEL} yaw rate")
    ax.legend(fontsize=legendSize)

    finalize_figure(fig, "yaw_rate")


def plotDistance(
    simTime,
    simData_or_distance,
    targetData=None,
    figNo=8,
    smooth=True,
    smooth_window_s=2.5,
    show_raw=False
):
    """
    Supports both:
        plotDistance(simTime, distanceHistory, figNo=8)
    and:
        plotDistance(simTime, simData, targetData, figNo=8)

    smooth_window_s controls smoothing in seconds.
    """

    t = np.asarray(simTime, dtype=float).reshape(-1)

    # Case 1: distance history passed directly
    if targetData is None:
        distance = np.asarray(simData_or_distance, dtype=float).reshape(-1)

    # Case 2: simData and targetData passed
    else:
        simData = np.asarray(simData_or_distance, dtype=float)
        targetData = np.asarray(targetData, dtype=float)

        n = min(len(t), simData.shape[0], targetData.shape[0])
        t = t[:n]
        simData = simData[:n]
        targetData = targetData[:n]

        usv_pos = simData[:, 0:2]
        target_pos = targetData[:, 0:2]

        distance = np.linalg.norm(target_pos - usv_pos, axis=1)

    # Match lengths
    n = min(len(t), len(distance))
    t = t[:n]
    distance = distance[:n]

    # Remove invalid values
    valid = np.isfinite(t) & np.isfinite(distance)
    t = t[valid]
    distance = distance[valid]

    if len(distance) == 0:
        print("No valid distance data to plot.")
        return

    distance_plot = distance.copy()

    if smooth and len(distance_plot) > 5:
        # Estimate sample time
        dt = np.median(np.diff(t))
        if dt <= 0 or not np.isfinite(dt):
            dt = 0.02

        # Convert smoothing window from seconds to samples
        window = int(round(smooth_window_s / dt))

        # Ensure reasonable odd window length
        window = max(window, 5)
        if window % 2 == 0:
            window += 1

        # Avoid window larger than data
        window = min(window, len(distance_plot) - 1)
        if window % 2 == 0:
            window -= 1

        if window >= 5:
            try:
                from scipy.signal import savgol_filter

                # Smooth without strong phase distortion
                polyorder = 2 if window > 7 else 1
                distance_plot = savgol_filter(
                    distance_plot,
                    window_length=window,
                    polyorder=polyorder,
                    mode="interp"
                )

            except Exception:
                # Fallback moving average
                kernel = np.ones(window) / window
                distance_plot = np.convolve(distance_plot, kernel, mode="same")

                # Avoid edge distortion
                edge = window // 2
                distance_plot[:edge] = distance[:edge]
                distance_plot[-edge:] = distance[-edge:]

    fig, ax = plt.subplots(
        num=figNo,
        figsize=(cm2inch(figSizeSingle[0]), cm2inch(figSizeSingle[1])),
        dpi=dpiValue,
    )

    if show_raw:
        ax.plot(
            t,
            distance,
            linewidth=0.5 * lineWidth,
            alpha=0.35,
            label="Raw distance"
        )

    ax.plot(
        t,
        distance_plot,
        linewidth=lineWidth,
        label="Distance to target"
    )

    apply_axis_style(
        ax,
        xlabel="Time (s)",
        ylabel="Distance (m)",
        title=f"{LABEL} distance to target"
    )

    ax.legend(fontsize=legendSize)

    finalize_figure(fig, "distance")


def plotHeadingError(simTime, simData, targetData, figNo=9):
    _, heading_error, _ = compute_tracking_errors(simData, targetData)

    fig, ax = plt.subplots(
        num=figNo,
        figsize=(cm2inch(figSizeSingle[0]), cm2inch(figSizeSingle[1])),
        dpi=dpiValue,
    )

    ax.plot(simTime, R2D(heading_error), linewidth=lineWidth, label="Heading error")
    apply_axis_style(ax, xlabel="Time (s)", ylabel="Heading error (deg)", title=f"{LABEL} heading error")
    ax.legend(fontsize=legendSize)

    finalize_figure(fig, "heading_error")


def plotPositionComponents(simTime, simData, targetData, figNo=10):
    usv_north = get_col(simData, 0)
    usv_east = get_col(simData, 1)

    target_north = targetData[:, 0]
    target_east = targetData[:, 1]

    fig, ax = plt.subplots(
        num=figNo,
        figsize=(cm2inch(figSizeSingle[0]), cm2inch(figSizeSingle[1])),
        dpi=dpiValue,
    )

    ax.plot(simTime, usv_north, linewidth=lineWidth, label="USV north")
    ax.plot(simTime, target_north, "--", linewidth=lineWidth, label="Target north")
    ax.plot(simTime, usv_east, linewidth=lineWidth, label="USV east")
    ax.plot(simTime, target_east, "--", linewidth=lineWidth, label="Target east")

    apply_axis_style(ax, xlabel="Time (s)", ylabel="Position (m)", title=f"{LABEL} position components")
    ax.legend(fontsize=legendSize)

    finalize_figure(fig, "position_components")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Loading CSV: {CSV_PATH}")
    print(f"Saving plots to: {PLOT_DIR}")

    simTime, simData, targetData = load_log(CSV_PATH, skip_first=SKIP_FIRST)

    print(f"Samples: {len(simTime)}")
    print(f"simData shape: {simData.shape}")
    print(f"targetData shape: {targetData.shape}")
    print(f"Time range: {simTime[0]:.3f} to {simTime[-1]:.3f} s")

    plots = PLOTS_TO_MAKE

    if "all" in plots:
        plots = [
            "states",
            "controls",
            "path",
            "ieee_path",
            "speed",
            "surge",
            "sway",
            "yaw",
            "distance",
            "heading",
            "position",
        ]

    fig_no = 1

    if "states" in plots:
        plotVehicleStates(simTime, simData, fig_no)
        fig_no += 1

    if "controls" in plots:
        plotControls(simTime, simData, fig_no)
        fig_no += 1

    if "path" in plots:
        plotPosTar2(simTime, simData, targetData, fig_no)
        fig_no += 1

    if "ieee_path" in plots:
        ieee_path_base = Path(PLOT_DIR) / f"{safe_label(LABEL)}_ieee_path"

        save_path = None

        if SAVE_PLOTS:
            Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

            if SAVE_PDF:
                save_path = str(ieee_path_base.with_suffix(".pdf"))
            else:
                save_path = str(ieee_path_base.with_suffix(".png"))

        plot_usv_tracking(
            sim_time=simTime,
            sim_data=simData,
            target_data=targetData,
            save_path=save_path,
            fig_number=fig_no,
            show=SHOW_PLOTS,
        )

        if SAVE_PLOTS and SAVE_PDF:
            png_path = str(ieee_path_base.with_suffix(".png"))
            plt.figure(fig_no).savefig(
                png_path,
                bbox_inches="tight",
                dpi=dpiValue,
                pad_inches=0.08,
            )
            print(f"[plot_usv_tracking] Saved → {png_path}")

        if SAVE_PLOTS and SAVE_EPS:
            eps_path = str(ieee_path_base.with_suffix(".eps"))
            plt.figure(fig_no).savefig(
                eps_path,
                format="eps",
                bbox_inches="tight",
                pad_inches=0.08,
            )
            print(f"[plot_usv_tracking] Saved → {eps_path}")

        fig_no += 1

    if "speed" in plots:
        plotSpeed(simTime, simData, fig_no)
        fig_no += 1

    if "surge" in plots:
        plotSurge(simTime, simData, fig_no)
        fig_no += 1

    if "sway" in plots:
        plotSway(simTime, simData, fig_no)
        fig_no += 1

    if "yaw" in plots:
        plotYaw(simTime, simData, fig_no)
        fig_no += 1

    if "distance" in plots:
        plotDistance(simTime, simData, targetData, fig_no)
        fig_no += 1

    if "heading" in plots:
        plotHeadingError(simTime, simData, targetData, fig_no)
        fig_no += 1

    if "position" in plots:
        plotPositionComponents(simTime, simData, targetData, fig_no)
        fig_no += 1

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    print(f"Running file: {__file__}")
    print(f"LABEL used by script: {LABEL}")
    print(f"CSV_PATH: {CSV_PATH}")
    print(f"PLOT_DIR: {PLOT_DIR}")

    print(f"Loading CSV: {CSV_PATH}")
    print(f"Saving plots to: {PLOT_DIR}")
    main()