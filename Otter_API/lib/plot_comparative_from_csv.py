import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset



# Basic plotting configuration (adjust as needed)


def cm2inch(x):
    return x / 2.54

figSize1 = (16.0, 9.0)   # in cm
dpiValue = 300
PLOT_DIR = "plots"



# Helpers


def clean_simtime(series):
    """Convert '[0.]'-style strings to float array."""
    return (
        series.astype(str)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .astype(float)
        .to_numpy()
    )


def load_log(path, skip_first=3):
    """Read one CSV and return (simTime, simData, targetData),
    skipping the first `skip_first` samples to remove transients.
    """
    df = pd.read_csv(path)

    simTime = clean_simtime(df["simTime"])

    sim_cols = [c for c in df.columns if c.startswith("simData_")]
    sim_cols = sorted(sim_cols, key=lambda s: int(s.split("_")[1]))
    simData = df[sim_cols].to_numpy()

    tar_cols = [c for c in df.columns if c.startswith("targetData_")]
    tar_cols = sorted(tar_cols, key=lambda s: int(s.split("_")[1]))
    targetData = df[tar_cols].to_numpy()

    #  Skip first few samples (aligned) 
    if skip_first > 0:
        simTime    = simTime[skip_first:]
        simData    = simData[skip_first:, :]
        targetData = targetData[skip_first:, :]

    return simTime, simData, targetData


# Multi-file path plot with inset

def plotPosTar_multi_with_inset(datasets,
                                figNo=1,
                                zoom_box=None,
                                savePlot=False,
                                plotName="path_comparison"):

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0.0, vmax=1.0)

    plt.figure(figNo,
               figsize=(cm2inch(figSize1[0]), cm2inch(figSize1[1])),
               dpi=dpiValue)
    ax = plt.gca()
    ax.grid()

    controller_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    all_east_list = []
    all_north_list = []

    L, W = 2.0, 1.08           # USV triangle size
    n_marks = 15               # number of marker positions


    # Main axes plot

    for idx, data in enumerate(datasets):
        label      = data["label"]
        simTime    = np.asarray(data["simTime"], dtype=float)
        simData    = data["simData"]
        targetData = data["targetData"]

        # ensure same length
        n_common   = min(len(simData), len(simTime), len(targetData))
        simTime    = simTime[:n_common]
        simData    = simData[:n_common, :]
        targetData = targetData[:n_common, :]

        usv_north  = simData[:, 0]
        usv_east   = simData[:, 1]
        yaw        = simData[:, 5]

        tar_north  = targetData[:, 0]
        tar_east   = targetData[:, 1]

        # colour for this controller
        color_i = controller_colors[(idx + 1) % len(controller_colors)]

        # marker indices and yaw samples
        n_pts     = len(simTime)
        mark_idx  = np.linspace(0, n_pts - 1, n_marks + 1, dtype=int)
        psi_marks = yaw[mark_idx]

        # colour parameter 0..1 along the path for target circles
        mark_cols = np.linspace(0.0, 1.0, n_marks + 1)

        # base dataset: plot target path + circles once
        if idx == 0:
            ax.plot(tar_east, tar_north,
                    color="C0", lw=1.0, linestyle="--",
                    label="Target path (m)")

            target_patches = []
            circle_radius = 0.8
            for x, y in zip(tar_east[mark_idx], tar_north[mark_idx]):
                target_patches.append(Circle((x, y), radius=circle_radius))

            pc_target = PatchCollection(
                target_patches,
                facecolor=cmap(norm(mark_cols)),
                edgecolor="black",
                linewidth=0.7,
                zorder=3,
            )
            ax.add_collection(pc_target)

        # path line for this controller
        ax.plot(usv_east, usv_north,
                color=color_i, lw=0.5,
                label=label)

        # triangles for this controller
        patches = []
        for k, angle in zip(mark_idx, psi_marks):
            pts_body = np.array([[ L,    0.0],
                                 [-L/2, -W ],
                                 [-L/2,  W ]])
            c, s = np.cos(angle), np.sin(angle)
            R_ne = np.array([[c, -s],
                             [s,  c]])
            pts_ne = pts_body @ R_ne.T
            # swap to (East, North) for plotting
            pts_plot = np.column_stack((pts_ne[:, 1], pts_ne[:, 0]))
            pts_plot[:, 0] += usv_east[k]
            pts_plot[:, 1] += usv_north[k]
            patches.append(Polygon(pts_plot))

        pc = PatchCollection(
            patches,
            edgecolor="black",
            linewidth=0.5,
            zorder=4,
        )
        pc.set_facecolor(color_i)
        ax.add_collection(pc)

        # collect for global view box
        if idx == 0:
            all_east_list.extend([tar_east, usv_east])
            all_north_list.extend([tar_north, usv_north])
        else:
            all_east_list.append(usv_east)
            all_north_list.append(usv_north)

    # axes formatting
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", which="major", labelsize=9)

    # view box over all paths
    all_east = np.concatenate(all_east_list)
    all_north = np.concatenate(all_north_list)
    min_e, max_e = all_east.min(), all_east.max()
    min_n, max_n = all_north.min(), all_north.max()
    width  = max_e - min_e
    height = max_n - min_n
    pad    = 0.2 * max(width, height)
    ax.set_xlim(min_e - pad, max_e + pad)
    ax.set_ylim(min_n - pad, max_n + pad)

    # legend proxies
    line_handles, line_labels = ax.get_legend_handles_labels()

    target_proxy = Line2D(
        [], [], linestyle="None",
        marker="o", markersize=4,
        markerfacecolor="0.6", markeredgecolor="0.3",
        label="Target position",
    )

    usv_proxy = Line2D(
        [], [], linestyle="None",
        marker=(3, 0, 0),
        markersize=6,
        markerfacecolor="0.6", markeredgecolor="0.3",
        label="USV position & heading",
    )

    ax.legend(handles=line_handles + [target_proxy, usv_proxy],
              fontsize=5,
              loc="center left",
              bbox_to_anchor=(1.02, 0.5))

    # Inset axes

    if zoom_box is not None:
        x1, x2, y1, y2 = zoom_box
        axins = inset_axes(
            ax,
            width="50%",
            height="50%",
            loc="lower right",
            bbox_to_anchor=(0.1, 0.08, 1, 1),
            bbox_transform=ax.transAxes,
        )
        axins.grid(True)

        L, W = 2.0, 1.08      # same triangle size as main plot
        n_marks = 15

        for idx, data in enumerate(datasets):
            simTime   = np.asarray(data["simTime"], dtype=float)
            simData   = data["simData"]
            targetDat = data["targetData"]

            # ensure same length
            n_common = min(len(simData), len(targetDat), len(simTime))
            simTime   = simTime[:n_common]
            simData   = simData[:n_common, :]
            targetDat = targetDat[:n_common, :]

            t0 = simTime[0]
            tf = simTime[-1]
            common_t = np.linspace(t0, tf, n_common)

            usv_north = np.interp(common_t, simTime, simData[:, 0])
            usv_east  = np.interp(common_t, simTime, simData[:, 1])
            yaw       = np.interp(common_t, simTime, simData[:, 5])

            tar_north = np.interp(common_t, simTime, targetDat[:, 0])
            tar_east  = np.interp(common_t, simTime, targetDat[:, 1])

            color_i = controller_colors[(idx + 1) % len(controller_colors)]

            # indices for markers in inset
            n_pts_ins = len(common_t)
            mark_idx  = np.linspace(0, n_pts_ins - 1, n_marks + 1, dtype=int)
            psi_marks = yaw[mark_idx]

            # target path only once
            if idx == 0:
                axins.plot(tar_east, tar_north,
                           color="C0", lw=1.0, linestyle="--")
                target_patches_inset = []
                circle_radius = 1.0

                for x, y in zip(tar_east[mark_idx], tar_north[mark_idx]):
                    target_patches_inset.append(Circle((x, y), radius=circle_radius))

                pc_target_inset = PatchCollection(
                    target_patches_inset,
                    facecolor="0.6",     # neutral gray like legend
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=3,
                )
                axins.add_collection(pc_target_inset)

            # path line for each controller
            axins.plot(usv_east, usv_north, color=color_i, lw=0.5)

            # triangles in inset
            patches_inset = []
            for k, angle in zip(mark_idx, psi_marks):
                pts_body = np.array([[ L,    0.0],
                                     [-L/2, -W ],
                                     [-L/2,  W ]])
                c, s = np.cos(angle), np.sin(angle)
                R_ne = np.array([[c, -s],
                                 [s,  c]])
                pts_ne = pts_body @ R_ne.T
                pts_plot = np.column_stack((pts_ne[:, 1], pts_ne[:, 0]))
                pts_plot[:, 0] += usv_east[k]
                pts_plot[:, 1] += usv_north[k]
                patches_inset.append(Polygon(pts_plot))

            pc_inset = PatchCollection(
                patches_inset,
                edgecolor="black",
                linewidth=0.1,
                zorder=4,
            )
            pc_inset.set_facecolor(color_i)
            axins.add_collection(pc_inset)

        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_aspect("equal", adjustable="box")
        axins.set_xticks(np.linspace(x1, x2, 2))  # only 3 ticks on x
        axins.set_yticks(np.linspace(y1, y2, 2))  # only 3 ticks on y
        axins.tick_params(labelsize=7)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.tight_layout()

    if savePlot:
        os.makedirs(PLOT_DIR, exist_ok=True)

        png_out = os.path.join(PLOT_DIR, plotName + ".png")
        eps_out = os.path.join(PLOT_DIR, plotName + ".eps")

        print(f"Saving plot to: {png_out}")
        print(f"Saving plot to: {eps_out}")

        plt.savefig(png_out, bbox_inches="tight", dpi=300)
        plt.savefig(eps_out, format="eps", bbox_inches="tight")
        plt.show()



if __name__ == "__main__":
    data_files = [
        {"path": "logs/sim_logs/pid_sim_logs/sim_log_pid.csv",   "label": "PID"},
        {"path": "logs/sim_logs/nmpc_sim_logs/sim_log_nmpc.csv", "label": "NMPC"},
        {"path": "logs/sim_logs/drl_sim_logs/sim_log_drl.csv",   "label": "DRL"},
    ]

    # Load all logs
    raw_logs = {}
    for info in data_files:
        simTime, simData, targetData = load_log(info["path"])
        raw_logs[info["label"]] = (simTime, simData, targetData)

    # Use DRL end time as comparison horizon (no scaling)
    T_end = raw_logs["DRL"][0][-1]

    datasets = []
    for info in data_files:
        label = info["label"]
        simTime, simData, targetData = raw_logs[label]

        # Crop PID/NMPC to DRL horizon (DRL left unchanged)
        if label != "DRL":
            mask       = simTime <= T_end
            simTime    = simTime[mask]
            simData    = simData[mask]
            targetData = targetData[mask]

        datasets.append({
            "label":      label,
            "simTime":    simTime,
            "simData":    simData,
            "targetData": targetData,
        })

    plotPosTar_multi_with_inset(
        datasets,
        zoom_box=(-65, -50, -15, 30),
        savePlot=True,
        plotName="usv_path_comparison",
    )
