# -*- coding: utf-8 -*-
"""
Simulator plotting functions:

plotVehicleStates(simTime, simData, figNo) 
plotControls(simTime, simData, vehicle, figNo)
def plot3D(simData, numDataPoints, FPS, filename, figNo)

Author:     Thor I. Fossen

Modified
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from lib.gnc import ssa
import matplotlib
matplotlib.use('TkAgg')
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import os


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
PLOT_DIR = os.path.join(_PROJECT_ROOT, "logs", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

legendSize = 10  # legend size
figSize1 = [25, 13]  # figure1 size in cm
figSize2 = [25, 13]  # figure2 size in cm
dpiValue = 150  # figure dpi value

#            signals = np.append(np.append(np.append(eta, nu), u_control), u_actual)
#            simData = [(eta, nu), u_control, u_actual]


def R2D(value):  # radians to degrees
    return value * 180 / math.pi


def cm2inch(value):  # inch to cm
    return value / 2.54


# plotVehicleStates(simTime, simData, figNo) plots the 6-DOF vehicle
# position/attitude and velocities versus time in figure no. figNo
def plotVehicleStates(simTime, simData, figNo):

    # Time vector
    t = simTime

    # State vectors
    x = simData[:, 0]
    y = simData[:, 1]
    z = simData[:, 2]
    phi = R2D(ssa(simData[:, 3]))
    theta = R2D(ssa(simData[:, 4]))
    psi = R2D(ssa(simData[:, 5]))
    u = simData[:, 6]
    v = simData[:, 7]
    w = simData[:, 8]
    p = R2D(simData[:, 9])
    q = R2D(simData[:, 10])
    r = R2D(simData[:, 11])

    # Speed
    U = np.sqrt(np.multiply(u, u) + np.multiply(v, v) + np.multiply(w, w))

    beta_c  = R2D(ssa(np.arctan2(v,u)))   # crab angle, beta_c    
    alpha_c = R2D(ssa(np.arctan2(w,u)))   # flight path angle
    chi = R2D(ssa(simData[:, 5] + np.arctan2(v, u)))  # course angle, chi=psi+beta_c

    # Plots
    plt.figure(
        figNo, figsize=(cm2inch(figSize1[0]), cm2inch(figSize1[1])), dpi=dpiValue
    )
    plt.grid()

    plt.subplot(3, 3, 1)
    plt.plot(y, x)
    plt.legend(["North-East positions (m)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 2)
    plt.plot(t, z)
    plt.legend(["Depth (m)"], fontsize=legendSize)
    plt.grid()

    plt.title("Vehicle states", fontsize=12)

    plt.subplot(3, 3, 3)
    plt.plot(t, phi, t, theta)
    plt.legend(["Roll angle (deg)", "Pitch angle (deg)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 4)
    plt.plot(t, U)
    plt.legend(["Speed (m/s)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 5)
    plt.plot(t, chi)
    plt.legend(["Course angle (deg)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 6)
    plt.plot(t, theta, t, alpha_c)
    plt.legend(["Pitch angle (deg)", "Flight path angle (deg)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 7)
    plt.plot(t, u, t, v, t, w)
    plt.xlabel("Time (s)", fontsize=12)
    plt.legend(
        ["Surge velocity (m/s)", "Sway velocity (m/s)", "Heave velocity (m/s)"],
        fontsize=legendSize,
    )
    plt.grid()

    plt.subplot(3, 3, 8)
    plt.plot(t, p, t, q, t, r)
    plt.xlabel("Time (s)", fontsize=12)
    plt.legend(
        ["Roll rate (deg/s)", "Pitch rate (deg/s)", "Yaw rate (deg/s)"],
        fontsize=legendSize,
    )
    plt.grid()

    plt.subplot(3, 3, 9)
    plt.plot(t, psi, t, beta_c)
    plt.xlabel("Time (s)", fontsize=12)
    plt.legend(["Yaw angle (deg)", "Crab angle (deg)"], fontsize=legendSize)
    plt.grid()


# plotControls(simTime, simData) plots the vehicle control inputs versus time
# in figure no. figNo denormalize
def plotControls(simTime, simData, vehicle, figNo):
    DOF = 6
    t = simTime

    # Extract all control and actual signals (for scaling)
    u_cmd_all = simData[:, 2 * DOF : 2 * DOF + vehicle.dimU]
    u_act_all = simData[:, 2 * DOF + vehicle.dimU : 2 * DOF + 2 * vehicle.dimU]

    # Global min/max for all controls (for consistent scaling)
    global_min = min(u_cmd_all.min(), u_act_all.min())
    global_max = max(u_cmd_all.max(), u_act_all.max())
    span = global_max - global_min if global_max > global_min else 1.0
    pad = 0.1 * span

    col = 2
    row = int(math.ceil(vehicle.dimU / col))

    fig, axs = plt.subplots(
        row,
        col,
        num=figNo,
        figsize=(cm2inch(figSize2[0]), cm2inch(figSize2[1])),
        dpi=dpiValue,
        sharex=True,
    )

    axs = np.atleast_1d(axs).ravel()

    for i in range(vehicle.dimU):
        ax = axs[i]

        u_control = u_cmd_all[:, i]
        u_actual  = u_act_all[:, i]

        # auto-rescale commands if they normalized 
        ctrl_span = u_control.max() - u_control.min()
        act_span  = u_actual.max() - u_actual.min()
        if ctrl_span > 0 and act_span > 0:
            # "normalized" if tiny span compared to actual and roughly within [-1, 1]
            if ctrl_span < 3.0 and act_span > 5.0 * ctrl_span and u_control.min() >= -1.5 and u_control.max() <= 1.5:
                # map [ctrl_min, ctrl_max] -> [act_min, act_max] for plotting
                cmin, cmax = u_control.min(), u_control.max()
                amin, amax = u_actual.min(),  u_actual.max()
                # avoid zeor div
                if cmax > cmin:
                    u_control = (u_control - cmin) / (cmax - cmin)  # 0..1
                    u_control = u_control * (amax - amin) + amin    # same scale as actual
     

        if "deg" in vehicle.controls[i]:
            u_control = R2D(u_control)
            u_actual  = R2D(u_actual)

        ax.plot(t, u_control, label=vehicle.controls[i] + ", command", linewidth=0.8)
        ax.plot(t, u_actual,  label=vehicle.controls[i] + ", actual",  linewidth=0.8)

        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(global_min - pad, global_max + pad)

        if i >= (vehicle.dimU - col):
            ax.set_xlabel("Time (s)", fontsize=12)

        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(True)
        ax.legend(fontsize=9, loc="upper right")

    for j in range(vehicle.dimU, len(axs)):
        axs[j].set_visible(False)

    fig.tight_layout(pad=0.3)



# plot3D(simData,numDataPoints,FPS,filename,figNo) plots the vehicles position (x, y, z) in 3D
# in figure no. figNo
def plot3D(simData, numDataPoints, FPS, filename, figNo):
    # State vectors
    x = simData[:, 0]
    y = simData[:, 1]
    z = simData[:, 2]

    # down-sampling the xyz data points
    N = y[::len(x) // numDataPoints];
    E = x[::len(x) // numDataPoints];
    D = z[::len(x) // numDataPoints];

    # Animation function
    def anim_function(num, dataSet, line):
        line.set_data(dataSet[0:2, :num])
        line.set_3d_properties(dataSet[2, :num])
        ax.view_init(elev=50.0, azim=-120.0)

        return line

    dataSet = np.array([N, E, -D])  # Down is negative z

    # Attaching 3D axis to the figure
    fig = plt.figure(figNo, figsize=(cm2inch(figSize1[0]), cm2inch(figSize1[1])),
                     dpi=dpiValue)
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # Line/trajectory plot
    line = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=2, c='b')[0]

    # Setting the axes properties
    ax.set_xlabel('X / East')
    ax.set_ylabel('Y / North')
    ax.set_zlim3d([-100, 20])  # default depth = -100 m

    if np.amax(z) > 100.0:
        ax.set_zlim3d([-np.amax(z), 20])

    ax.set_zlabel('-Z / Down')

    # Plot 2D surface for z = 0
    [x_min, x_max] = ax.get_xlim()
    [y_min, y_max] = ax.get_ylim()
    x_grid = np.arange(x_min - 20, x_max + 20)
    y_grid = np.arange(y_min - 20, y_max + 20)
    [xx, yy] = np.meshgrid(x_grid, y_grid)
    zz = 0 * xx
    ax.plot_surface(xx, yy, zz, alpha=0.3)

    # Title of plot
    ax.set_title('North-East-Down')

    # Create the animation object
    ani = animation.FuncAnimation(fig,
                                  anim_function,
                                  frames=numDataPoints,
                                  fargs=(dataSet, line),
                                  interval=200,
                                  blit=False,
                                  repeat=True)

    # Save the 3D animation as a gif file
    ani.save(filename, writer=animation.PillowWriter(fps=FPS))


def plot2D(simData, numDataPoints, FPS, filename, figNo, targetData, figSize1=(10, 10), dpiValue=400):
    # Ensure target data is processed correctly
    targetData = targetData[:-1]
    tar_x = targetData[:, 1]
    tar_y = targetData[:, 0]

    # State vectors
    x = simData[:, 0]
    y = simData[:, 1]

    # Down-sampling the xyz data points
    indices = np.linspace(0, len(x) - 1, numDataPoints).astype(int)
    N = x[indices]
    E = y[indices]

    tarindices = np.linspace(0, len(tar_x) - 1, numDataPoints).astype(int)
    tx = tar_x[tarindices]
    ty = tar_y[tarindices]
    dataSet = np.array([E, N])
    targetDataSet = np.array([tx, ty])


    # Animation function
    def anim_function(num, dataSet, line, targetDataSet, target_line):
        line.set_data(dataSet[0, :num], dataSet[1, :num])
        target_line.set_data(targetDataSet[0, :num], targetDataSet[1, :num])
        return line, target_line

    # Attaching 2D axis to the figure
    fig, ax = plt.subplots(figsize=(cm2inch(figSize1[0]), cm2inch(figSize1[1])), dpi=dpiValue)

    # Line/trajectory plot
    line, = ax.plot([], [], lw=2, c='b', label='Ship Path')
    target_line, = ax.plot([], [], lw=2, c='r', linestyle='--', label='Target Path')

    # Setting the axes properties
    ax.set_xlabel('X / East')
    ax.set_ylabel('Y / North')
    ax.set_title('North-East positions')
    ax.legend()

    # Initialize the plot limits
    ax.set_xlim(np.min(E) - 10, np.max(E) + 10)
    ax.set_ylim(np.min(N) - 10, np.max(N) + 10)

    # Initialize lines
    def init():
        line.set_data([], [])
        target_line.set_data([], [])
        return line, target_line

    # Create the animation object
    ani = animation.FuncAnimation(fig,
                                  anim_function,
                                  init_func=init,
                                  frames=numDataPoints,
                                  fargs=(dataSet, line, targetDataSet, target_line),
                                  interval=300 // FPS,
                                  blit=True,
                                  repeat=True)

    # Save the 2D animation as a gif file
    ani.save(filename, writer=animation.PillowWriter(fps=FPS))



def plotPosTar(simTime, simData, figNo, targetData):
    usv_x = simData[:, 0]
    usv_y = simData[:, 1]

    targetData = targetData[1:-1]
    tar_x = targetData[:, 1]
    tar_y = targetData[:, 0]

    plt.figure(
        figNo, figsize=(cm2inch(figSize1[0]), cm2inch(figSize1[1])), dpi=dpiValue
    )
    plt.grid()

    plt.plot(usv_y, usv_x, tar_x, tar_y, "--")
    plt.legend(["North-East positions (m)", "Target position (m)"], fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid()


def plotSpeed(simTime, simData, figNo):
    x = simTime

    u = simData[:, 6]
    v = simData[:, 7]
    w = simData[:, 8]

    U = np.sqrt(np.multiply(u, u) + np.multiply(v, v) + np.multiply(w, w))

    plt.figure(
        figNo, figsize=(cm2inch(figSize1[0]), cm2inch(figSize1[1])), dpi=dpiValue
    )
    plt.grid()

    plt.plot(x, U, linewidth=0.8)
    plt.legend(["Total velocity (m/s)"], fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)

def plotSurge(simTime, simData, figNo):
    x = simTime

    u = simData[:, 6]


    plt.figure(
        figNo, figsize=(cm2inch(figSize1[0]), cm2inch(figSize1[1])), dpi=dpiValue
    )
    plt.grid()

    plt.plot(x, u, linewidth=0.8)
    plt.legend(["Surge velocity (m/s)"], fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)

#            simData = [(eta, nu), u_control, u_actual]
# eta 0-5
# nu 6-11
def plotYaw(simTime, simData, figNo):
    x = simTime

    r = simData[:, 11]


    plt.figure(
        figNo, figsize=(cm2inch(figSize1[0]), cm2inch(figSize1[1])), dpi=dpiValue
    )
    plt.grid()

    plt.plot(x, r, linewidth=0.8)
    plt.legend(["Yaw rate (rad/s)"], fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)




def plotPosTar2(simTime, simData, figNo, targetData, savePlot=False, plotName="test_path"):
    
    
    n_marks   = 10 # how many USV/target marks 
    
    targetData = targetData[1:-1]

    n_common = min(len(simData), len(targetData), len(simTime))
    simTime   = simTime[:n_common]
    usv_north = simData[:n_common, 0]
    usv_east  = simData[:n_common, 1]
    yaw       = simData[:n_common, 5]
    tar_north = targetData[:n_common, 0]
    tar_east  = targetData[:n_common, 1]

    t_norm = ((simTime - simTime[0]) / (simTime[-1] - simTime[0])).ravel()

    
    mark_idx  = np.linspace(0, n_common - 1, n_marks + 1, dtype=int)
    mark_cols = t_norm[mark_idx]
    psi_marks = yaw[mark_idx]

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0.0, vmax=1.0)

    plt.figure(figNo,
               figsize=(cm2inch(figSize1[0]), cm2inch(figSize1[1])),
               dpi=dpiValue)
    ax = plt.gca()
    ax.grid()

    # thin paths (these provide the line labels)
    ax.plot(tar_east, tar_north,
            color="C0", lw=1.0, linestyle="--",
            label="Target path (m)")
    ax.plot(usv_east, usv_north,
            color="C1", lw=1.0,
            label="USV path (m)")

    # target circles (data-space)
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

    # USV triangles (data-space)
    L, W = 2.0, 1.08
    patches = []
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

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", which="major", labelsize=14)

    # normalised viewbox
    all_east  = np.concatenate([tar_east, usv_east])
    all_north = np.concatenate([tar_north, usv_north])
    min_e, max_e = all_east.min(),  all_east.max()
    min_n, max_n = all_north.min(), all_north.max()
    width  = max_e - min_e
    height = max_n - min_n
    pad    = 0.2 * max(width, height)
    ax.set_xlim(min_e - pad, max_e + pad)
    ax.set_ylim(min_n - pad, max_n + pad)

    
    line_handles, line_labels = ax.get_legend_handles_labels()

    target_proxy = Line2D(
        [], [], linestyle="None",
        marker="o", markersize=8,
        markerfacecolor="0.6", markeredgecolor="0.3",
        label="Target position",
    )

    usv_proxy = Line2D(
        [], [], linestyle="None",
        marker=(3, 0, 0),
        markersize=10,
        markerfacecolor="0.6", markeredgecolor="0.3",
        label="USV position & heading",
    )

    ax.legend(handles=line_handles + [target_proxy, usv_proxy],
              fontsize=10,
              loc="center left",
              bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    
    if savePlot:
        outfile = os.path.join(PLOT_DIR, plotName + ".png")
        print(f"Saving plot to: {outfile}")
        plt.savefig(outfile, bbox_inches='tight')