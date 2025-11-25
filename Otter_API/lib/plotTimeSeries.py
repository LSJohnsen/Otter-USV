# -*- coding: utf-8 -*-
"""
Simulator plotting functions:

plotVehicleStates(simTime, simData, figNo) 
plotControls(simTime, simData, vehicle, figNo)
def plot3D(simData, numDataPoints, FPS, filename, figNo)

Author:     Thor I. Fossen
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

legendSize = 10  # legend size
figSize1 = [25, 13]  # figure1 size in cm
figSize2 = [25, 13]  # figure2 size in cm
dpiValue = 150  # figure dpi value


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
# in figure no. figNo
def plotControls(simTime, simData, vehicle, figNo):

    DOF = 6

    # Time vector
    t = simTime

    plt.figure(
        figNo, figsize=(cm2inch(figSize2[0]), cm2inch(figSize2[1])), dpi=dpiValue
    )

    # Columns and rows needed to plot vehicle.dimU control inputs
    col = 2
    row = int(math.ceil(vehicle.dimU / col))

    # Plot the vehicle.dimU active control inputs
    for i in range(0, vehicle.dimU):

        u_control = simData[:, 2 * DOF + i]  # control input, commands
        u_actual = simData[:, 2 * DOF + vehicle.dimU + i]  # actual control input

        if vehicle.controls[i].find("deg") != -1:  # convert angles to deg
            u_control = R2D(u_control)
            u_actual = R2D(u_actual)

        plt.subplot(row, col, i + 1)
        plt.plot(t, u_control, t, u_actual)
        plt.legend(
            [vehicle.controls[i] + ", command", vehicle.controls[i] + ", actual"],
            fontsize=14,
        )
        plt.xlabel("Time (s)", fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.grid()


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

    plt.plot(x, U)
    plt.legend(["Speed (m/s)"], fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid()



from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection

def plotPosTar2(simTime, simData, figNo, targetData, savePlot=False, plotName="test_path"):
    targetData = targetData[1:-1]

    n_common = min(len(simData), len(targetData), len(simTime))
    simTime   = simTime[:n_common]
    usv_north = simData[:n_common, 0]
    usv_east  = simData[:n_common, 1]
    yaw       = simData[:n_common, 5]
    tar_north = targetData[:n_common, 0]
    tar_east  = targetData[:n_common, 1]

    t_norm = ((simTime - simTime[0]) / (simTime[-1] - simTime[0])).ravel()

    n_marks   = 15
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

    # -------- legend: lines + proxies --------
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
    
    if savePlot == True:
        plt.savefig("logs/plots/" + plotName + ".png",bbox_inches='tight')
    