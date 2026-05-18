import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
# matplotlib.use('Agg')
import time
import math


class live_plotter():

    def __init__(self, otter):

        self.otter = otter

        self.fig, ax = plt.subplots(2, 3, figsize=(10, 8))
        self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6 = ax.flatten()

        self.fig.suptitle("Live data")

        # -------------------------
        # Plot 1: Forces / moments
        # -------------------------
        self.xs11 = []
        self.xs12 = []
        self.ys11 = []
        self.ys12 = []
        self.x11 = "time"
        self.x12 = "time"
        self.y11 = "tau_X"
        self.y12 = "tau_N"

        # -------------------------
        # Plot 2: USV / target path
        # -------------------------
        self.xs21 = []
        self.xs22 = []
        self.ys21 = []
        self.ys22 = []

        self.x21 = "east_from_observer"
        self.x22 = "target_east_from_observer"
        self.y21 = "north_from_observer"
        self.y22 = "target_north_from_observer"

        # Store initial positions so the path is plotted relative to start
        self.usv_path_origin = None       # (east0, north0)
        self.target_path_origin = None    # (target_east0, target_north0)

        # Minimum visible span in meters.
        # Lower value = more zoomed in on small movement.
        self.min_path_span = 0.5

        # -------------------------
        # Plot 3: Yaw angle
        # -------------------------
        self.xs31 = []
        self.xs32 = []
        self.ys31 = []
        self.ys32 = []
        self.x31 = "time"
        self.x32 = "time"
        self.y31 = "current_orientation_3"
        self.y32 = "yaw_setpoint"

        # -------------------------
        # Plot 4: Distance to target
        # -------------------------
        self.xs41 = []
        self.xs42 = []
        self.ys41 = []
        self.ys42 = []
        self.x41 = "time"
        self.x42 = "time"
        self.y41 = "distance_to_target"
        self.y42 = "distance_to_target"

        # -------------------------
        # Plot 5: Controller commands
        # -------------------------
        self.xs51 = []
        self.xs52 = []
        self.ys51 = []
        self.ys52 = []
        self.x51 = "time"
        self.x52 = "time"
        self.y51 = "controller_X_cmd"
        self.y52 = "controller_N_cmd"

        self.start_time = 0.0

        self.plot()

    def animate(self, i):

        current_time = float(time.time() - self.start_time)

        # ============================================================
        # Plot 1: tau_X and tau_N
        # ============================================================

        if self.x11 == "time" and self.x12 == "time":
            self.xs11.append(current_time)
            self.xs12.append(current_time)
        else:
            self.xs11.append(float(self.otter.sorted_values.get(self.x11, 0.0)))
            self.xs12.append(float(self.otter.sorted_values.get(self.x12, 0.0)))

        self.ys11.append(float(self.otter.sorted_values.get(self.y11, 0.0)))
        self.ys12.append(float(self.otter.sorted_values.get(self.y12, 0.0)))

        self.ax1.clear()
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("tau_X [N], tau_N [Nm]")
        self.ax1.plot(self.xs11, self.ys11, "r-", label=self.y11)
        self.ax1.plot(self.xs12, self.ys12, "b-", label=self.y12)
        self.ax1.grid(True)
        self.ax1.legend()

           # ============================================================
        # Plot 2: USV and target path relative to initial USV position
        # ============================================================

        east = float(self.otter.sorted_values.get(self.x21, 0.0))
        north = float(self.otter.sorted_values.get(self.y21, 0.0))

        target_east = float(self.otter.sorted_values.get(self.x22, 0.0))
        target_north = float(self.otter.sorted_values.get(self.y22, 0.0))

        # Set first USV sample as the shared map origin.
        # Both USV and target are plotted in the same local coordinate frame.
        if self.usv_path_origin is None:
            self.usv_path_origin = (east, north)

        east0, north0 = self.usv_path_origin

        # Relative position from initial USV position
        rel_east = east - east0
        rel_north = north - north0

        rel_target_east = target_east - east0
        rel_target_north = target_north - north0

        self.xs21.append(rel_east)
        self.ys21.append(rel_north)

        self.xs22.append(rel_target_east)
        self.ys22.append(rel_target_north)

        self.ax2.clear()
        self.ax2.set_xlabel("East from USV start (m)")
        self.ax2.set_ylabel("North from USV start (m)")

        # USV path and current position
        self.ax2.plot(self.xs21, self.ys21, "r-", label="Otter path")
        self.ax2.plot(self.xs21[-1], self.ys21[-1], "ro", label="Otter current")

        # Target path and current/static position
        self.ax2.plot(self.xs22, self.ys22, "c--", label="Target path")
        self.ax2.plot(
            self.xs22[-1],
            self.ys22[-1],
            "c*",
            markersize=12,
            label="Target current"
        )

        # Line of sight from USV to target
        self.ax2.plot(
            [self.xs21[-1], self.xs22[-1]],
            [self.ys21[-1], self.ys22[-1]],
            "k:",
            label="USV-target line"
        )

        # Start marker
        self.ax2.plot(0.0, 0.0, "ks", markersize=6, label="USV start")

        # Equal scale for East and North so movement is not distorted
        self.ax2.set_aspect("equal", adjustable="box")

        all_x = self.xs21 + self.xs22 + [0.0]
        all_y = self.ys21 + self.ys22 + [0.0]

        x_min = min(all_x)
        x_max = max(all_x)
        y_min = min(all_y)
        y_max = max(all_y)

        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)

        span = max(
            x_max - x_min,
            y_max - y_min,
            self.min_path_span
        )

        margin = 0.25 * span
        half_span = 0.5 * span + margin

        self.ax2.set_xlim(x_center - half_span, x_center + half_span)
        self.ax2.set_ylim(y_center - half_span, y_center + half_span)

        self.ax2.grid(True)
        self.ax2.legend(loc="best")

        # ============================================================
        # Plot 3: Current yaw and desired yaw
        # ============================================================

        if self.x31 == "time" and self.x32 == "time":
            self.xs31.append(current_time)
            self.xs32.append(current_time)
        else:
            self.xs31.append(float(self.otter.sorted_values.get(self.x31, 0.0)))
            self.xs32.append(float(self.otter.sorted_values.get(self.x32, 0.0)))

        current_yaw = float(self.otter.sorted_values.get(self.y31, 0.0))
        yaw_setpoint_rad = float(self.otter.sorted_values.get(self.y32, 0.0))

        # Assumes current_orientation_3 is already in degrees.
        # If it is actually radians, change current_yaw to:
        # current_yaw = current_yaw * (180.0 / math.pi)
        self.ys31.append(current_yaw)
        self.ys32.append(yaw_setpoint_rad * (180.0 / math.pi))

        self.ax3.clear()
        self.ax3.set_xlabel("Time (s)")
        self.ax3.set_ylabel("Angle (deg)")
        self.ax3.plot(self.xs31, self.ys31, "m-", label="Current angle")
        self.ax3.plot(self.xs32, self.ys32, "y-", label="Desired angle")
        self.ax3.grid(True)
        self.ax3.legend()

        # ============================================================
        # Plot 4: Distance to target
        # ============================================================

        if self.x41 == "time" and self.x42 == "time":
            self.xs41.append(current_time)
            self.xs42.append(current_time)
        else:
            self.xs41.append(float(self.otter.sorted_values.get(self.x41, 0.0)))
            self.xs42.append(float(self.otter.sorted_values.get(self.x42, 0.0)))

        y41_val = float(self.otter.sorted_values.get(self.y41, 0.0))

        self.ys41.append(y41_val)

        self.ax4.clear()
        self.ax4.set_xlabel("Time (s)")
        self.ax4.set_ylabel("Distance (m)")
        self.ax4.plot(self.xs41, self.ys41, "k-", label="Distance to target")
        self.ax4.grid(True)
        self.ax4.legend()

        # ============================================================
        # Plot 5: Controller commands
        # ============================================================

        if self.x51 == "time" and self.x52 == "time":
            self.xs51.append(current_time)
            self.xs52.append(current_time)
        else:
            self.xs51.append(float(self.otter.sorted_values.get(self.x51, 0.0)))
            self.xs52.append(float(self.otter.sorted_values.get(self.x52, 0.0)))

        y51_val = float(self.otter.sorted_values.get(self.y51, 0.0))
        y52_val = float(self.otter.sorted_values.get(self.y52, 0.0))

        self.ys51.append(y51_val)
        self.ys52.append(y52_val)

        self.ax5.clear()
        self.ax5.set_xlabel("Time (s)")
        self.ax5.set_ylabel("PMARMAN command [-1, 1]")
        self.ax5.plot(self.xs51, self.ys51, "r-", label="X surge command")
        self.ax5.plot(self.xs52, self.ys52, "b-", label="N yaw command")
        self.ax5.set_ylim(-1.1, 1.1)
        self.ax5.grid(True)
        self.ax5.legend()

        # ============================================================
        # Plot 6: Optional / unused
        # ============================================================

        self.ax6.clear()
        self.ax6.axis("off")

    def plot(self):
        self.start_time = time.time()
        self.ani = animation.FuncAnimation(
            self.fig,
            self.animate,
            interval=200,
            cache_frame_data=False
        )
        plt.tight_layout()
        plt.show()