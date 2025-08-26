import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import matplotlib
matplotlib.use('Agg')
import time
import math


class live_plotter():

    def __init__(self, otter):

        self.otter = otter

        self.fig, ax = plt.subplots(2, 3, figsize=(10,8))
        self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6 = ax.flatten()

        self.fig.suptitle("Live data")

        self.xs11 = []
        self.xs12 = []
        self.ys11 = []
        self.ys12 = []
        self.x11 = "time"
        self.x12 = "time"
        self.y11 = "tau_X"
        self.y12 = "tau_N"

        self.xs21 = []
        self.xs22 = []
        self.ys21 = []
        self.ys22 = []
        self.x21 = "east_from_observer"
        self.x22 = "target_east_from_observer"
        self.y21 = "north_from_observer"
        self.y22 = "target_north_from_observer"

        self.xs31 = []
        self.xs32 = []
        self.ys31 = []
        self.ys32 = []
        self.x31 = "time"
        self.x32 = "time"
        self.y31 = "current_orientation_3"
        self.y32 = "yaw_setpoint"

        self.xs41 = []
        self.xs42 = []
        self.ys41 = []
        self.ys42 = []
        self.x41 = "time"
        self.x42 = "time"
        self.y41 = "distance_to_target"
        self.y42 = "distance_to_target"

        self.xs51 = []
        self.xs52 = []
        self.ys51 = []
        self.ys52 = []
        self.x51 = "time"
        self.x52 = "time"
        self.y51 = "n1"
        self.y52 = "n2"

        self.start_time = 0.0

        self.plot()


    def animate(self, i):

        if self.x11 == "time" and self.x12 == "time":
            self.xs11.append(float(time.time() - self.start_time))
            self.xs12 = self.xs11

        else:
            self.xs11.append(float(self.otter.sorted_values[self.x11]))
            self.xs12.append(float(self.otter.sorted_values[self.x12]))
        self.ys11.append(float(self.otter.sorted_values[self.y11]))
        self.ys12.append(float(self.otter.sorted_values[self.y12]))

        self.ax1.clear()
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("N")
        self.ax1.plot(self.xs11, self.ys11, "r-", label=self.y11)
        self.ax1.plot(self.xs12, self.ys12, "b-", label=self.y12)
        self.ax1.legend()


        if self.x21 == "time" and self.x22 == "time":
            self.xs21.append(float(time.time() - self.start_time))
            self.xs22 = self.xs21

        else:
            self.xs21.append(float(self.otter.sorted_values[self.x21]))
            self.xs22.append(float(self.otter.sorted_values[self.x22]))
        self.ys21.append(float(self.otter.sorted_values[self.y21]))
        self.ys22.append(float(self.otter.sorted_values[self.y22]))

        self.ax2.clear()
        self.ax2.set_xlabel("East (m)")
        self.ax2.set_ylabel("North (m)")
        self.ax2.plot(self.xs21, self.ys21, "r-", label="Otter position")
        self.ax2.plot(self.xs22, self.ys22, "c-", label="Target position")
        self.ax2.legend()


        if self.x31 == "time" and self.x32 == "time":
            self.xs31.append(float(time.time() - self.start_time))
            self.xs32 = self.xs31

        else:
            self.xs31.append(float(self.otter.sorted_values[self.x31]))
            self.xs32.append(float(self.otter.sorted_values[self.x32]))
        self.ys31.append(float(self.otter.sorted_values[self.y31]))
        self.ys32.append(float(self.otter.sorted_values[self.y32] * (180/math.pi)))
        self.ax3.clear()
        self.ax3.set_xlabel("Time (s)")
        self.ax3.set_ylabel("Angle (deg)")
        self.ax3.plot(self.xs31, self.ys31, "m-", label="Current angle")
        self.ax3.plot(self.xs32, self.ys32, "y-", label="Desired angle")
        self.ax3.legend()


        if self.x41 == "time" and self.x42 == "time":
            self.xs41.append(float(time.time() - self.start_time))
            self.xs42 = self.xs41

        else:
            self.xs41.append(float(self.otter.sorted_values[self.x41]))
            self.xs42.append(float(self.otter.sorted_values[self.x42]))
        self.ys41.append(float(self.otter.sorted_values[self.y41]))
        self.ys42.append(float(self.otter.sorted_values[self.y42]))

        self.ax4.clear()
        self.ax4.set_xlabel("Time (s)")
        self.ax4.set_ylabel("Distance (m)")
        self.ax4.plot(self.xs41, self.ys41, "w-", label="Distance_to_target")
        self.ax4.plot(self.xs42, self.ys42, "k-", label="Distance_to_target")
        self.ax4.legend()


        if self.x51 == "time" and self.x52 == "time":
            self.xs51.append(float(time.time() - self.start_time))
            self.xs52 = self.xs51

        else:
            self.xs51.append(float(self.otter.sorted_values[self.x51]))
            self.xs52.append(float(self.otter.sorted_values[self.x52]))
        self.ys51.append(float(self.otter.sorted_values[self.y51]))
        self.ys52.append(float(self.otter.sorted_values[self.y52]))

        self.ax5.clear()
        self.ax5.set_xlabel("Time (s)")
        self.ax5.set_ylabel("Thruster speed (rad/s")
        self.ax5.plot(self.xs51, self.ys51, "r-", label="Left thruster")
        self.ax5.plot(self.xs52, self.ys52, "b-", label="Right thruster")
        self.ax5.legend()


    def plot(self):
        self.start_time = time.time()
        ani = animation.FuncAnimation(self.fig, self.animate, interval=200)
        plt.show()

