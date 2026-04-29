
import Otter_api
import lib.PID_Controller_test_v2 as PID_Controller_test_v2
from lib.PID_Controller_v3 import PIDController, SurgePIDAdapter, YawPIDAdapter
import Otter_simulator
import lib.Live_guidance as Live_guidance
import lib.Live_plotter as Live_plotter
from lib.plotTimeSeries import *
import threading
import atexit
import time
import requests

from casadi_control.MPC_control import NMPCControl
from casadi_control.lib.usv_params import usv_params_6dof
from casadi_control.model_3dof import Otter3DOF

from lib.Live_DRL import LiveDRLController

from logs.IO import log_to_csv


##########################################################################################################################################################
#                                                                      OPTIONS                                                                           #
#    For testing of the Otter USV. First options determine simulation parameters, secondary options either for UGPS or live tracking with sim target     #                                                                                                                                                                                                                 #
#          Ensure the correct IP and port are used for both UGPS, and Otter USV Radio/wifi depending method. Check in otter VCS                          #  
#                                     For WiFi, make sure every laptop in process is connected to otternet45 Wifi                                        #                                                                                                               
##########################################################################################################################################################



N = 13333                                                                                               # Number of simulation samples
N = 7500
sampleTime = 0.02                                                                                       # Simulation time per sample. Usually at 0.02, other values could cause instabillity in the simulation
use_target_coordinates = False                                                                          # To use coordinates as a target or to use a linear path
use_moving_target = False                                                                               # To use moving target instead of target list (path following)
target_list = [[0, 10000]]                                                                              # List of targets to use if use_target_coordinates is set to True
end_when_last_target_reached = True                                                                     # Ends the simulation when the final target is reached
moving_target_start = [0, -20]                                                                          # Start point of the moving target if use_moving_target is set to True
moving_target_increase = [0.5, 0.0]                                                                     # Movement of the moving target each second
target_radius = 0.2                                                                                     # Radius from center of target that counts as target reached, change this depending on the complete size of the run. Very low values causes instabillity
verbose = True                                                                                          # Enable verbose printing
log_simulation = False                                                                                 # Enable verbose for logging sim
store_force_file = False                                                                                # Store the simulated control forces in a .csv file
circular_target = False                                                                                  # Make the moving target a circle in the simulation
animate_path = False                                                                                    # This takes a lot of time! File stored as 2D_animation.gif

# NMPC
N_horizon   = 15                                                                                        # Prediction Horizon e.g. (less than 30 generally for fast updates)
control_dt  = 0.1                                                                                       # MPC update period (s)

# Disturbances
wave_disturbance = True
wind_disturbance = True

# Third order target reference model
# MAKE SURE THIS IS NEGATIVE FOR DRL AND STATIONARY DYNAMIC TRACKING
third_order_ref = False



# When connecting to live otter:
# Otter USV IPs (radio: 196.168.53.2 : 2009) (WiFi: 10.0.5.1 : 2009)

ip = "192.168.53.2"
ip = "10.0.5.1"
port = 2009
#port = 32001

# UGPS ip
BASE_URL = "http://192.168.2.94/"  
URL_GLOBAL   = BASE_URL + "/api/v1/position/global"                                                     
URL_ACOUSTIC = BASE_URL + "/api/v1/position/acoustic/filtered"

# path options
start_north = -5                                                                                       # Target north position from referance point
start_east = -5                                                                                        # Target east position from referance point
v_north = 0                                                                                             # Moving target speed north (m/s)
v_east = -1.5                                                                                           # Moving target speed east (m/s)
radius = 10                                                                                             # If tracking a circular motion
v_circle = 1.5                                                                                          # Angular velocity (m/s)
side_length = 50                                                                                        # Square tracking side length
side_target_speed = 1                                                                                   # Speed of square target
enable_live_plot = True                                                                                 # Enables live plotting
ugps_stop_event = None                                                                                  # For threading stop event


parameter_list = 4 # Tuning parameters, 1 for trial and error, 2 for pole placement wb = 0.5, and 3 for pole placement wb = 0.4, 
                   # 5 no gains for testing disturbance behavior


trial_and_error_parameters = {"surge_kp" : 12, "surge_ki" : 0.7, "surge_kd" : 0, "yaw_kp" : 37, "yaw_ki" : 4, "yaw_kd" : 8}
pp_05 = {"surge_kp" : 22.48, "surge_ki" : 3.92, "surge_kd" : 11.62, "yaw_kp" : 23.72, "yaw_ki" : 4.13, "yaw_kd" : 15.08}
pp_04 = {"surge_kp" : 14.39, "surge_ki" : 3.13, "surge_kd" : 0, "yaw_kp" : 15.21, "yaw_ki" : 0.7, "yaw_kd" : 1.86}
pp_04 = {"surge_kp" : 14.39, "surge_ki" : 25.13, "surge_kd" : 1, "yaw_kp" : 25.21, "yaw_ki" : 0.7, "yaw_kd" : 1.86} # third order trjacectory (>integral surge)
pd_stationkeeping = {"surge_kp" : 50.39, "surge_ki" : 0, "surge_kd" : 50, "yaw_kp" : 30.21, "yaw_ki" : 0.7, "yaw_kd" : 0} # surge: kp 50, disturbance kp60
test_disturbance = {"surge_kp" : 0, "surge_ki" : 0, "surge_kd" : 0, "yaw_kp" : 0, "yaw_ki" : 0, "yaw_kd" : 0}

if parameter_list == 1:
    pdi = trial_and_error_parameters
elif parameter_list == 2:
    pdi = pp_05
elif parameter_list == 3:
    pdi = pp_04
elif parameter_list == 4:
    pdi = pd_stationkeeping
elif parameter_list == 5:
    pdi = test_disturbance


# DRL agent path
DRL_MODEL_PATH = "DRL_control\ppo_saves\stationary_tracking\ppo_otter_checkpoint_station_fin.zip"
DRL_VECNORM_PATH = "DRL_control\ppo_saves\stationary_tracking\ppo_otter_checkpoint_vecnormalize_station_fin.pkl"

#############################################################################################################################################################################################################################################################
#                                                                                                                                                                                                                                                           #
#                                                                                                                                                                                                                                                           #
#                                                                                                               MAIN CODE                                                                                                                            #
#                                                                                                                                                                                                                                                           #
#                                                                                                                                                                                                                                                           #
#############################################################################################################################################################################################################################################################




otter = Otter_api.otter()                                                                                                                                                                                                          # Creates Otter object from the API
simulator = Otter_simulator.otter_simulator(target_list, 
                                            use_target_coordinates, 
                                            target_radius, 
                                            use_moving_target, 
                                            moving_target_start, 
                                            moving_target_increase, 
                                            end_when_last_target_reached, 
                                            verbose, 
                                            store_force_file, 
                                            circular_target,
                                            use_waves=wave_disturbance,
                                            use_wind=wind_disturbance)                                            # Creates Simulator object




otter.controls = ["Left propeller shaft speed (rad/s)", "Right propeller shaft speed (rad/s)"]          # Some values needed for the plotting
otter.dimU = len(otter.controls)        
                                              

numDataPoints = 830                                                                                     # number of 3D data points
FPS = 60                                                                                                # frames per second (animated GIF)
filename = '3D_animation.gif'                                                                           # data file for animated GIF
browser = 'chrome'                                                                                      # browser for visualization of animated GIF




surge_kp = pdi["surge_kp"]                                                                                              #
surge_ki = pdi["surge_ki"]                                                                                              # Surge PID controller values
surge_kd = pdi["surge_kd"]                                                                                              #

yaw_kp = pdi["yaw_kp"]                                                                                                  #
yaw_ki = pdi["yaw_ki"]                                                                                                  # Yaw PID controller values
yaw_kd = pdi["yaw_kd"]                                                                                                  #


# PID
#surge_PID = PID_Controller_test_v2.PIDController(surge_kp, surge_ki, surge_kd)                                  # Surge PID object
#yaw_PID = PID_Controller_test_v2.PIDController(yaw_kp, yaw_ki, yaw_kd)                                          # Yaw PID object
# pid testing v3

pid = PIDController(
    kp_surge=surge_kp, ki_surge=surge_ki, kd_surge=surge_kd,
    kp_yaw=yaw_kp,   ki_yaw=yaw_ki,   kd_yaw=yaw_kd,
    Imax_surge=10.0,
    Imax_yaw=40.0
)
surge_PID = SurgePIDAdapter(pid)
yaw_PID   = YawPIDAdapter(pid)

# Live guidance object
live_guidance = Live_guidance.live_guidance(ip, port, surge_PID, yaw_PID, target_radius, otter, third_order_ref=third_order_ref)  

# Live DRL agent object
live_drl = LiveDRLController(live_guidance=live_guidance, model_path=DRL_MODEL_PATH, vecnormalize_path=DRL_VECNORM_PATH, scale_action=True)

#initialize nmpc - model_3dof.py transforms 6dof to 3dof by matrix reduction
otter_6dof_params = usv_params_6dof()
otter_3dof = Otter3DOF(otter_6dof_params)
nmpc = NMPCControl(
    f=otter_3dof,     
    N=N_horizon,
    sampleTime=control_dt,
)
if use_moving_target == False:
    nmpc.set_mode("stationkeeping")

print("Welcome to the Otter controller simulator and socket")

try:
    main_option = int(input("Choose environment: simulation (1), live Otter (2): "))
except ValueError:
    print("Invalid option, running simulation.")
    main_option = 1

try:
    ctrl_option = int(input("Choose controller: PID (1), NMPC (2), DRL (3): "))
except ValueError:
    print("Invalid controller, applying PID.")
    ctrl_option = 1

use_nmpc = ctrl_option == 2
use_drl = ctrl_option == 3

live_source_option = None
path_option = None

if main_option == 2:
    try:
        live_source_option = int(input("Choose target source: simulated target (1), UGPS target (2): "))
    except ValueError:
        print("Invalid target source, using simulated target.")
        live_source_option = 1

    if live_source_option == 1:
        try:
            path_option = int(input("Choose path: straight (1), circular (2), square (3), stationary (4): "))
        except ValueError:
            print("Invalid path, using stationary.")
            path_option = 4

live_guidance = Live_guidance.live_guidance(
    ip=ip,
    port=port,
    surge_PID=surge_PID,
    yaw_PID=yaw_PID,
    surge_setpoint=target_radius,
    otter=otter,
    nmpc=nmpc,
    use_nmpc=use_nmpc,
    control_dt=control_dt,
    third_order_ref=third_order_ref
)

live_drl = LiveDRLController(
    live_guidance=live_guidance,
    model_path=DRL_MODEL_PATH,
    vecnormalize_path=DRL_VECNORM_PATH,
    scale_action=True
)


def run_live_straight():
    if use_drl:
        live_drl.straight_tracking(start_north, start_east, v_north, v_east)
    else:
        live_guidance.target_tracking(start_north, start_east, v_north, v_east)


def run_live_circular():
    if use_drl:
        live_drl.circular_tracking(start_north, start_east, radius, v_circle)
    else:
        live_guidance.circular_tracking(start_north, start_east, radius, v_circle)


def run_live_square():
    if use_drl:
        live_drl.square_tracking(start_north, start_east, side_length, side_target_speed)
    else:
        live_guidance.square_tracking(start_north, start_east, side_length, side_target_speed)


def run_live_stationary():
    if use_drl:
        live_drl.stationary_tracking(forward_offset=10.0, starboard_offset=5.0)
    else:
        live_guidance.stationary_target_tracking(forward_offset=10.0, starboard_offset=5.0)


def exit_handler():
    global ugps_stop_event

    if ugps_stop_event is not None:
        ugps_stop_event.set()

    try:
        if use_drl:
            live_drl._save_log()
        else:
            live_guidance.save_log()
    except Exception:
        pass


def plot_simulation_results(simTime, simData, targetData, save_position_plot=False):
    plotVehicleStates(simTime, simData, 1)
    plotControls(simTime, simData, otter, 2)
    plotPosTar2(simTime, simData, 4, targetData, savePlot=save_position_plot)
    plotSurge(simTime, simData, 6)
    plotYaw(simTime, simData, 7)

    if not save_position_plot:
        plotSpeed(simTime, simData, 5)

    if animate_path:
        print("Checking data before animation...")
        print("simData size:", len(simData))
        print("targetData size:", len(targetData))

        plot3D(simData, numDataPoints, FPS, filename, 3)
        plot2D(simData, numDataPoints, FPS, "./2D_animation.gif", 6, targetData)

    plt.show()
    plt.close()


def main(main_option):
    global ugps_stop_event

    if main_option == 1:
        if ctrl_option == 3:
            print("DRL simulation is not implemented in this main script, see drl_control/Otter_dl, or otter_simulator_DRL for sim logic.")
            return

        if ctrl_option == 1:
            simTime, simData, targetData = simulator.simulate(
                N,
                sampleTime,
                otter,
                surge_PID,
                yaw_PID,
                trajectory_reference=True
            )
            plot_simulation_results(simTime, simData, targetData, save_position_plot=True)

        elif ctrl_option == 2:
            simTime, simData, targetData = simulator.simulate_NMPC(
                N=N,
                sampleTime=sampleTime,
                otter=otter,
                nmpc=nmpc,
                control_dt=control_dt
            )
            plot_simulation_results(simTime, simData, targetData)

        else:
            print("Invalid controller option.")

    elif main_option == 2:
        otter.sorted_values.setdefault("target_north_from_observer", start_north)
        otter.sorted_values.setdefault("target_east_from_observer", start_east)
        otter.sorted_values.setdefault("tau_X", 0.0)
        otter.sorted_values.setdefault("tau_N", 0.0)

        atexit.register(exit_handler)

        if live_source_option == 2:
            if use_drl:
                print("UGPS tracking is currently implemented through live_guidance, not LiveDRLController.")

            ugps_stop_event = threading.Event()

            ugps_thread = threading.Thread(
                target=live_guidance.ugps_reader,
                args=(ugps_stop_event, URL_GLOBAL, URL_ACOUSTIC, otter),
                daemon=True
            )

            tracking_thread = threading.Thread(
                target=live_guidance.ugps_target_tracking,
                args=(ugps_stop_event,),
                daemon=True
            )

            ugps_thread.start()

            if enable_live_plot:
                tracking_thread.start()
                print("Waiting for data")
                time.sleep(6)
                Live_plotter.live_plotter(otter)
            else:
                live_guidance.ugps_target_tracking(ugps_stop_event)

        elif live_source_option == 1:
            path_map = {
                1: run_live_straight,
                2: run_live_circular,
                3: run_live_square,
                4: run_live_stationary,
            }

            selected_function = path_map.get(path_option)

            if selected_function is None:
                print("Invalid path option.")
                return

            if enable_live_plot:
                tracking_thread = threading.Thread(target=selected_function, daemon=True)
                tracking_thread.start()

                print("Waiting for data")
                time.sleep(6)
                Live_plotter.live_plotter(otter)
            else:
                selected_function()

        else:
            print("Invalid live target source option.")

    else:
        print("Invalid main option.")


if __name__ == "__main__":
    main(main_option)