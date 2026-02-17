
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

from logs.IO import log_to_csv


##########################################################################################################################################################
#                                                                      OPTIONS                                                                           #
#    For testing of the Otter USV. First options determine simulation parameters, secondary options either for UGPS or live tracking with sim target     #                                                                                                                                                                                                                 #
#          Ensure the correct IP and port are used for both UGPS, and Otter USV Radio/wifi depending method. Check in otter VCS                          #                                                                                                                                                        
##########################################################################################################################################################



N = 13333                                                                                               # Number of simulation samples
sampleTime = 0.02                                                                                       # Simulation time per sample. Usually at 0.02, other values could cause instabillity in the simulation
use_target_coordinates = False                                                                          # To use coordinates as a target or to use a linear path
use_moving_target = True                                                                                # To use moving target instead of target list (path following)
target_list = [[0, 10000]]                                                                              # List of targets to use if use_target_coordinates is set to True
end_when_last_target_reached = True                                                                     # Ends the simulation when the final target is reached
moving_target_start = [0, -10]                                                                          # Start point of the moving target if use_moving_target is set to True
moving_target_increase = [-0.5, 0.0]                                                                    # Movement of the moving target each second
target_radius = 0.1                                                                                     # Radius from center of target that counts as target reached, change this depending on the complete size of the run. Very low values causes instabillity
verbose = True                                                                                          # Enable verbose printing
log_simulation = True                                                                                   # Enable verbose for logging sim
store_force_file = False                                                                                # Store the simulated control forces in a .csv file
circular_target = True                                                                                  # Make the moving target a circle in the simulation
animate_path = False                                                                                    # This takes a lot of time! File stored as 2D_animation.gif

# NMPC
N_horizon   = 15                                                                                        # Prediction Horizon e.g. (less than 30 generally for fast updates)
control_dt  = 0.1                                                                                       # MPC update period (s)


# When connecting to live otter:
# Otter USV IPs (radio: 196.168.53.2 : 2009)
ip = "192.168.53.2"
#ip = "10.0.5.1"
port = 2009

# UGPS ip
BASE_URL = "http://192.168.2.94/"  
URL_GLOBAL   = BASE_URL + "/api/v1/position/global"                                                     
URL_ACOUSTIC = BASE_URL + "/api/v1/position/acoustic/filtered"

start_north = -20                                                                                       # Target north position from referance point
start_east = -20                                                                                        # Target east position from referance point
v_north = 0                                                                                             # Moving target speed north (m/s)
v_east = -1.5                                                                                           # Moving target speed east (m/s)
radius = 40                                                                                             # If tracking a circular motion
v_circle = 1.5                                                                                          # Angular velocity (m/s)
side_length = 50                                                                                        # Square tracking side length
side_target_speed = 1                                                                                   # Speed of square target
enable_live_plot = True                                                                                 # Enables live plotting
ugps_stop_event = None                                                                                  # For threading stop eventt


parameter_list = 3 # Tuning parameters, 1 for trial and error, 2 for pole placement wb = 0.5, and 3 for pole placement wb = 0.4


trial_and_error_parameters = {"surge_kp" : 12, "surge_ki" : 0.7, "surge_kd" : 0, "yaw_kp" : 37, "yaw_ki" : 4, "yaw_kd" : 8}
pp_05 = {"surge_kp" : 22.48, "surge_ki" : 3.92, "surge_kd" : 11.62, "yaw_kp" : 23.72, "yaw_ki" : 4.13, "yaw_kd" : 15.08}
pp_04 = {"surge_kp" : 14.39, "surge_ki" : 3.13, "surge_kd" : 0, "yaw_kp" : 15.21, "yaw_ki" : 0.7, "yaw_kd" : 1.86}
pp_04 = {"surge_kp" : 14.39, "surge_ki" : 25.13, "surge_kd" : 1, "yaw_kp" : 25.21, "yaw_ki" : 0.7, "yaw_kd" : 1.86} # third order trjacectory (>integral surge)
test_pdi = {"surge_kp" : 14.39, "surge_ki" : 3.13, "surge_kd" : 0, "yaw_kp" : 15.21, "yaw_ki" : 0.7, "yaw_kd" : 0}

if parameter_list == 1:
    pdi = trial_and_error_parameters
elif parameter_list == 2:
    pdi = pp_05
elif parameter_list == 3:
    pdi = pp_04
elif parameter_list == 4:
    pdi = test_pdi

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
                                            circular_target)                                            # Creates Simulator object




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

live_guidance = Live_guidance.live_guidance(ip, port, surge_PID, yaw_PID, target_radius, otter)                 # Live guidance object


#initialize nmpc - model_3dof.py transforms 6dof to 3dof by matrix reduction
otter_6dof_params = usv_params_6dof()
otter_3dof = Otter3DOF(otter_6dof_params)
nmpc = NMPCControl(
    f=otter_3dof,     
    N=N_horizon,
    sampleTime=control_dt,
)

print("Welcome to the Otter controller simulator and socket")                                                      
try:
    option = int(input("Enter option to simulate (1), or connect to the Otter USV (2): "))
except ValueError:
    print("You entered an invalid option, running simulation (1).")
    option = 1

try:
    ctrl_option = int(input("Choose control system to use: NMPC (1), PID (2): "))
except ValueError:
    print("Invalid control option, applying PID.")
    ctrl_option = 2

#apply NMPC if true
use_nmpc = (ctrl_option == 1)
live_guidance = Live_guidance.live_guidance(ip=ip, port=port, surge_PID=surge_PID, yaw_PID=yaw_PID, surge_setpoint=target_radius, otter=otter, 
    nmpc=nmpc,                 # NMPC object (can be ignored)
    use_nmpc=use_nmpc,         # True for NMPC, False for PID
    control_dt=control_dt
)


# Change to one indexed function?
def _target_tracking():
    live_guidance.target_tracking(start_north, start_east, v_north, v_east)

def _circular_tracking():
    live_guidance.circular_tracking(start_north, start_east, radius, v_circle)

def _square_tracking():
    live_guidance.square_tracking(start_north, start_east, side_length, side_target_speed)

def exit_handler():
    if ugps_stop_event is not None:
        ugps_stop_event.set()
    live_guidance.save_log()



# Main:

def main(option):
    #sim
    if option == 1:
        
        if ctrl_option == 2:
            [simTime, simData, targetData] = simulator.simulate(N, 
                                                                sampleTime, 
                                                                otter, 
                                                                surge_PID, 
                                                                yaw_PID,
                                                                trajectory_reference=True)   # This runs the whole simulation
            log_to_csv(simTime, simData, targetData, filename="sim_log_PID.csv", verbose=log_simulation)

            plotVehicleStates(simTime, simData, 1)                                                          #
            plotControls(simTime, simData, otter, 2)                                                        #
                                                                                                            #
            plotPosTar2(simTime, simData, 4, targetData, savePlot=True)                                                     # Plotting
            plotSpeed(simTime, simData, 5) 
            plotSurge(simTime, simData, 6)
            plotYaw(simTime, simData, 7)                                                                  #
            if animate_path:
                print("Checking data before animation...")
                print("simData size:", len(simData))
                print("targetData size:", len(targetData))

                plot3D(simData, numDataPoints, FPS, filename, 3)
                plot2D(simData, numDataPoints, FPS, "./2D_animation.gif", 6, targetData)
            # Saves a GIF for 3d animation in the same folder as main

            plt.show()
            plt.close()


        elif ctrl_option == 1:
            [simTime, simData, targetData] = simulator.simulate_NMPC(
                                                        N=N,
                                                        sampleTime=sampleTime,
                                                        otter=otter,
                                                        nmpc=nmpc,
                                                        control_dt=control_dt)
            log_to_csv(simTime, simData, targetData, filename="sim_log_nmpc.csv", verbose=log_simulation)

            plotVehicleStates(simTime, simData, 1)                                                          #
            plotControls(simTime, simData, otter, 2)                                                        #
                                                                                                            #
            plotPosTar2(simTime, simData, 4, targetData)                                                    # Plotting
            #plotSpeed(simTime, simData, 5)
            plotSurge(simTime, simData, 6)
            plotYaw(simTime, simData, 7)                                                                 #
            if animate_path:
                print("Checking data before animation...")
                print("simData size:", len(simData))
                print("targetData size:", len(targetData))

                plot3D(simData, numDataPoints, FPS, filename, 3)
                plot2D(simData, numDataPoints, FPS, "./2D_animation.gif", 6, targetData)
            # Saves a GIF for 3d animation in the same folder as main

            plt.show()
            plt.close()
    #live
    elif option == 2:

        _target_thread = threading.Thread(target=_target_tracking, args=())
        _target_thread.daemon = True
        _circle_thread = threading.Thread(target=_circular_tracking, args=())
        _circle_thread.daemon = True
        _square_thread = threading.Thread(target=_square_tracking, args=())
        _square_thread.daemon = True
               
        # Some initial values to avoid nmpc crash
        otter.sorted_values.setdefault("target_north_from_observer", start_north)                               
        otter.sorted_values.setdefault("target_east_from_observer",  start_east)
        otter.sorted_values.setdefault("tau_X", 0.0)
        otter.sorted_values.setdefault("tau_N", 0.0)               

        option = float(input("Enter 1 for target tracking with moving target, 2 for circular motion, 3 for square tracking or 4 for UGPS tracking: "))

        if option == 1:
            if enable_live_plot:    
                _target_thread.start()
                print(" Waiting for data")
                time.sleep(6)
                p1 = Live_plotter.live_plotter(otter)
                atexit.register(exit_handler)
            else:
                live_guidance.target_tracking(start_north, start_east, v_north, v_east)
                atexit.register(exit_handler)

        elif option == 2:
            if enable_live_plot:
                _circle_thread.start()
                print(" Waiting for data")
                time.sleep(6)
                p1 = Live_plotter.live_plotter(otter)
                atexit.register(exit_handler)
            else:
                live_guidance.circular_tracking(start_north, start_east, radius, v_circle)
                atexit.register(exit_handler)

        elif option == 3:
            if enable_live_plot:
                _square_thread.start()
                print(" Waiting for data")
                time.sleep(6)
                p1 = Live_plotter.live_plotter(otter)
                atexit.register(exit_handler)
            else:
                live_guidance.square_tracking(start_north, start_east, radius, v_circle)
                atexit.register(exit_handler)

        elif option == 4:

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
            atexit.register(exit_handler)

            if enable_live_plot:
                tracking_thread.start()
                print("Waiting for data")
                time.sleep(6)
                p1 = Live_plotter.live_plotter(otter)
                
            else:
                live_guidance.ugps_target_tracking(ugps_stop_event)



if __name__ == "__main__":
    main(option)
