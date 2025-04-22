import Plot_control_forces as pcf
import Plot_yaw_velocity as pyv
import Plot_surge_velocity as psv
import Plot_thruster_speed as pts
import Plot_distance_to_target as pdt
import Plot_ship_target_path as pstp
import animate as animate
import os


#datelist = ["2024-04-23_15-30-521", "2024-04-23_12-55-233", "2024-04-23_12-50-003", "2024-04-23_12-44-083", "2024-04-23_12-37-482", "2024-04-23_12-35-582",
            #"2024-04-23_12-32-311", "2024-04-23_12-27-203", "2024-04-23_12-22-102", "2024-04-23_12-08-00"]

datelist = ["2024-04-23_12-44-083"]

#date = "2024-04-23_15-30-521"

def plotAll(date):
    if not os.path.exists(f"plots/{date}"):
        os.makedirs(f"plots/{date}")

    pcf.plot(date)

    pyv.plot(date)

    psv.plot(date)

    pts.plot(date)

    pdt.plot(date)

    pstp.plot(date)

#    animate.plot(date)


for date in datelist:
    plotAll(date)