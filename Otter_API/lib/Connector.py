import time
import socket
import select
import numpy as np
from numpy import pi
import pymap3d as pm
from copy import copy


#
#   This file handles all the connections and messages to and from the Otter, aswell as the Otters current values such as speed and position.
#



# Calculates the checksum for the Otter. ---CHECKSUM IS NMEA STANDARD---
def checksum(message):
    checksum = 0
    for character in message:
        checksum ^= ord(character)
    checksum = hex(checksum)
    checksum = checksum[2:]
    if len(checksum) == 1:
        checksum = "0" + checksum
    return checksum

# Finds the difference between two angles
def smallest_signed_angle_between(x, y):
        a = (x - y) % (2 * pi)
        b = (y - x) % (2 * pi)
        return -a if a < b else b


# Main class for connecting to the Otter.
class otter_connector():
    def __init__(self):

        # This enables the printing of messages. Used for debugging. Slows down the software a bit.
        self.verbose = True



        # Keeping track of the connection status
        self.connection_status = False

        # Stores the last message received from the Otter
        self.last_message_received = ""

        # VARIABLES
        self.current_position = [0.0, 0.0, 0.0]
        self.previous_position = [0.0, 0.0, 0.0]
        self.last_speed_update = time.time()
        self.current_course_over_ground = 0
        self.current_speed = 0
        self.current_fuel_capacity = 0
        self.current_orientation = [0, 0, 0] # roll, pitch, yaw
        self.current_rotational_velocities = [0, 0, 0] # roll, pitch, yaw


    # Establishes a socket communication to the Otter with ip and port. Default ip and port is for Wifi connection. Pass other parameters to alter these.
    def establish_connection(self, ip, port):

        try:
            if self.verbose:
                print(f"Connecting with ip {ip} and port {port}")
            # Creates a socket object.
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((ip, port))
            # camera = cv2.VideoCapture("rtsp://admin:pwd4hik!@10.0.5.1/Streaming/Channels/101")
            # if not camera.isOpened():
            # print("Error connecting to Otter camera")
            # return False
            self.connection_status = True
            print("connected")
            return True
        except:
            print("Could not connect to Otter")
            return False


    # Sends a message through the socket connection to the Otter. Calculates checksum and adds \r\n to the message.     ---CHECKSUM IS NMEA STANDARD---
    def send_message(self, message, checksum_needed):
        try:
            if checksum_needed:
                if self.verbose:
                    print("Adding checksum")
                message += "*"
                message += checksum(message[1:-1])
            message += "\r\n"
            self.sock.sendall(message.encode())
            if self.verbose:
                print("Sending message:", message)
                print("Message sendt OK")
            return True
        except:
            print("Couldn't send message to Otter")
            return False

    # Closes the socket connection.
    def close_connection(self):
        try:
            self.sock.close()
            self.connection_status = False
            return True
        except:
            print("Error when disconnecting from Otter")
            return False

    # Checks if a socket connection is established and returns a boolean.
    def check_connection(self):
        if self.verbose:
            print(f"Connection status is {self.connection_status}")
        return self.connection_status


    # Reads a message from the Otter and returns it and stores it in "last_message_recieved".
    def read_message(self, timeout = 10):
        if self.verbose:
            print("Listening to message from Otter")
        self.sock.setblocking(0)
        ready = select.select([self.sock], [], [], timeout)
        if ready[0]:
            try:
                received_message = self.sock.recv(1024).decode()
                self.last_message_received = received_message
                return received_message
            except:
                if self.verbose:
                    print("Error in recieving message. This is going to fast")

        else:
            return None


    # Updates all the values for the Otter with the messages that are sendt from the Otter. This needs to be called every time the values should be updated. Timeout for the read message is by default 10, but can be changed as an argument.
    def update_values(self, timeout = 10):
        msg = self.read_message(timeout)
        if msg is None:
            if self.verbose:
                print("No message received from Otter")
                print("Check communication")
            return
        list = msg.split()

        # We skip the last one because it is usually incomplete
        list = list[:-1]

        # Get the newest messages
        gps_message = ""
        imu_message = ""
        mod_message = ""
        for message in list:
            if message[:8] == "$PMARGPS":
                gps_message = message
            elif message[:8] == "$PMARIMU":
                imu_message = message
            elif message[:8] == "$PMARMOD":
                mod_message = message
            elif message[:8] == "$PMARERR":
                error_message = message
                print(error_message)

        # Check for checksum error
        if checksum(gps_message[1:-3]) != gps_message[-2:].lower():
            print("Checksum error in $PMARGPS message")
            return

        gps_message = gps_message.split("*")[0] # Removing checksum
        gps_message = gps_message.split(",")

        # Update position
        lat_msg = gps_message[2]
        lon_msg = gps_message[4]
        lat_deg = lat_msg[:2]
        lon_deg = lon_msg[:3]
        lat_min = lat_msg[2:]
        lon_min = lon_msg[3:]

        try:
            lat = float(lat_deg) + ( (float(lat_min)/100) / 0.6)
            lon = float(lon_deg) + ( (float(lon_min)/100) / 0.6)
        except:
            if self.verbose:
                print("Could not get GPS coordintes. Check GPS coverage!")
            lat = 0
            lon = 0

        if gps_message[3] == "S":
            lat *= -1
        if gps_message[5] == "W":
            lon *= -1

        # Creates current position. Height is set as 0.0 as this is not implemented yet.
        self.current_position = [lat, lon, 0.0]


        # Update speed
        n, e, d = pm.geodetic2ned(self.current_position[0], self.current_position[1], 0, \
        self.previous_position[0], self.previous_position[1], 0)
        s = np.hypot(n, e)
        dt = time.time() - self.last_speed_update 
        if dt <= 0:
            return
        v = s / (time.time() - self.last_speed_update)
        #self.current_speed = float(gps_message[7])
        self.current_speed = v
        self.last_speed_update = time.time()
        self.previous_position = copy(self.current_position)


        # Update course over ground
        if gps_message[8] != "":
            self.current_course_over_ground = float(gps_message[8])
        else:
            print("Unable to read course over ground from Otter")

        # Check for checksum error
        if checksum(imu_message[1:-3]) != imu_message[-2:].lower():
            print("Checksum error in $PMARIMU message")
            return

        # Update orientation
        imu_message = imu_message.split("*")[0] # Removing checksum
        imu_message = imu_message.split(",")

        if imu_message[1] != "":
            self.current_orientation[0] = float(imu_message[1])
        if imu_message[2] != "":
            self.current_orientation[1] = float(imu_message[2])
        if imu_message[3] != "":
            self.current_orientation[2] = float(imu_message[3])
            self.yaw = smallest_signed_angle_between(0, np.deg2rad(-self.current_orientation[2]))

        # Update rotational velocities
        if imu_message[4] != "":
            self.current_rotational_velocities[0] = float(imu_message[4])
        if imu_message[5] != "":
            self.current_rotational_velocities[1] = float(imu_message[5])
        if imu_message[6] != "":
            self.current_rotational_velocities[2] = float(imu_message[6])

        # Check for checksum error
        if checksum(mod_message[1:-3]) != mod_message[-2:].lower():
            print("Checksum error in $PMARMOD message")
            return

        # Update fuel capacity
        mod_message = mod_message.split("*")[0] # Removing checksum
        mod_message = mod_message.split(",")

        self.current_fuel_capacity = mod_message[2]


        return

