import time
import socket
import select
import numpy as np
from numpy import pi
import pymap3d as pm
from copy import copy


#
#   This file handles all the connections and messages to and from the Otter,
#   as well as the Otter's current values such as speed and position.
#


# Calculates the checksum for the Otter. ---CHECKSUM IS NMEA STANDARD---
def checksum(message):
    checksum_value = 0

    for character in message:
        checksum_value ^= ord(character)

    checksum_hex = hex(checksum_value)[2:]

    if len(checksum_hex) == 1:
        checksum_hex = "0" + checksum_hex

    return checksum_hex.lower()


# Finds the difference between two angles
def smallest_signed_angle_between(x, y):
    a = (x - y) % (2 * pi)
    b = (y - x) % (2 * pi)
    return -a if a < b else b


# Main class for connecting to the Otter.
class otter_connector:
    def __init__(self):

        # This enables the printing of messages. Used for debugging.
        # It slows down the software a bit.
        self.verbose = True

        # Keeping track of the connection status
        self.connection_status = False

        # Stores the last message received from the Otter
        self.last_message_received = ""

        # Socket placeholder
        self.sock = None

        # Variables
        self.current_position = [0.0, 0.0, 0.0]
        self.previous_position = [0.0, 0.0, 0.0]
        self.last_speed_update = time.time()

        self.current_course_over_ground = 0.0
        self.current_speed = 0.0
        self.current_fuel_capacity = 0.0

        # roll, pitch, yaw
        self.current_orientation = [0.0, 0.0, 0.0]

        # roll rate, pitch rate, yaw rate
        self.current_rotational_velocities = [0.0, 0.0, 0.0]

        # Optional yaw field used elsewhere
        self.yaw = 0.0

    # Establishes a socket communication to the Otter.
    def establish_connection(self, ip, port):
        try:
            if self.verbose:
                print(f"Connecting with ip {ip} and port {port}")

            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((ip, port))

            self.connection_status = True
            print("connected")
            return True

        except Exception as e:
            print(f"Could not connect to Otter: {e}")
            self.connection_status = False
            return False

    # Sends a message through the socket connection to the Otter.
    # Calculates checksum and adds \r\n to the message.
    def send_message(self, message, checksum_needed):
        try:
            if self.sock is None:
                print("Couldn't send message to Otter: socket is not initialized")
                return False

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

        except Exception as e:
            print(f"Couldn't send message to Otter: {e}")
            return False

    # Closes the socket connection.
    def close_connection(self):
        try:
            if self.sock is not None:
                self.sock.close()

            self.connection_status = False
            return True

        except Exception as e:
            print(f"Error when disconnecting from Otter: {e}")
            return False

    # Checks if a socket connection is established and returns a boolean.
    def check_connection(self):
        if self.verbose:
            print(f"Connection status is {self.connection_status}")

        return self.connection_status

    # Reads a message from the Otter and returns it.
    def read_message(self, timeout=10):
        if self.verbose:
            print("Listening to message from Otter")

        if self.sock is None:
            if self.verbose:
                print("No socket initialized")
            return None

        try:
            self.sock.setblocking(0)
            ready = select.select([self.sock], [], [], timeout)

            if ready[0]:
                received_message = self.sock.recv(4096).decode(errors="ignore")
                self.last_message_received = received_message
                return received_message

            return None

        except Exception as e:
            if self.verbose:
                print(f"Error in receiving message: {e}")
            return None

    def _valid_nmea_checksum(self, raw_message):
        """
        Checks NMEA-style checksum.

        Expects something like:
            $PMARIMU,...*hh
        """

        if raw_message is None:
            return False

        if raw_message == "":
            return False

        if "*" not in raw_message:
            return False

        if len(raw_message) < 4:
            return False

        try:
            expected = raw_message[-2:].lower()
            calculated = checksum(raw_message[1:-3])
            return calculated == expected

        except Exception:
            return False

    def _parse_gps(self, gps_message):
        """
        Parses and updates GPS values.

        If GPS is bad, returns False and keeps previous GPS values.
        """

        if gps_message == "":
            if self.verbose:
                print("No $PMARGPS message found - keeping previous GPS values")
            return False

        if not self._valid_nmea_checksum(gps_message):
            print("Checksum error in $PMARGPS message - keeping previous GPS values")
            return False

        try:
            gps_fields = gps_message.split("*")[0]
            gps_fields = gps_fields.split(",")

            if len(gps_fields) < 9:
                print("Invalid $PMARGPS message length - keeping previous GPS values")
                return False

            lat_msg = gps_fields[2]
            lon_msg = gps_fields[4]

            if lat_msg == "" or lon_msg == "":
                if self.verbose:
                    print("Empty GPS coordinates - keeping previous GPS values")
                return False

            lat_deg = lat_msg[:2]
            lon_deg = lon_msg[:3]
            lat_min = lat_msg[2:]
            lon_min = lon_msg[3:]

            lat = float(lat_deg) + ((float(lat_min) / 100.0) / 0.6)
            lon = float(lon_deg) + ((float(lon_min) / 100.0) / 0.6)

            if gps_fields[3] == "S":
                lat *= -1.0

            if gps_fields[5] == "W":
                lon *= -1.0

            # Update position
            self.current_position = [lat, lon, 0.0]

            # Update speed from position difference
            n, e, _ = pm.geodetic2ned(
                self.current_position[0],
                self.current_position[1],
                0.0,
                self.previous_position[0],
                self.previous_position[1],
                0.0
            )

            distance = np.hypot(n, e)
            now = time.time()
            dt = now - self.last_speed_update

            if dt > 0:
                self.current_speed = distance / dt
                self.last_speed_update = now
                self.previous_position = copy(self.current_position)

            # Update course over ground
            if gps_fields[8] != "":
                self.current_course_over_ground = float(gps_fields[8])
            else:
                if self.verbose:
                    print("Unable to read course over ground from Otter")

            return True

        except Exception as e:
            print(f"Error parsing $PMARGPS message: {e}")
            return False

    def _parse_imu(self, imu_message):
        """
        Parses and updates IMU values.

        If IMU checksum is bad, keeps previous IMU values.
        """

        if imu_message == "":
            if self.verbose:
                print("No $PMARIMU message found - keeping previous IMU values")
            return False

        if not self._valid_nmea_checksum(imu_message):
            print("Checksum error in $PMARIMU message - keeping previous IMU values")
            return False

        try:
            imu_fields = imu_message.split("*")[0]
            imu_fields = imu_fields.split(",")

            # Update orientation
            if len(imu_fields) > 1 and imu_fields[1] != "":
                self.current_orientation[0] = float(imu_fields[1])

            if len(imu_fields) > 2 and imu_fields[2] != "":
                self.current_orientation[1] = float(imu_fields[2])

            if len(imu_fields) > 3 and imu_fields[3] != "":
                self.current_orientation[2] = float(imu_fields[3])

                # Keep your original yaw convention here
                self.yaw = smallest_signed_angle_between(
                    0,
                    np.deg2rad(-self.current_orientation[2])
                )

            # Update rotational velocities
            if len(imu_fields) > 4 and imu_fields[4] != "":
                self.current_rotational_velocities[0] = float(imu_fields[4])

            if len(imu_fields) > 5 and imu_fields[5] != "":
                self.current_rotational_velocities[1] = float(imu_fields[5])

            if len(imu_fields) > 6 and imu_fields[6] != "":
                self.current_rotational_velocities[2] = float(imu_fields[6])

            return True

        except Exception as e:
            print(f"Error parsing $PMARIMU message: {e}")
            return False

    def _parse_mod(self, mod_message):
        """
        Parses and updates MOD/fuel values.

        If MOD is bad, keeps previous MOD values.
        This message should not be allowed to abort the control-state update.
        """

        if mod_message == "":
            if self.verbose:
                print("No $PMARMOD message found - keeping previous MOD values")
            return False

        if not self._valid_nmea_checksum(mod_message):
            print("Checksum error in $PMARMOD message - keeping previous MOD values")
            return False

        try:
            mod_fields = mod_message.split("*")[0]
            mod_fields = mod_fields.split(",")

            if len(mod_fields) > 2:
                self.current_fuel_capacity = mod_fields[2]

            return True

        except Exception as e:
            print(f"Error parsing $PMARMOD message: {e}")
            return False

    # Updates all the values for the Otter with the messages that are sent from the Otter.
    def update_values(self, timeout=10):
        msg = self.read_message(timeout)

        if msg is None:
            if self.verbose:
                print("No message received from Otter")
                print("Check communication")
            return

        messages = msg.split()

        # We skip the last one because it is usually incomplete
        if len(messages) > 0:
            messages = messages[:-1]

        # Get the newest messages
        gps_message = ""
        imu_message = ""
        mod_message = ""

        for message in messages:
            if message.startswith("$PMARGPS"):
                gps_message = message

            elif message.startswith("$PMARIMU"):
                imu_message = message

            elif message.startswith("$PMARMOD"):
                mod_message = message

            elif message.startswith("$PMARERR"):
                print(message)

        # Parse available messages.
        # GPS failure means position stays previous.
        # IMU failure means orientation stays previous.
        # MOD failure means fuel/status stays previous.
        self._parse_gps(gps_message)
        self._parse_imu(imu_message)
        self._parse_mod(mod_message)

        return