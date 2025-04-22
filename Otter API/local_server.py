import socket
import time


#
# This is just a test socket server for testing the socket connections to the otter. Use ip "localhost" if running this on a local machine.
#

status = False

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 2009)
print ("[+] Server IP {} | Port {}".format(server_address[0],server_address[1]))





while True:
    if not status:
        sock.bind(server_address)
        sock.listen(1)
        status = 1

    print("[+] Waiting for a client connection")
    connection, client_address = sock.accept()

    try:
        print("[+] Connection from", client_address)

        while True:
            try:
                data = connection.recv(1024)
                print("Received", data)
                if data:
                    print("Sending data back to client")
                    time.sleep(0.5)
                    connection.sendall(data)
                else:
                    print("No more data from", client_address)
                    break
            except:
                break
    finally:
        connection.close()
