import socket

ips = ["192.168.53.2", "10.0.5.1"]
ports = [2009, 3200, 8080, 22, 32001]

for ip in ips:
    print(f"\nTesting IP: {ip}")

    for port in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)

        try:
            s.connect((ip, port))
            print(f"Success: {ip}:{port}")

        except Exception as e:
            print(f"Failed:  {ip}:{port} -> {e}")

        finally:
            s.close()