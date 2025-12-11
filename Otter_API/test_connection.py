import socket

ip = "192.168.53.2"
for port in [2009, 3200, 8080, 22, 32001]:
    try:
        s = socket.socket()
        s.settimeout(2)
        s.connect(("192.168.53.2", port))
        print(f"Success on port {port}")
    except Exception as e:
        print(f"Port {port} failed:", e)
    finally:
        s.close()
        