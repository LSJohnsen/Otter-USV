import requests
import threading
import time
#http://192.168.7.1/#/
BASE_URL = "http://192.168.2.94/"  
URL_GLOBAL   = BASE_URL + "/api/v1/position/global"
URL_ACOUSTIC = BASE_URL + "/api/v1/position/acoustic/filtered"

def ugps_get(url):
    try:
        r = requests.get(url, timeout=0.3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

print("Reading UGPS data...\n")

def ugps_reader(otter=None):
    print("Starting UGPS reader thread...\n")
    while True:
        g = ugps_get(URL_GLOBAL)
        a = ugps_get(URL_ACOUSTIC)

        if g and a:
            # Global position
            lat   = float(g.get("lat", 0.0))
            lon   = float(g.get("lon", 0.0))
            depth = -float(a.get("z", 0.0))  # z is negative down, make depth positive

            # Local coordinates (relative to antenna)
            x = float(a.get("x", 0.0))
            y = float(a.get("y", 0.0))
            z = float(a.get("z", 0.0))

            # store into otter.sorted_values 
            if otter is not None:
                otter.sorted_values["ugps_lat"]   = lat
                otter.sorted_values["ugps_lon"]   = lon
                otter.sorted_values["ugps_depth"] = depth
                otter.sorted_values["ugps_x"]     = x
                otter.sorted_values["ugps_y"]     = y
                otter.sorted_values["ugps_z"]     = z

            
            print(f"Global:  Lat:{lat:.6f}, Lon:{lon:.6f}, Depth:{depth:.2f} m")
            print(f"Local XYZ: X:{x:.2f} m, Y:{y:.2f} m, Z:{z:.2f} m")
            print("-" * 40)
        else:
            print("Waiting for valid UGPS data...")
            pass

        time.sleep(0.5)

if __name__ == "__main__":
    try:
        ugps_reader()
    except KeyboardInterrupt:
        print("Stopping.")