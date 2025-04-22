import Otter_api
import pandas as pd
import time



lines_per_second_stored = 50
lines_per_second_output = 10

ip = "192.168.53.2"
port = 2009



lines_to_average = lines_per_second_stored // lines_per_second_output
groups = lines_per_second_stored // lines_to_average

df = pd.read_csv("force_array.csv", delimiter=";")

average_list = []

for i in range(groups):
    start_idx = i * lines_to_average
    end_idx = start_idx + lines_to_average

    for second_start in range(0, len(df), lines_per_second_stored):
        avg = df.iloc[second_start + start_idx: second_start + end_idx].mean()
        average_list.append(avg)

average_df = pd.DataFrame(average_list)

otter = Otter_api.otter()
otter.establish_connection(ip, port)


cycletime = 1/lines_per_second_output
counter = 0

otter.controller_inputs_torque(20, 0)
time.sleep(1)
otter.controller_inputs_torque(20, 0)
time.sleep(1)
otter.controller_inputs_torque(20, 0)
time.sleep(1)
otter.controller_inputs_torque(20, 0)
time.sleep(1)


while True:
    start_time = time.time()


    asd = average_df.iloc[counter]
    otter.controller_inputs_torque(asd["tau_X"], asd["tau_N"])



    counter = counter + 1
    elapsed_time = time.time() - start_time
    sleep_time = max(0, cycletime - elapsed_time)
    time.sleep(sleep_time)






