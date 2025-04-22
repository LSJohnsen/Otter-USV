import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

date = "2024-04-23_15-30-521"
data = pd.read_csv(f'{date}.csv', sep=';')
dpi = 600


data_corrected = data.iloc[3:]


data_corrected['north_from_observer'] = pd.to_numeric(data_corrected['north_from_observer'], errors='coerce')
data_corrected['east_from_observer'] = pd.to_numeric(data_corrected['east_from_observer'], errors='coerce')
data_corrected['target_north_from_observer'] = pd.to_numeric(data_corrected['target_north_from_observer'], errors='coerce')
data_corrected['target_east_from_observer'] = pd.to_numeric(data_corrected['target_east_from_observer'], errors='coerce')


plt.figure(figsize=(10, 6))


plt.plot(data_corrected['east_from_observer'], data_corrected['north_from_observer'], label='Ship Path', linestyle='-', linewidth=1.5)

plt.plot(data_corrected['target_east_from_observer'], data_corrected['target_north_from_observer'], label='Target Path', linestyle='-', linewidth=1.5)

plt.title('Ship and Target Paths')
plt.xlabel('East from Observer (m)')
plt.ylabel('North from Observer (m)')
plt.legend()
plt.grid(True)
plt.savefig(f"plots/{date}/Paths_{date}_2x.png", dpi=dpi)
plt.show()
