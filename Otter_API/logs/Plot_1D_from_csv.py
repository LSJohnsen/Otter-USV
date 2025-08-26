import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy.interpolate import interp1d

# Ensure the 'plots' directory exists
os.makedirs('plots', exist_ok=True)

matplotlib.use('Agg')

date = "2024-04-23_15-30-521"
y_label = "Surge velocity (m/s)"
filename = "Surge velocity"
dpi = 600

# Load the data
file_path = f'{date}.csv'
data = pd.read_csv(file_path, sep=';')

data_corrected = data.iloc[3:]

data_corrected['Timestamp'] = pd.to_datetime(data_corrected.iloc[:, 0], format='%Y-%m-%d_%H:%M:%S:%f', errors='coerce')

data_corrected = data_corrected.dropna(subset=['Timestamp'])

# Ensure numeric values
data_corrected['north_from_observer'] = pd.to_numeric(data_corrected['north_from_observer'], errors='coerce')
data_corrected['current_orientation_3'] = pd.to_numeric(data_corrected['current_orientation_3'], errors='coerce')

# Calculate the change in "north_from_observer"
data_corrected['delta_north'] = data_corrected['north_from_observer'].diff()

# Calculate the north speed (change in north position over time interval)
data_corrected['north_speed'] = data_corrected['delta_north'] / 0.1  # time interval is 0.1s

# Convert current_orientation_3 to radians for the cosine function
data_corrected['heading_radians'] = np.radians(data_corrected['current_orientation_3'])

# Calculate the surge speed
data_corrected['surge_speed'] = data_corrected['north_speed'] * np.cos(data_corrected['heading_radians'])

# Replace NaN values resulting from the diff() operation
data_corrected['surge_speed'].fillna(0, inplace=True)

# Interpolate surge_speed to smooth the data using scipy's interp1d
valid_indices = data_corrected['surge_speed'] != 0
x_valid = data_corrected['Timestamp'][valid_indices].apply(lambda x: x.timestamp()).values
y_valid = data_corrected['surge_speed'][valid_indices].values

# Create interpolation function
interpolation_function = interp1d(x_valid, y_valid, kind='linear', fill_value="extrapolate")

# Apply interpolation to all timestamps
x_all = data_corrected['Timestamp'].apply(lambda x: x.timestamp()).values
data_corrected['surge_speed_interpolated'] = interpolation_function(x_all)

# Variable for plotting every nth data point
n = 5
data_thinned = data_corrected.iloc[::n, :]

initial_time = data_thinned['Timestamp'].iloc[0]
data_thinned['Seconds'] = (data_thinned['Timestamp'] - initial_time).dt.total_seconds()

plt.figure(figsize=(12, 6))

plt.plot(data_thinned['Seconds'], data_thinned['surge_speed_interpolated'], label='Surge Velocity', linestyle='-', linewidth=1.5)

plt.title(f'{filename} over time')
plt.xlabel('Time (seconds)')
plt.ylabel(y_label)
plt.legend()

plt.ylim(data_corrected['surge_speed_interpolated'].min(), data_corrected['surge_speed_interpolated'].max())

plt.grid(True)
plt.tight_layout()
plt.savefig(f"plots/{date}/{filename}_{date}_1x.png", dpi=dpi)
plt.show()
