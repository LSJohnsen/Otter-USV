import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def plot(date):
    y_label = "Surge velocity (m/s)"
    filename = "Surge velocity"
    dpi = 600
    fontsize = 14
    max_value = 3
    n = 7  # Number of points to average for downsampling

    file_path = f'{date}.csv'
    data = pd.read_csv(file_path, sep=';')

    # Skipping first three rows and keeping a copy of the original data
    data_corrected = data.iloc[3:].copy()

    # Convert Timestamp to datetime
    data_corrected['Timestamp'] = pd.to_datetime(data_corrected.iloc[:, 0], format='%Y-%m-%d_%H:%M:%S:%f',
                                                 errors='coerce')

    # Drop rows where Timestamp couldn't be parsed
    data_corrected = data_corrected.dropna(subset=['Timestamp'])

    # Ensure numeric values
    data_corrected['north_from_observer'] = pd.to_numeric(data_corrected['north_from_observer'], errors='coerce')
    data_corrected['east_from_observer'] = pd.to_numeric(data_corrected['east_from_observer'], errors='coerce')
    data_corrected['current_orientation_3'] = pd.to_numeric(data_corrected['current_orientation_3'], errors='coerce')

    # Downsample the data
    def downsample(data, n):
        return data.groupby(data.index // n).mean()

    numeric_cols = data_corrected.select_dtypes(include=[np.number]).columns
    non_numeric_cols = data_corrected.select_dtypes(exclude=[np.number]).columns

    numeric_downsampled = downsample(data_corrected[numeric_cols], n)
    non_numeric_downsampled = data_corrected[non_numeric_cols].groupby(data_corrected.index // n).first()

    data_downsampled = pd.concat([numeric_downsampled, non_numeric_downsampled], axis=1).sort_index()

    # Calculate the changes and speeds
    data_downsampled['delta_north'] = data_downsampled['north_from_observer'].diff()
    data_downsampled['delta_east'] = data_downsampled['east_from_observer'].diff()
    data_downsampled['delta_time'] = data_downsampled['Timestamp'].diff().dt.total_seconds()

    data_downsampled['north_speed'] = data_downsampled['delta_north'] / data_downsampled['delta_time']
    data_downsampled['east_speed'] = data_downsampled['delta_east'] / data_downsampled['delta_time']

    # Function to handle zero-value runs
    def handle_zero_runs(series):
        zero_run_start = None
        for i in range(len(series)):
            if series[i] == 0:
                if zero_run_start is None:
                    zero_run_start = i
            else:
                if zero_run_start is not None:
                    zero_run_end = i
                    num_zeros = zero_run_end - zero_run_start
                    prev_value = series[zero_run_start - 1] if zero_run_start > 0 else series[zero_run_end]
                    next_value = series[zero_run_end]
                    average_value = (prev_value + next_value) / 2
                    series[zero_run_start:zero_run_end] = average_value
                    zero_run_start = None
        return series

    data_downsampled['north_speed'] = handle_zero_runs(data_downsampled['north_speed'].values)
    data_downsampled['east_speed'] = handle_zero_runs(data_downsampled['east_speed'].values)

    # Calculate surge speed
    data_downsampled['surge_speed'] = np.sqrt(
        data_downsampled['north_speed'] ** 2 + data_downsampled['east_speed'] ** 2)

    # Interpolate and smooth surge speed
    valid_indices = data_downsampled['surge_speed'] != 0
    x_valid = data_downsampled['Timestamp'][valid_indices].apply(lambda x: x.timestamp()).values
    y_valid = data_downsampled['surge_speed'][valid_indices].values

    if len(x_valid) < 2 or len(y_valid) < 2:
        print("Not enough valid data points for interpolation")
        return

    interpolation_function = interp1d(x_valid, y_valid, kind='linear', fill_value="extrapolate")
    x_all = data_downsampled['Timestamp'].apply(lambda x: x.timestamp()).values
    data_downsampled['surge_speed_interpolated'] = interpolation_function(x_all)

    # Smooth exceeding values
    def smooth_exceeding_values(series, max_value):
        for i in range(len(series)):
            if series[i] > max_value:
                start_idx = max(i - 2, 0)
                end_idx = min(i + 3, len(series))
                surrounding_values = np.concatenate((series[start_idx:i], series[i + 1:end_idx]))
                series[i] = np.mean(surrounding_values)
        return series

    data_downsampled['surge_speed_interpolated'] = smooth_exceeding_values(
        data_downsampled['surge_speed_interpolated'].values, max_value)

    # Apply the rolling average
    window_size = 10  # Adjust the window size as needed
    data_downsampled['surge_speed_smoothed'] = data_downsampled['surge_speed_interpolated'].rolling(window=window_size,
                                                                                                    center=True).mean()

    # Fill NaN values in the rolling average
    data_downsampled['surge_speed_smoothed'].bfill(inplace=True)
    data_downsampled['surge_speed_smoothed'].ffill(inplace=True)

    # Add the initial zero point after averaging
    initial_zero_point_after_avg = pd.DataFrame({
        'Seconds': [0],
        'surge_speed_smoothed': [0]
    })

    initial_time = data_downsampled['Timestamp'].iloc[0]
    data_downsampled['Seconds'] = (data_downsampled['Timestamp'] - initial_time).dt.total_seconds()

    #data_downsampled = pd.concat([initial_zero_point_after_avg, data_downsampled]).sort_values(
        #by='Seconds').reset_index(drop=True)

    # Debugging info
    print(data_downsampled[['Seconds', 'surge_speed_smoothed']].head(10))
    print(data_downsampled[['Seconds', 'surge_speed_smoothed']].tail(10))

    plt.figure(figsize=(12, 6))
    plt.plot(data_downsampled['Seconds'], data_downsampled['surge_speed_smoothed'], label='Surge Velocity',
             linestyle='-', linewidth=1.5)
    plt.title(f'{filename} over time', fontsize=fontsize)
    plt.xlabel('Time (seconds)', fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.legend()

    plt.ylim(0, max(data_downsampled['surge_speed_smoothed'].max(), 3))

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{date}/{filename}_{date}.eps", dpi=dpi)
    plt.show()


#plot("2024-04-23_12-50-003")
