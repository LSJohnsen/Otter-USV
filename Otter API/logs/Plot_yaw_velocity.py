import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def plot(date):


    y = "current_rotational_velocities_3"
    y_label = "Yaw velocity (deg/s)"
    filename = "Yaw velocity"
    dpi = 600
    fontsize = 14

    data = pd.read_csv(f"{date}.csv", sep=';')



    data_corrected = data.iloc[3:]


    data_corrected['Timestamp'] = pd.to_datetime(data_corrected.iloc[:, 0], format='%Y-%m-%d_%H:%M:%S:%f', errors='coerce')


    data_corrected = data_corrected.dropna(subset=['Timestamp'])


    data_corrected[y] = pd.to_numeric(data_corrected[y], errors='coerce')

    # variable for plotting every nth data point
    n = 5
    data_thinned = data_corrected.iloc[::n, :]


    initial_time = data_thinned['Timestamp'].iloc[0]
    data_thinned['Seconds'] = (data_thinned['Timestamp'] - initial_time).dt.total_seconds()


    plt.figure(figsize=(12, 6))


    plt.plot(data_thinned['Seconds'], data_thinned[y], label='Distance to Target', linestyle='-', linewidth=1.5)

    plt.title(f'{filename} over time', fontsize=fontsize)
    plt.xlabel('Time (seconds)', fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.legend()


    plt.ylim(data_corrected[y].min(), data_corrected[y].max())

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{date}/{filename}_{date}.eps", dpi=dpi)
    plt.show()


