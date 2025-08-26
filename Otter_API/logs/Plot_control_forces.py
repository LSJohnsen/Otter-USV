import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def plot(date):

    y1 = "tau_X"
    y2 = "tau_N"
    title = "Control force"
    y_label = "N"
    dpi = 600
    fontsize = 14

    data = pd.read_csv(f"{date}.csv", sep=';')


    data_corrected = data.iloc[3:]


    data_corrected['Timestamp'] = pd.to_datetime(data_corrected.iloc[:, 0], format='%Y-%m-%d_%H:%M:%S:%f', errors='coerce')


    data_corrected = data_corrected.dropna(subset=['Timestamp'])


    data_corrected[y1] = pd.to_numeric(data_corrected[y1], errors='coerce')
    data_corrected[y2] = pd.to_numeric(data_corrected[y2], errors='coerce')

    # Every nth data
    n = 5
    data_thinned = data_corrected.iloc[::n, :]


    initial_time = data_thinned['Timestamp'].iloc[0]
    data_thinned['Seconds'] = (data_thinned['Timestamp'] - initial_time).dt.total_seconds()


    plt.figure(figsize=(12, 6))


    plt.plot(data_thinned['Seconds'], data_thinned[y1], label=y1,  linestyle='-', linewidth=1.5)

    plt.plot(data_thinned['Seconds'], data_thinned[y2], label=y2, linestyle='-', linewidth=1.5)

    plt.title(f'{title} over time', fontsize=fontsize)
    plt.xlabel('Time (seconds)', fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.legend()


    plt.ylim(min(data_corrected[[y1, y2]].min()), max(data_corrected[[y1, y2]].max()))

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{date}/{title}_{date}.eps", dpi=dpi)
    plt.show()


