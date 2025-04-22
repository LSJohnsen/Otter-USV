import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def plot(date):

    data = pd.read_csv(f'{date}.csv', sep=';')
    dpi = 600
    fontsize = 14


    data_corrected = data.iloc[3:]


    data_corrected['north_from_observer'] = pd.to_numeric(data_corrected['north_from_observer'], errors='coerce')
    data_corrected['east_from_observer'] = pd.to_numeric(data_corrected['east_from_observer'], errors='coerce')
    data_corrected['target_north_from_observer'] = pd.to_numeric(data_corrected['target_north_from_observer'], errors='coerce')
    data_corrected['target_east_from_observer'] = pd.to_numeric(data_corrected['target_east_from_observer'], errors='coerce')


    plt.figure(figsize=(10, 6))


    plt.plot(data_corrected['east_from_observer'], data_corrected['north_from_observer'], label='Ship Path', linestyle='-', linewidth=1.5)

    plt.plot(data_corrected['target_east_from_observer'], data_corrected['target_north_from_observer'], label='Target Path', linestyle='-', linewidth=1.5)

    plt.title('Ship and Target Paths', fontsize=fontsize)
    plt.xlabel('East from Observer (m)', fontsize=fontsize)
    plt.ylabel('North from Observer (m)', fontsize=fontsize)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/{date}/Paths_{date}.eps", dpi=dpi)
    plt.show()


