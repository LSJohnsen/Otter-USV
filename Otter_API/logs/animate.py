import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')

def plot(date):
    column_names = [
        'Timestamp', 'lat', 'lon', 'height', 'previous_lat', 'previous_lon',
        'previous_height', 'last_speed_update', 'current_course_over_ground',
        'current_speed', 'current_fuel_capacity', 'current_orientation_1',
        'current_orientation_2', 'current_orientation_3',
        'current_rotational_velocities_1', 'current_rotational_velocities_2',
        'current_rotational_velocities_3', 'observer_lat', 'observer_lon',
        'observer_height', 'north_from_observer', 'east_from_observer',
        'down_from_observer', 'previous_time', 'cycle_time', 'speed_n', 'speed_e', 'speed_surge', 'speed_sway', 'n1', 'n2', 'north_error', 'east_error',
        'distance_to_target', 'yaw_setpoint', 'current_angle', 'tau_X', 'tau_N',
        'target_north_from_observer', 'target_east_from_observer'
    ]



    data = pd.read_csv(f'{date}.csv', sep=';', names=column_names, skiprows=4)



    if data.empty:
        raise ValueError("No data available after skipping rows.")


    for col in ['north_from_observer', 'east_from_observer',
                'target_north_from_observer', 'target_east_from_observer']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        if data[col].isna().all():
            raise ValueError(f"All values in {col} are NaN.")



    fig, ax = plt.subplots()

    ship_line, = ax.plot([], [], linestyle='-', linewidth=1, label='Ship Path')
    target_line, = ax.plot([], [], linestyle='--', linewidth=1, label='Target Path')
    ax.legend()


    def init():

        ax.set_xlim([min(data['east_from_observer'].min(), data['target_east_from_observer'].min()),
                     max(data['east_from_observer'].max(), data['target_east_from_observer'].max())])
        ax.set_ylim([min(data['north_from_observer'].min(), data['target_north_from_observer'].min()),
                     max(data['north_from_observer'].max(), data['target_north_from_observer'].max())])
        ship_line.set_data([], [])
        target_line.set_data([], [])
        return ship_line, target_line

    def update(frame):

        ship_line.set_data(data['east_from_observer'].iloc[:frame], data['north_from_observer'].iloc[:frame])

        target_line.set_data(data['target_east_from_observer'].iloc[:frame], data['target_north_from_observer'].iloc[:frame])
        return ship_line, target_line


    ani = FuncAnimation(fig, update, frames=len(data), init_func=init, blit=False)


    ani.save(f'plots/{date}/animation_{date}.gif', writer='pillow', fps=20)


    plt.show()


