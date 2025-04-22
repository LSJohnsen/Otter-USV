# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:57:30 2024

@author: 47900
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.colors import LinearSegmentedColormap, Normalize

file_path = r'C:\Users\47900\Documents\Robotikk\Otter API\lib\throttle_map_v2_neg.csv'
df_new = pd.read_csv(file_path, delimiter=';', header=0, index_col=0)

colors = ["darkblue", "white", "darkred"]
custom_colormap = LinearSegmentedColormap.from_list("custom_colormap", colors)
norm = Normalize(vmin=-900, vmax=1200)


throttle_positions = df_new.index.astype(float).values


engine_speeds = df_new.columns.astype(float).values


z_values_left = []
z_values_right = []


for index, row in df_new.iterrows():
    row_values_left = []
    row_values_right = []
    for cell in row:
        numbers = re.findall(r'-?\d+\.?\d*', cell)

        row_values_left.append(float(numbers[0]) if numbers else 0)
        row_values_right.append(float(numbers[1]) if len(numbers) > 1 else 0)
    z_values_left.append(row_values_left)
    z_values_right.append(row_values_right)


z_array_left = np.array(z_values_left)
z_array_right = np.array(z_values_right)


fig, ax1 = plt.subplots(figsize=(10, 5))
cp_left = ax1.contourf(engine_speeds, throttle_positions, z_array_left, cmap=custom_colormap, norm=norm, levels=np.linspace(-900, 1200, num=22))
fig.colorbar(cp_left, ax=ax1)
ax1.set_title('Left Thruster (RPM)')
ax1.set_xlabel('Throttle Input x-direction')
ax1.set_ylabel('Throttle Input y-direction')
ax1.grid(True)


output_file_path_left = r'C:\Users\47900\Documents\Robotikk\Otter API\lib\throttle_map_v2_left_plot.eps'
plt.savefig(output_file_path_left, format="eps")
plt.show() 
plt.close()


fig, ax2 = plt.subplots(figsize=(10, 5))
cp_right = ax2.contourf(engine_speeds, throttle_positions, z_array_right, cmap=custom_colormap, norm=norm, levels=np.linspace(-900, 1200, num=22))
fig.colorbar(cp_right, ax=ax2)
ax2.set_title('Right Thruster (RPM)')
ax2.set_xlabel('Throttle Input x-direction', fontsize=12)
ax2.set_ylabel('Throttle Input y-direction', fontsize=12)
ax2.grid(True)

output_file_path_right = r'C:\Users\47900\Documents\Robotikk\Otter API\lib\throttle_map_v2_right_plot.eps'
plt.savefig(output_file_path_right, format="eps")
plt.show()  
plt.close()


print(output_file_path_left)
print(output_file_path_right)