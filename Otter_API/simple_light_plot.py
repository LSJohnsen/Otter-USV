import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Grid
x = np.linspace(0, 100, 400)
z = np.linspace(0, 50, 250)
X, Z = np.meshgrid(x, z)

# Source position
x0, z0 = 0, 0

# Distance
R = np.sqrt((X - x0)**2 + (Z - z0)**2)
R = np.maximum(R, 0.5)  # avoid singularity

# Attenuation (tune this)
k = 0.05

# Intensity model
I = np.exp(-k * R) / (1 + 0.02 * R**2)


plt.figure(figsize=(10, 5))

im = plt.imshow(
    I,
    extent=[x.min(), x.max(), z.max(), z.min()],
    aspect='auto',
    cmap='jet',              # classic "physics" heatmap
    norm=LogNorm(vmin=I.min() + 1e-6, vmax=I.max())
)

plt.colorbar(label='Light Intensity')
plt.xlabel('Horizontal Range (m)')
plt.ylabel('Depth (m)')
plt.title('Underwater Light Propagation')
plt.show()