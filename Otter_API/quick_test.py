import numpy as np
import matplotlib.pyplot as plt

# ================= PARAMETERS =================
theta = 90  # semi-angle at half power [degrees]
P_total = 20  # transmitted optical power per LED [W]
Adet = 1e-4  # detector physical area [m^2]

# Optics parameters
Ts = 1.0
index = 1.5
FOV = 100 * np.pi / 180  # receiver FOV [rad]
G_Con = (index ** 2) / (np.sin(FOV) ** 2)

# Space dimensions
lx, ly = 20, 20  # meters
Nx, Ny = lx * 30, ly * 30

XT, YT = 0, 0  # LED position

x = np.linspace(-lx / 2, lx / 2, Nx)
y = np.linspace(-ly / 2, ly / 2, Ny)
XR, YR = np.meshgrid(x, y)

# Heights to compare
heights = [1, 5, 10, 30]

# ================= LAMBERTIAN ORDER =================
theta_eff = min(theta, 89.999)
m = -np.log(2) / np.log(np.cos(np.deg2rad(theta_eff)))

print(f"Lambertian order m = {m:.4f}")

# Store results for 3D plots
results = []

# ================= 2D CONTOUR PLOTS =================
fig2d, axes2d = plt.subplots(2, 2, figsize=(12, 10))

for i, h in enumerate(heights):
    D = np.sqrt((XR - XT) ** 2 + (YR - YT) ** 2 + h ** 2)

    cos_phi = h / D
    cos_psi = h / D

    psi = np.arccos(cos_psi)
    inside_fov = psi <= FOV

    H = ((m + 1) * Adet / (2 * np.pi * D ** 2)) * (cos_phi ** m) * cos_psi

    P_rec = P_total * H * Ts * G_Con
    P_rec[~inside_fov] = 0

    P_rec = np.maximum(P_rec, 1e-15)
    P_rec_dBm = 10 * np.log10(P_rec / 1e-3)

    results.append((h, P_rec_dBm))

    ax = axes2d.flat[i]
    contour = ax.contourf(XR, YR, P_rec_dBm, levels=100, cmap="viridis")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"2D received power, h = {h} m")
    ax.set_xlim([-lx / 2, lx / 2])
    ax.set_ylim([-ly / 2, ly / 2])
    ax.set_aspect("equal")

    fig2d.colorbar(contour, ax=ax, label="Received power (dBm)")

plt.tight_layout()
plt.show()


# ================= 3D SURFACE PLOTS =================
fig3d = plt.figure(figsize=(14, 11))

for i, (h, P_rec_dBm) in enumerate(results):
    ax = fig3d.add_subplot(2, 2, i + 1, projection="3d")

    surf = ax.plot_surface(
        XR,
        YR,
        P_rec_dBm,
        cmap="viridis",
        linewidth=0,
        antialiased=True
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Received power (dBm)")
    ax.set_title(f"3D received power, h = {h} m")

    ax.set_xlim([-lx / 2, lx / 2])
    ax.set_ylim([-ly / 2, ly / 2])

    fig3d.colorbar(surf, ax=ax, shrink=0.6, label="Received power (dBm)")

plt.tight_layou