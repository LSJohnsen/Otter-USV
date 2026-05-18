import os
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Save directory
# =============================================================================
SAVE_DIR = "nmpc_dp_cost_plots"
os.makedirs(SAVE_DIR, exist_ok=True)


def save_figure(fig, name: str):
    fig.savefig(os.path.join(SAVE_DIR, f"{name}.pdf"), bbox_inches="tight")


# =============================================================================
# NMPC dynamic-positioning scaling / cost functions
# Matches the structure in your stationkeeping NMPC
# =============================================================================

def sigma_approach(distance, d_approach):
    """
    sigma_approach = d^2 / (d^2 + d_approach^2)

    Far from target  -> close to 1
    Near target      -> close to 0
    """
    distance = np.asarray(distance, dtype=float)
    return distance**2 / (distance**2 + d_approach**2)


def sigma_hold(distance, d_hold):
    """
    sigma_hold = d^2 / (d^2 + d_hold^2)

    Far from target  -> close to 1
    Near target      -> close to 0
    """
    distance = np.asarray(distance, dtype=float)
    return distance**2 / (distance**2 + d_hold**2)


def near_target(distance, d_hold):
    """
    near_target = 1 - sigma_hold
    """
    return 1.0 - sigma_hold(distance, d_hold)


def heading_cost(distance, dpsi, d_approach, w_psi):
    """
    heading_cost = sigma_approach(d)^2 * w_psi * (1 - cos(dpsi))
    """
    return sigma_approach(distance, d_approach) ** 2 * w_psi * (1.0 - np.cos(dpsi))


def radial_damping_cost(distance, v_radial, d_hold, radial_weight):
    """
    radial_damping_cost = radial_weight * near_target(d) * v_radial^2
    """
    return radial_weight * near_target(distance, d_hold) * v_radial**2


def tangential_damping_cost(distance, v_tangential, d_hold, tangential_weight):
    """
    tangential_damping_cost = tangential_weight * near_target(d) * v_tangential^2
    """
    return tangential_weight * near_target(distance, d_hold) * v_tangential**2


def surge_misalignment_cost(distance, dpsi, tau_x, d_approach):
    """
    surge_misalignment_cost =
        0.03 * sigma_approach(d) * (1 - heading_alignment) * tau_x^2

    heading_alignment = (1 + cos(dpsi)) / 2
    """
    heading_alignment = (1.0 + np.cos(dpsi)) / 2.0
    return 0.03 * sigma_approach(distance, d_approach) * (1.0 - heading_alignment) * tau_x**2



def style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()



# Distance-based scaling functions


def plot_dp_distance_scaling(d_approach=6.0, d_hold=2.5):
    d = np.linspace(0.0, 15.0, 600)

    sigma_a = sigma_approach(d, d_approach)
    sigma_h = sigma_hold(d, d_hold)
    near_t = 1.0 - sigma_h

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(d, sigma_a, label=r"$\sigma_{\mathrm{approach}}$")
    ax.plot(d, sigma_h, label=r"$\sigma_{\mathrm{hold}}$")
    ax.plot(d, near_t, label=r"$1-\sigma_{\mathrm{hold}}$")

    style_axis(
        ax,
        rf"NMPC dynamic-positioning distance scaling "
        rf"($d_{{approach}}={d_approach}$ m, $d_{{hold}}={d_hold}$ m)",
        "Distance to target [m]",
        "Scaling value",
    )

    plt.tight_layout()
    save_figure(fig, "nmpc_dp_distance_scaling")
    plt.show()



#  Heading-cost multiplier vs distance


def plot_dp_heading_scaling(d_approach=6.0):
    d = np.linspace(0.0, 15.0, 600)
    heading_scale = sigma_approach(d, d_approach) ** 2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(d, heading_scale, label=r"$\sigma_{\mathrm{approach}}^2$")

    style_axis(
        ax,
        rf"Heading-cost scaling for dynamic positioning "
        rf"($d_{{approach}}={d_approach}$ m)",
        "Distance to target [m]",
        "Heading-cost multiplier",
    )

    plt.tight_layout()
    save_figure(fig, "nmpc_dp_heading_scaling")
    plt.show()



# Velocity-damping activation vs distance


def plot_dp_velocity_damping_scaling(d_hold=2.5, radial_weight=15.0, tangential_weight=25.0):
    d = np.linspace(0.0, 15.0, 600)

    near_t = near_target(d, d_hold)
    radial_activation = radial_weight * near_t
    tangential_activation = tangential_weight * near_t

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(d, radial_activation, label="Radial damping activation")
    ax.plot(d, tangential_activation, label="Tangential damping activation")

    style_axis(
        ax,
        rf"Velocity-damping activation for dynamic positioning "
        rf"($d_{{hold}}={d_hold}$ m, $w_r={radial_weight}$, $w_t={tangential_weight}$)",
        "Distance to target [m]",
        "Effective damping weight",
    )

    plt.tight_layout()
    save_figure(fig, "nmpc_dp_velocity_damping_scaling")
    plt.show()



# Heading cost vs heading error, for different distances


def plot_dp_heading_cost_vs_error(d_approach=6.0, w_psi=5.0, distances=(0.5, 2.5, 6.0, 10.0)):
    dpsi = np.linspace(-np.pi, np.pi, 600)

    fig, ax = plt.subplots(figsize=(10, 6))

    for d in distances:
        vals = heading_cost(d, dpsi, d_approach, w_psi)
        ax.plot(dpsi, vals, label=rf"$d={d}$ m")

    style_axis(
        ax,
        rf"Heading cost vs heading error "
        rf"($d_{{approach}}={d_approach}$ m, $w_\psi={w_psi}$)",
        "Heading error $d\\psi$ [rad]",
        "Heading cost",
    )

    plt.tight_layout()
    save_figure(fig, "nmpc_dp_heading_cost_vs_error")
    plt.show()



# Radial damping cost vs radial velocity


def plot_dp_radial_cost_vs_velocity(d_hold=2.5, radial_weight=15.0, distances=(0.2, 1.0, 2.5, 6.0)):
    v_radial = np.linspace(-2.0, 2.0, 600)

    fig, ax = plt.subplots(figsize=(10, 6))

    for d in distances:
        vals = radial_damping_cost(d, v_radial, d_hold, radial_weight)
        ax.plot(v_radial, vals, label=rf"$d={d}$ m")

    style_axis(
        ax,
        rf"Radial damping cost vs radial velocity "
        rf"($d_{{hold}}={d_hold}$ m, $w_r={radial_weight}$)",
        "Radial velocity $v_{radial}$ [m/s]",
        "Radial damping cost",
    )

    plt.tight_layout()
    save_figure(fig, "nmpc_dp_radial_damping_cost")
    plt.show()


#  Tangential damping cost vs tangential velocity


def plot_dp_tangential_cost_vs_velocity(d_hold=2.5, tangential_weight=25.0, distances=(0.2, 1.0, 2.5, 6.0)):
    v_tang = np.linspace(-2.0, 2.0, 600)

    fig, ax = plt.subplots(figsize=(10, 6))

    for d in distances:
        vals = tangential_damping_cost(d, v_tang, d_hold, tangential_weight)
        ax.plot(v_tang, vals, label=rf"$d={d}$ m")

    style_axis(
        ax,
        rf"Tangential damping cost vs tangential velocity "
        rf"($d_{{hold}}={d_hold}$ m, $w_t={tangential_weight}$)",
        "Tangential velocity $v_{tangential}$ [m/s]",
        "Tangential damping cost",
    )

    plt.tight_layout()
    save_figure(fig, "nmpc_dp_tangential_damping_cost")
    plt.show()



#  Surge misalignment cost vs heading error


def plot_dp_surge_misalignment_vs_error(
    d_approach=6.0,
    tau_x=100.0,
    distances=(0.5, 2.5, 6.0, 10.0),
):
    dpsi = np.linspace(-np.pi, np.pi, 600)

    fig, ax = plt.subplots(figsize=(10, 6))

    for d in distances:
        vals = surge_misalignment_cost(d, dpsi, tau_x, d_approach)
        ax.plot(dpsi, vals, label=rf"$d={d}$ m")

    style_axis(
        ax,
        rf"Surge-misalignment cost vs heading error "
        rf"($d_{{approach}}={d_approach}$ m, $\tau_X={tau_x}$ N)",
        "Heading error $d\\psi$ [rad]",
        "Surge-misalignment cost",
    )

    plt.tight_layout()
    save_figure(fig, "nmpc_dp_surge_misalignment_cost")
    plt.show()



if __name__ == "__main__":
    # Typical dynamic-positioning parameters from your NMPC
    d_approach = 6.0
    d_hold = 2.5
    w_psi = 5.0
    radial_weight = 15.0
    tangential_weight = 25.0
    tau_x_example = 100.0

    # Scaling plots
    plot_dp_distance_scaling(d_approach=d_approach, d_hold=d_hold)
    plot_dp_heading_scaling(d_approach=d_approach)
    plot_dp_velocity_damping_scaling(
        d_hold=d_hold,
        radial_weight=radial_weight,
        tangential_weight=tangential_weight,
    )

    # Cost-shape plots
    plot_dp_heading_cost_vs_error(
        d_approach=d_approach,
        w_psi=w_psi,
        distances=(0.5, 2.5, 6.0, 10.0),
    )

    plot_dp_radial_cost_vs_velocity(
        d_hold=d_hold,
        radial_weight=radial_weight,
        distances=(0.2, 1.0, 2.5, 6.0),
    )

    plot_dp_tangential_cost_vs_velocity(
        d_hold=d_hold,
        tangential_weight=tangential_weight,
        distances=(0.2, 1.0, 2.5, 6.0),
    )

    plot_dp_surge_misalignment_vs_error(
        d_approach=d_approach,
        tau_x=tau_x_example,
        distances=(0.5, 2.5, 6.0, 10.0),
    )