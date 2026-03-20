import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as colors

SAVE_DIR = "reward_plots"
os.makedirs(SAVE_DIR, exist_ok=True)


def save_figure(fig, name):
    fig.savefig(os.path.join(SAVE_DIR, f"{name}.pdf"), bbox_inches="tight")

# Reward plots

def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def compute_reward_components(
    *,
    d: float,
    u: float,
    v: float,
    r: float,
    psi: float,
    last_distance: float,
    sampletime: float,
    target_heading_ref: float,
    target_pos: tuple[float, float],
    eta_pos: tuple[float, float],
    target_speed: float,
    use_moving_target: bool,
    stationary_heading_ref: float,
    hold_time: float,
    hold_time_required: float,
    tau_cmd: tuple[float, float],
    prev_cmd: tuple[float, float],
    tauX_max: float,
    tauN_max: float,
    success: bool = False,
    # tuning parameters from your reward structure
    sigma_p: float = 1.5,
    C_p: float = 1.0,
    sigma_psi: float = 0.6,
    C_psi: float = 0.5,
    sigma_u: float = 0.5,
    C_u: float = 0.5,
    sigma_v: float = 0.5,
    C_v: float = 0.5,
    K_d: float = 3.0,
    C_d_dot: float = 1.0,
    d_acc: float = 1.0,
    C_r: float = 1.2,
    d_opt: float = 0.1,
    C_t: float = 0.001,
    C_a: float = 0.01,
    alpha_u: float = 0.1,
    final_scale: float = 0.1,
) -> dict[str, float]:
    """Compute each reward component separately using your current reward design."""
    # Distance derivative term used in reward
    d_dot = (d - last_distance) / sampletime  # positive = moving away, negative = closing

    # Relative range weight
    in_range = np.clip((d_acc - d) / d_acc, 0.0, 1.0)  # 1 when close, 0 when outside acceptable range

    # Position reward
    r_pos = C_p * np.exp(-((d - d_opt) ** 2) / (2 * sigma_p**2))  # Gaussian distance reward

    # Distance-rate reward
    r_d_dot = -C_d_dot * np.tanh(K_d * d_dot)  # positive when closing, negative when moving away

    # Heading reward terms
    e_track = wrap_to_pi(target_heading_ref - psi)  # heading error relative to target trajectory
    heading_scale = np.clip(1.0 - abs(e_track) / np.pi, 0.0, 1.0)  # reduce yaw-rate penalty when heading error is large

    if use_moving_target:
        psi_los = float(np.arctan2(target_pos[1] - eta_pos[1], target_pos[0] - eta_pos[0]))  # LOS heading to target
        e_los = wrap_to_pi(psi_los - psi)  # LOS heading error
        r_heading = (
            C_psi * (1.0 - in_range) * np.exp(-(e_los**2) / (2 * sigma_psi**2))
            + C_psi * in_range * np.exp(-(e_track**2) / (2 * sigma_psi**2))
        )  # far: LOS alignment, close: target-track alignment
    else:
        e_hold = wrap_to_pi(stationary_heading_ref - psi)  # heading error for stationary hold
        r_heading = C_psi * np.exp(-(e_hold**2) / (2 * sigma_psi**2))  # Gaussian heading reward for docking

    r_heading2 = -in_range * heading_scale * C_r * abs(r)  # yaw-rate penalty near target

    # Surge reward
    u_far = 1.0  # desired surge when far away
    u_close = target_speed if use_moving_target else 0.0  # desired surge near target
    u_d = u_far if d > sigma_p else u_close  # piecewise desired surge
    r_surge = (C_u + alpha_u) * np.exp(-((u - u_d) ** 2) / (2 * sigma_u**2)) - alpha_u  # Gaussian surge reward

    # Relative velocity reward in world frame
    vx_usv = u * np.cos(psi) - v * np.sin(psi)  # vessel world-frame x velocity
    vy_usv = u * np.sin(psi) + v * np.cos(psi)  # vessel world-frame y velocity
    vx_t = target_speed * np.cos(target_heading_ref) if use_moving_target else 0.0  # target x velocity
    vy_t = target_speed * np.sin(target_heading_ref) if use_moving_target else 0.0  # target y velocity
    e_vx = vx_usv - vx_t  # relative x velocity
    e_vy = vy_usv - vy_t  # relative y velocity
    e_v = np.sqrt(e_vx**2 + e_vy**2)  # relative speed magnitude
    r_vel = in_range * C_v * np.exp(-(e_v**2) / (2 * sigma_v**2))  # reward for matching target velocity near target

    # Time penalty
    r_time = C_t  # constant per-step penalty

    # Action penalty
    scale = np.array([tauX_max, tauN_max], dtype=float)  # actuator normalization
    delta_cmd = (np.array(tau_cmd, dtype=float) - np.array(prev_cmd, dtype=float)) / scale  # normalized command change
    r_action = C_a * (abs(delta_cmd[0]) + abs(delta_cmd[1]))  # penalty on aggressive command changes

    # Hold reward
    t_short = hold_time_required / 5.0  # short-hold timescale
    t_long = hold_time_required  # full-hold timescale
    hold_ratio_short = np.clip(hold_time / max(t_short, 1e-6), 0.0, 1.0) ** 2  # faster increase for sustained short hold
    hold_ratio_long = np.clip(hold_time / max(t_long, 1e-6), 0.0, 1.0)  # long-hold completion ratio
    r_hold = in_range * (0.2 * hold_ratio_short + 0.8 * hold_ratio_long)  # sustained hold reward

    # Success bonus
    r_success = 5.0 if success else 0.0  # terminal success bonus

    # Total reward before and after final scale
    total_unscaled = (
        r_pos
        + r_d_dot
        + r_heading
        + r_heading2
        + r_surge
        + r_vel
        - r_time
        - r_action
        + r_hold
        + r_success
    )
    total_scaled = final_scale * total_unscaled  # final reward returned by environment

    return {
        "r_pos": float(r_pos),
        "r_d_dot": float(r_d_dot),
        "r_heading": float(r_heading),
        "r_heading2": float(r_heading2),
        "r_surge": float(r_surge),
        "r_vel": float(r_vel),
        "r_time": float(r_time),
        "r_action": float(r_action),
        "r_hold": float(r_hold),
        "r_success": float(r_success),
        "total_unscaled": float(total_unscaled),
        "total_scaled": float(total_scaled),
        "in_range": float(in_range),
        "d_dot": float(d_dot),
        "u_d": float(u_d),
        "heading_scale": float(heading_scale),
    }



def _style_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str = "Reward value") -> None:
    """Apply a consistent axis style."""
    ax.set_title(title)  # subplot title
    ax.set_xlabel(xlabel)  # x-axis label
    ax.set_ylabel(ylabel)  # y-axis label
    ax.grid(True)  # enable grid
    ax.legend()  # show legend


def plot_reward_vs_heading_error(*, use_moving_target: bool = True) -> None:
    """Plot heading-related reward terms as a function of heading error."""
    e_values = np.linspace(-np.pi, np.pi, 400)  # sweep heading error over full range

    r_heading_vals = []  # Gaussian heading reward curve
    r_heading2_vals = []  # yaw-rate penalty curve
    total_vals = []  # total reward curve for this slice

    for e in e_values:
        # choose psi so that target_heading_ref - psi = e
        target_heading_ref = 0.0  # fixed reference heading
        psi = wrap_to_pi(target_heading_ref - e)  # implied vessel heading for desired error

        comps = compute_reward_components(
            d=0.3,  # near-target condition so heading matters
            u=0.4 if use_moving_target else 0.0,  # representative surge
            v=0.0,
            r=0.3,  # representative yaw rate
            psi=psi,
            last_distance=0.31,  # slightly larger previous distance -> closing
            sampletime=0.1,
            target_heading_ref=target_heading_ref,
            target_pos=(1.0, 0.0),  # target ahead on x-axis
            eta_pos=(0.0, 0.0),
            target_speed=0.4 if use_moving_target else 0.0,
            use_moving_target=use_moving_target,
            stationary_heading_ref=0.0,
            hold_time=5.0,
            hold_time_required=10.0,
            tau_cmd=(10.0, 4.0),
            prev_cmd=(9.5, 3.5),
            tauX_max=150.0,
            tauN_max=110.0,
            success=False,
        )
        r_heading_vals.append(comps["r_heading"])  # store heading reward
        r_heading2_vals.append(comps["r_heading2"])  # store yaw-rate penalty
        total_vals.append(comps["total_scaled"])  # store total reward

    fig, ax = plt.subplots(figsize=(10, 6))  # create figure for heading sweep
    ax.plot(e_values, r_heading_vals, label="Target heading reward")  # plot heading reward
    ax.plot(e_values, r_heading2_vals, label="Yaw-rate penalty")  # plot yaw-rate penalty
    _style_axis(ax, "Heading rewards", "Heading error [rad]")
    plt.tight_layout()  # improve spacing
    fig.savefig("heading_reward.pdf", bbox_inches="tight")
    plt.show()  # display plot


def plot_reward_vs_d_dot() -> None:
    """Plot the distance-rate reward term as a function of d_dot."""
    d_dot_values = np.linspace(-2.0, 2.0, 400)  # sweep approach/divergence rate
    vals = []  # store r_d_dot values

    for d_dot in d_dot_values:
        d = 1.0  # fixed representative distance
        last_distance = d - d_dot * 0.1  # invert d_dot = (d - last_distance)/dt with dt=0.1

        comps = compute_reward_components(
            d=d,
            u=0.5,
            v=0.0,
            r=0.0,
            psi=0.0,
            last_distance=last_distance,
            sampletime=0.1,
            target_heading_ref=0.0,
            target_pos=(d, 0.0),
            eta_pos=(0.0, 0.0),
            target_speed=0.5,
            use_moving_target=True,
            stationary_heading_ref=0.0,
            hold_time=0.0,
            hold_time_required=10.0,
            tau_cmd=(0.0, 0.0),
            prev_cmd=(0.0, 0.0),
            tauX_max=150.0,
            tauN_max=110.0,
            success=False,
        )
        vals.append(comps["r_d_dot"])  # store distance-rate reward

    fig, ax = plt.subplots(figsize=(10, 6))  # create figure for d_dot sweep
    ax.plot(d_dot_values, vals, label="Euclidean distance rate of change")  # plot distance-rate reward
    _style_axis(ax, "Distance rate of change reward", "d_dot [m/s]")
    plt.tight_layout()  # improve spacing
    fig.savefig("d_dot_reward.pdf", bbox_inches="tight")
    plt.show()  # display plot


def plot_reward_vs_surge_comparison():
    """Plot stationary and moving-target surge rewards side by side."""
    u_values = np.linspace(-1.0, 2.0, 400)  # sweep surge speed range

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # two side-by-side subplots

    for ax, use_moving_target, title in zip(
        axes,
        [False, True],  # stationary first, moving second
        ["Stationary target", "Moving target"]
    ):
        vals = []  # store surge reward values for this mode

        for u in u_values:
            comps = compute_reward_components(
                d=0.1 if not use_moving_target else 0.3,  # representative close-target distance
                u=u,  # swept surge velocity
                v=0.0,  # zero sway for this slice
                r=0.0,  # zero yaw rate for this slice
                psi=0.0,  # vessel heading
                last_distance=0.11 if not use_moving_target else 0.31,  # slightly larger previous distance
                sampletime=0.1,  # same as training step
                target_heading_ref=0.0,  # target moving along x-axis
                target_pos=(1.0, 0.0),  # target position in world frame
                eta_pos=(0.0, 0.0),  # vessel at origin
                target_speed=0.0 if not use_moving_target else 0.6,  # target speed for each case
                use_moving_target=use_moving_target,  # switch between stationary and moving target
                stationary_heading_ref=0.0,  # stationary heading reference
                hold_time=0.0,  # no hold-time contribution for this slice
                hold_time_required=10.0,  # same as training setup
                tau_cmd=(0.0, 0.0),  # zero action change for isolated surge plot
                prev_cmd=(0.0, 0.0),  # previous action
                tauX_max=150.0,  # actuator scaling
                tauN_max=110.0,  # actuator scaling
                success=False,  # no terminal bonus
            )
            vals.append(comps["r_surge"])  # store surge reward only

        ax.plot(u_values, vals, label="surge reward")  # plot surge reward for this mode
        ax.set_xlim(u_values[0] - 0.1, u_values[-1] + 0.1)
        _style_axis(ax, title, "Surge velocity u [m/s]")  # style subplot

    axes[0].set_ylabel("Reward value")  # shared y-axis label
    plt.tight_layout()  # improve spacing
    fig.savefig("surge_rewards.pdf", bbox_inches="tight")
    plt.show()  # display plot

def plot_reward_vs_relative_speed() -> None:
    """Plot the relative-velocity reward as a function of relative speed magnitude."""
    e_v_values = np.linspace(0.0, 2.0, 400)  # sweep relative speed magnitude
    vals = []  # store relative-velocity reward values

    for e_v in e_v_values:
        # realize the requested relative speed by setting vessel surge while target moves along x-axis
        target_speed = 0.6  # representative target speed
        u = target_speed + e_v  # choose vessel speed so relative speed magnitude equals e_v

        comps = compute_reward_components(
            d=0.2,  # close-target condition so r_vel is active
            u=u,
            v=0.0,
            r=0.0,
            psi=0.0,
            last_distance=0.21,
            sampletime=0.1,
            target_heading_ref=0.0,
            target_pos=(1.0, 0.0),
            eta_pos=(0.0, 0.0),
            target_speed=target_speed,
            use_moving_target=True,
            stationary_heading_ref=0.0,
            hold_time=4.0,
            hold_time_required=10.0,
            tau_cmd=(0.0, 0.0),
            prev_cmd=(0.0, 0.0),
            tauX_max=150.0,
            tauN_max=110.0,
            success=False,
        )
        vals.append(comps["r_vel"])  # store relative velocity reward

    fig, ax = plt.subplots(figsize=(10, 6))  # create figure for relative speed sweep
    ax.plot(e_v_values, vals, label="relative velocity reward")  # plot relative-velocity reward
    _style_axis(ax, "Relative velocity reward", "Relative velocity magnitude [m/s]")
    plt.tight_layout()  # improve spacing
    fig.savefig("relative_velocity_reward.pdf", bbox_inches="tight")
    plt.show()  # display plot


def plot_reward_vs_hold_time() -> None:
    """Plot hold reward as a function of accumulated hold time."""
    t_values = np.linspace(0.0, 12.0, 400)  # sweep hold time beyond required hold duration
    vals = []  # store hold reward values

    for hold_time in t_values:
        comps = compute_reward_components(
            d=0.1,  # close enough to be fully in range
            u=0.0,
            v=0.0,
            r=0.0,
            psi=0.0,
            last_distance=0.1,
            sampletime=0.1,
            target_heading_ref=0.0,
            target_pos=(1.0, 0.0),
            eta_pos=(0.0, 0.0),
            target_speed=0.0,
            use_moving_target=False,
            stationary_heading_ref=0.0,
            hold_time=hold_time,
            hold_time_required=10.0,
            tau_cmd=(0.0, 0.0),
            prev_cmd=(0.0, 0.0),
            tauX_max=150.0,
            tauN_max=110.0,
            success=False,
        )
        vals.append(comps["r_hold"])  # store hold reward

    fig, ax = plt.subplots(figsize=(10, 6))  # create figure for hold-time sweep
    ax.plot(t_values, vals, label="Hold time reward")  # plot hold reward
    _style_axis(ax, "Hold reward vs accumulated hold time", "Hold time [s]")
    plt.tight_layout()  # improve spacing
    fig.savefig("hold_reward.pdf", bbox_inches="tight")
    plt.show()  # display plot


def plot_reward_contributions(*, use_moving_target: bool = True):
    """Plot reward terms as a horizontal signed bar chart."""
    comps = compute_reward_components(
        d=0.3 if use_moving_target else 0.1,          # representative distance
        u=0.6 if use_moving_target else 0.0,          # representative surge
        v=0.05,                                       # small sway
        r=0.1,                                        # small yaw rate
        psi=0.1,                                      # slight heading offset
        last_distance=0.32,                           # slightly larger previous distance
        sampletime=0.1,                               # sample time
        target_heading_ref=0.0,                       # target heading
        target_pos=(1.0, 0.0),                        # target position
        eta_pos=(0.0, 0.0),                           # vessel position
        target_speed=0.6 if use_moving_target else 0.0,  # target speed
        use_moving_target=use_moving_target,          # moving/stationary case
        stationary_heading_ref=0.0,                   # stationary heading reference
        hold_time=6.0,                                # representative hold time
        hold_time_required=10.0,                      # required hold time
        tau_cmd=(20.0, 5.0),                          # current command
        prev_cmd=(18.0, 4.0),                         # previous command
        tauX_max=150.0,                               # surge limit
        tauN_max=110.0,                               # yaw limit
        success=False,                                # no success bonus
    )

    labels = [
        "Position reward",
        "Distance-rate reward",
        "Heading reward",
        "Yaw-rate penalty",
        "Surge reward",
        "Velocity-matching reward",
        "Hold reward",
    ]  # user-friendly labels

    values = [
        comps["r_pos"],
        comps["r_d_dot"],
        comps["r_heading"],
        comps["r_heading2"],
        comps["r_surge"],
        comps["r_vel"],
        comps["r_hold"],
    ]  # signed contributions before final scaling

    order = np.argsort(np.abs(values))[::-1]  # sort by magnitude, largest first
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["green" if v >= 0 else "red" for v in values]  # positive/negative coloring

    ax.barh(labels, values, color=colors)                    # horizontal signed bars
    ax.axvline(0.0, color="black", linewidth=1)             # zero reference line

    mode_label = "Moving target" if use_moving_target else "Stationary target"
    ax.set_title(f"Reward term contributions")
    ax.set_xlabel("Contribution before final scaling")
    ax.set_ylabel("Reward term")
    ax.grid(True, axis="x")

    plt.tight_layout()
    fig.savefig("total_reward_term_contributions.pdf", bbox_inches="tight")
    plt.show()


def plot_r_pos_only():
    """Plot the Gaussian distance reward r_pos on its own."""
    d_values = np.linspace(0.0, 5.0, 400)
    vals = []

    for d in d_values:
        comps = compute_reward_components(
            d=d,
            u=0.0,
            v=0.0,
            r=0.0,
            psi=0.0,
            last_distance=d,
            sampletime=0.1,
            target_heading_ref=0.0,
            target_pos=(d, 0.0),
            eta_pos=(0.0, 0.0),
            target_speed=0.0,
            use_moving_target=False,
            stationary_heading_ref=0.0,
            hold_time=0.0,
            hold_time_required=10.0,
            tau_cmd=(0.0, 0.0),
            prev_cmd=(0.0, 0.0),
            tauX_max=150.0,
            tauN_max=110.0,
            success=False,
        )
        vals.append(comps["r_pos"])  # store only position reward

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(d_values, vals, label="Distance to target reward")
    _style_axis(ax, "Euclidean distance reward", "Distance to target d [m]")
    plt.tight_layout()
    fig.savefig("distance_reward.pdf", bbox_inches="tight")
    plt.show()


def plot_reward_vs_action_change_heatmap():
    """Plot signed action-penalty contribution over normalized surge/yaw command changes."""
    dx_vals = np.linspace(-1.0, 1.0, 200)  # normalized surge command change
    dn_vals = np.linspace(-1.0, 1.0, 200)  # normalized yaw command change

    Z = np.zeros((len(dn_vals), len(dx_vals)))  # heatmap values

    for i, dn in enumerate(dn_vals):
        for j, dx in enumerate(dx_vals):
            tau_cmd = (dx * 150.0, dn * 110.0)  # map normalized changes to physical commands
            prev_cmd = (0.0, 0.0)               # compare against zero previous command

            comps = compute_reward_components(
                d=0.3,
                u=0.6,
                v=0.0,
                r=0.0,
                psi=0.0,
                last_distance=0.31,
                sampletime=0.1,
                target_heading_ref=0.0,
                target_pos=(1.0, 0.0),
                eta_pos=(0.0, 0.0),
                target_speed=0.6,
                use_moving_target=True,
                stationary_heading_ref=0.0,
                hold_time=0.0,
                hold_time_required=10.0,
                tau_cmd=tau_cmd,
                prev_cmd=prev_cmd,
                tauX_max=150.0,
                tauN_max=110.0,
                success=False,
            )

            Z[i, j] = -comps["r_action"]  # signed contribution used in total reward

    fig, ax = plt.subplots(figsize=(8, 6))
    norm = colors.Normalize(vmin=np.min(Z), vmax=0.0)

    im = ax.imshow(
        Z,
        extent=[dx_vals[0], dx_vals[-1], dn_vals[0], dn_vals[-1]],
        origin="lower",
        aspect="auto",
        cmap="coolwarm_r",   # blue → red
        norm=norm)
    ax.set_title("Action-penalty contribution")
    ax.set_xlabel("Normalized surge command change")
    ax.set_ylabel("Normalized yaw command change")
    fig.colorbar(im, ax=ax, label="Contribution to reward")
    plt.tight_layout()
    fig.savefig("command_penalty_heatmap.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    #plot_r_pos_only()  # standalone r_pos plot

    # Example plots. Comment out any you do not want.
    """ plot_reward_vs_heading_error(use_moving_target=True)  # heading-error sweep
    plot_reward_vs_d_dot()  # distance-rate sweep
    plot_reward_vs_surge_comparison()
    plot_reward_vs_relative_speed()  # relative-velocity reward sweep
    plot_reward_vs_hold_time()  # hold-time reward sweep
    plot_reward_contributions(use_moving_target=True)  # single operating-point reward breakdown """
    plot_reward_vs_action_change_heatmap()
   