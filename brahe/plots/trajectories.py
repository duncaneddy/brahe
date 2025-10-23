"""
Orbital element time series visualization.

Provides 2D plots of Keplerian and Cartesian orbital elements over time.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import brahe as bh
from brahe.plots.backend import validate_backend, apply_scienceplots_style


def plot_cartesian_trajectory(
    trajectories,
    time_range=None,
    position_units="km",
    velocity_units="km/s",
    backend="matplotlib",
):
    """Plot Cartesian orbital elements (position and velocity) vs time.

    Creates a 2x3 subplot layout:
    - Row 1: x, y, z positions
    - Row 2: vx, vy, vz velocities

    Args:
        trajectories (list of dict): List of trajectory groups, each with:
            - trajectory: OrbitTrajectory or numpy array [N×6] or [N×7]
            - times (np.ndarray, optional): Time array if trajectory is numpy array without time column
            - color (str, optional): Line/marker color
            - marker (str, optional): Marker style
            - label (str, optional): Legend label

        time_range (tuple, optional): (start_epoch, end_epoch) to filter data
        position_units (str, optional): 'm' or 'km'. Default: 'km'
        velocity_units (str, optional): 'm/s' or 'km/s'. Default: 'km/s'
        backend (str, optional): 'matplotlib' or 'plotly'. Default: 'matplotlib'

    Returns:
        Figure object (matplotlib.figure.Figure or plotly.graph_objects.Figure)

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Create trajectory
        eop = bh.FileEOPProvider.from_default_standard(bh.EarthOrientationFileType.STANDARD, True)
        bh.set_global_eop_provider(eop)

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        oe = np.array([bh.R_EARTH + 500e3, 0.01, np.radians(97.8), 0.0, 0.0, 0.0])
        state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)

        prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)
        traj = prop.propagate(epoch, epoch + 2*bh.orbital_period(oe[0]), 60.0)

        # Plot Cartesian elements
        fig = bh.plot_cartesian_trajectory(
            [{"trajectory": traj, "label": "LEO Orbit"}],
            position_units='km',
            velocity_units='km/s',
            backend='matplotlib'
        )
        ```
    """
    validate_backend(backend)

    # Normalize inputs
    traj_groups = _normalize_trajectory_groups(trajectories)

    # Dispatch to backend
    if backend == "matplotlib":
        return _cartesian_elements_matplotlib(
            traj_groups, time_range, position_units, velocity_units
        )
    else:  # plotly
        return _cartesian_elements_plotly(
            traj_groups, time_range, position_units, velocity_units
        )


def plot_keplerian_trajectory(
    trajectories,
    time_range=None,
    angle_units="deg",
    sma_units="km",
    backend="matplotlib",
):
    """Plot Keplerian orbital elements vs time.

    Creates a 2x3 subplot layout:
    - Row 1: a (semi-major axis), e (eccentricity), i (inclination)
    - Row 2: Ω (RAAN), ω (argument of periapsis), ν (true anomaly)

    Args:
        trajectories (list of dict): List of trajectory groups, each with:
            - trajectory: OrbitTrajectory or numpy array [N×6] or [N×7] as [a, e, i, Ω, ω, ν]
            - times (np.ndarray, optional): Time array if trajectory is numpy array without time column
            - color (str, optional): Line/marker color
            - marker (str, optional): Marker style
            - label (str, optional): Legend label

        time_range (tuple, optional): (start_epoch, end_epoch) to filter data
        angle_units (str, optional): 'rad' or 'deg'. Default: 'deg'
        sma_units (str, optional): 'm' or 'km'. Default: 'km'
        backend (str, optional): 'matplotlib' or 'plotly'. Default: 'matplotlib'

    Returns:
        Figure object (matplotlib.figure.Figure or plotly.graph_objects.Figure)

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Create trajectory
        eop = bh.FileEOPProvider.from_default_standard(bh.EarthOrientationFileType.STANDARD, True)
        bh.set_global_eop_provider(eop)

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        oe = np.array([bh.R_EARTH + 500e3, 0.01, np.radians(97.8), 0.0, 0.0, 0.0])
        state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)

        prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)
        traj = prop.propagate(epoch, epoch + 2*bh.orbital_period(oe[0]), 60.0)

        # Plot Keplerian elements
        fig = bh.plot_keplerian_trajectory(
            [{"trajectory": traj, "label": "LEO Orbit"}],
            angle_units='deg',
            sma_units='km',
            backend='matplotlib'
        )
        ```
    """
    validate_backend(backend)

    # Normalize inputs
    traj_groups = _normalize_trajectory_groups(trajectories)

    # Dispatch to backend
    if backend == "matplotlib":
        return _keplerian_elements_matplotlib(
            traj_groups, time_range, angle_units, sma_units
        )
    else:  # plotly
        return _keplerian_elements_plotly(
            traj_groups, time_range, angle_units, sma_units
        )


def _normalize_trajectory_groups(trajectories):
    """Normalize trajectory input to list of dicts with defaults."""
    defaults = {
        "times": None,
        "color": None,
        "marker": None,
        "label": None,
    }

    if trajectories is None:
        return []

    if not isinstance(trajectories, list):
        return [{**defaults, "trajectory": trajectories}]

    if len(trajectories) == 0:
        return []

    if not isinstance(trajectories[0], dict):
        # List of trajectories without config
        return [{**defaults, "trajectory": t} for t in trajectories]

    # List of dicts - apply defaults
    return [{**defaults, **group} for group in trajectories]


def _cartesian_elements_matplotlib(
    traj_groups, time_range, position_units, velocity_units
):
    """Matplotlib implementation of Cartesian elements plot."""
    # Apply scienceplots if available
    apply_scienceplots_style()

    # Create 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Cartesian Orbital Elements")

    # Unit conversions (for future use in TODO)
    # pos_scale = 1.0 if position_units == 'm' else 1e-3
    # vel_scale = 1.0 if velocity_units == 'm/s' else 1e-3

    # Plot each trajectory
    for i, group in enumerate(traj_groups):
        trajectory = group.get("trajectory")
        color = group.get("color")
        marker = group.get("marker", "-")
        label = group.get("label")

        if trajectory is None:
            continue

        # Extract times and states
        if hasattr(trajectory, "states"):
            # OrbitTrajectory
            times = trajectory.times()
            states = trajectory.states()

            # Extract positions and velocities
            pos_x = [state[0] for state in states]
            pos_y = [state[1] for state in states]
            pos_z = [state[2] for state in states]
            vel_x = [state[3] for state in states]
            vel_y = [state[4] for state in states]
            vel_z = [state[5] for state in states]

        elif isinstance(trajectory, np.ndarray):
            # Numpy array [N×6] or [N×7]
            if trajectory.shape[1] >= 6:
                if trajectory.shape[1] == 7:
                    # Has time column
                    times = trajectory[:, 0]
                    pos_x = trajectory[:, 1]
                    pos_y = trajectory[:, 2]
                    pos_z = trajectory[:, 3]
                    vel_x = trajectory[:, 4]
                    vel_y = trajectory[:, 5]
                    vel_z = trajectory[:, 6]
                else:
                    # No time column, use indices
                    times = np.arange(len(trajectory))
                    pos_x = trajectory[:, 0]
                    pos_y = trajectory[:, 1]
                    pos_z = trajectory[:, 2]
                    vel_x = trajectory[:, 3]
                    vel_y = trajectory[:, 4]
                    vel_z = trajectory[:, 5]
            else:
                continue
        else:
            continue

        # Apply unit conversions
        if position_units == "km":
            pos_x = np.array(pos_x) * 1e-3
            pos_y = np.array(pos_y) * 1e-3
            pos_z = np.array(pos_z) * 1e-3

        if velocity_units == "km/s":
            vel_x = np.array(vel_x) * 1e-3
            vel_y = np.array(vel_y) * 1e-3
            vel_z = np.array(vel_z) * 1e-3

        # Plot positions
        axes[0, 0].plot(times, pos_x, marker, color=color, label=label)
        axes[0, 1].plot(times, pos_y, marker, color=color, label=label)
        axes[0, 2].plot(times, pos_z, marker, color=color, label=label)

        # Plot velocities
        axes[1, 0].plot(times, vel_x, marker, color=color, label=label)
        axes[1, 1].plot(times, vel_y, marker, color=color, label=label)
        axes[1, 2].plot(times, vel_z, marker, color=color, label=label)

    # Configure subplots
    axes[0, 0].set_ylabel(f"X Position ({position_units})")
    axes[0, 1].set_ylabel(f"Y Position ({position_units})")
    axes[0, 2].set_ylabel(f"Z Position ({position_units})")
    axes[1, 0].set_ylabel(f"VX Velocity ({velocity_units})")
    axes[1, 1].set_ylabel(f"VY Velocity ({velocity_units})")
    axes[1, 2].set_ylabel(f"VZ Velocity ({velocity_units})")

    for ax in axes.flat:
        ax.set_xlabel("Time")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    return fig


def _cartesian_elements_plotly(traj_groups, time_range, position_units, velocity_units):
    """Plotly implementation of Cartesian elements plot."""
    # Create 2x3 subplot layout
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "X Position",
            "Y Position",
            "Z Position",
            "VX Velocity",
            "VY Velocity",
            "VZ Velocity",
        ),
    )

    # Plot each trajectory
    for i, group in enumerate(traj_groups):
        trajectory = group.get("trajectory")
        color = group.get("color")
        label = group.get("label")

        if trajectory is None:
            continue

        # Extract times and states
        if hasattr(trajectory, "states"):
            # OrbitTrajectory
            times = trajectory.times()
            states = trajectory.states()

            # Extract positions and velocities
            pos_x = [state[0] for state in states]
            pos_y = [state[1] for state in states]
            pos_z = [state[2] for state in states]
            vel_x = [state[3] for state in states]
            vel_y = [state[4] for state in states]
            vel_z = [state[5] for state in states]

        elif isinstance(trajectory, np.ndarray):
            # Numpy array [N×6] or [N×7]
            if trajectory.shape[1] >= 6:
                if trajectory.shape[1] == 7:
                    # Has time column
                    times = trajectory[:, 0]
                    pos_x = trajectory[:, 1]
                    pos_y = trajectory[:, 2]
                    pos_z = trajectory[:, 3]
                    vel_x = trajectory[:, 4]
                    vel_y = trajectory[:, 5]
                    vel_z = trajectory[:, 6]
                else:
                    # No time column, use indices
                    times = np.arange(len(trajectory))
                    pos_x = trajectory[:, 0]
                    pos_y = trajectory[:, 1]
                    pos_z = trajectory[:, 2]
                    vel_x = trajectory[:, 3]
                    vel_y = trajectory[:, 4]
                    vel_z = trajectory[:, 5]
            else:
                continue
        else:
            continue

        # Apply unit conversions
        if position_units == "km":
            pos_x = np.array(pos_x) * 1e-3
            pos_y = np.array(pos_y) * 1e-3
            pos_z = np.array(pos_z) * 1e-3

        if velocity_units == "km/s":
            vel_x = np.array(vel_x) * 1e-3
            vel_y = np.array(vel_y) * 1e-3
            vel_z = np.array(vel_z) * 1e-3

        # Plot positions
        fig.add_trace(
            go.Scatter(
                x=times,
                y=pos_x,
                mode="lines",
                name=label,
                line=dict(color=color),
                showlegend=(i == 0),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=pos_y,
                mode="lines",
                name=label,
                line=dict(color=color),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=pos_z,
                mode="lines",
                name=label,
                line=dict(color=color),
                showlegend=False,
            ),
            row=1,
            col=3,
        )

        # Plot velocities
        fig.add_trace(
            go.Scatter(
                x=times,
                y=vel_x,
                mode="lines",
                name=label,
                line=dict(color=color),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=vel_y,
                mode="lines",
                name=label,
                line=dict(color=color),
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=vel_z,
                mode="lines",
                name=label,
                line=dict(color=color),
                showlegend=False,
            ),
            row=2,
            col=3,
        )

    # Update axes labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_xaxes(title_text="Time", row=1, col=3)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    fig.update_xaxes(title_text="Time", row=2, col=3)

    fig.update_yaxes(title_text=f"X Position ({position_units})", row=1, col=1)
    fig.update_yaxes(title_text=f"Y Position ({position_units})", row=1, col=2)
    fig.update_yaxes(title_text=f"Z Position ({position_units})", row=1, col=3)
    fig.update_yaxes(title_text=f"VX Velocity ({velocity_units})", row=2, col=1)
    fig.update_yaxes(title_text=f"VY Velocity ({velocity_units})", row=2, col=2)
    fig.update_yaxes(title_text=f"VZ Velocity ({velocity_units})", row=2, col=3)

    fig.update_layout(title_text="Cartesian Orbital Elements")
    return fig


def _keplerian_elements_matplotlib(traj_groups, time_range, angle_units, sma_units):
    """Matplotlib implementation of Keplerian elements plot."""
    # Apply scienceplots if available
    apply_scienceplots_style()

    # Create 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Keplerian Orbital Elements")

    # Unit conversions (for future use in TODO)
    # sma_scale = 1.0 if sma_units == 'm' else 1e-3
    # angle_scale = 1.0 if angle_units == 'rad' else 180.0 / np.pi

    # Plot each trajectory
    for i, group in enumerate(traj_groups):
        trajectory = group.get("trajectory")
        color = group.get("color")
        marker = group.get("marker", "-")
        label = group.get("label")

        if trajectory is None:
            continue

        # Extract times and elements
        if hasattr(trajectory, "states"):
            # OrbitTrajectory - need to convert Cartesian to Keplerian
            times = trajectory.times()
            states = trajectory.states()

            # Convert each state to Keplerian elements
            a_list = []
            e_list = []
            i_list = []
            raan_list = []
            argp_list = []
            anom_list = []

            for state in states:
                # Convert Cartesian to Keplerian (returns radians)
                oe = bh.state_cartesian_to_osculating(state, bh.AngleFormat.RADIANS)
                a_list.append(oe[0])
                e_list.append(oe[1])
                i_list.append(oe[2])
                raan_list.append(oe[3])
                argp_list.append(oe[4])
                anom_list.append(oe[5])

        elif isinstance(trajectory, np.ndarray):
            # Numpy array [N×6] or [N×7] - assume already Keplerian [a, e, i, Ω, ω, ν]
            if trajectory.shape[1] >= 6:
                if trajectory.shape[1] == 7:
                    # Has time column
                    times = trajectory[:, 0]
                    a_list = trajectory[:, 1]
                    e_list = trajectory[:, 2]
                    i_list = trajectory[:, 3]
                    raan_list = trajectory[:, 4]
                    argp_list = trajectory[:, 5]
                    anom_list = trajectory[:, 6]
                else:
                    # No time column
                    times = np.arange(len(trajectory))
                    a_list = trajectory[:, 0]
                    e_list = trajectory[:, 1]
                    i_list = trajectory[:, 2]
                    raan_list = trajectory[:, 3]
                    argp_list = trajectory[:, 4]
                    anom_list = trajectory[:, 5]
            else:
                continue
        else:
            continue

        # Convert to numpy arrays for unit conversion
        a_arr = np.array(a_list)
        e_arr = np.array(e_list)
        i_arr = np.array(i_list)
        raan_arr = np.array(raan_list)
        argp_arr = np.array(argp_list)
        anom_arr = np.array(anom_list)

        # Apply unit conversions
        if sma_units == "km":
            a_arr = a_arr * 1e-3

        if angle_units == "deg":
            i_arr = np.degrees(i_arr)
            raan_arr = np.degrees(raan_arr)
            argp_arr = np.degrees(argp_arr)
            anom_arr = np.degrees(anom_arr)

        # Plot elements
        axes[0, 0].plot(times, a_arr, marker, color=color, label=label)
        axes[0, 1].plot(times, e_arr, marker, color=color, label=label)
        axes[0, 2].plot(times, i_arr, marker, color=color, label=label)
        axes[1, 0].plot(times, raan_arr, marker, color=color, label=label)
        axes[1, 1].plot(times, argp_arr, marker, color=color, label=label)
        axes[1, 2].plot(times, anom_arr, marker, color=color, label=label)

    # Configure subplots
    axes[0, 0].set_ylabel(f"Semi-major Axis ({sma_units})")
    axes[0, 1].set_ylabel("Eccentricity")
    axes[0, 2].set_ylabel(f"Inclination ({angle_units})")
    axes[1, 0].set_ylabel(f"RAAN ({angle_units})")
    axes[1, 1].set_ylabel(f"Arg. Periapsis ({angle_units})")
    axes[1, 2].set_ylabel(f"True Anomaly ({angle_units})")

    for ax in axes.flat:
        ax.set_xlabel("Time")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    return fig


def _keplerian_elements_plotly(traj_groups, time_range, angle_units, sma_units):
    """Plotly implementation of Keplerian elements plot."""
    # Create 2x3 subplot layout
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Semi-major Axis",
            "Eccentricity",
            "Inclination",
            "RAAN",
            "Arg. Periapsis",
            "True Anomaly",
        ),
    )

    # Plot each trajectory
    for i, group in enumerate(traj_groups):
        trajectory = group.get("trajectory")
        color = group.get("color")
        label = group.get("label")

        if trajectory is None:
            continue

        # Extract times and elements
        if hasattr(trajectory, "states"):
            # OrbitTrajectory - need to convert Cartesian to Keplerian
            times = trajectory.times()
            states = trajectory.states()

            # Convert each state to Keplerian elements
            a_list = []
            e_list = []
            i_list = []
            raan_list = []
            argp_list = []
            anom_list = []

            for state in states:
                # Convert Cartesian to Keplerian (returns radians)
                oe = bh.state_cartesian_to_osculating(state, bh.AngleFormat.RADIANS)
                a_list.append(oe[0])
                e_list.append(oe[1])
                i_list.append(oe[2])
                raan_list.append(oe[3])
                argp_list.append(oe[4])
                anom_list.append(oe[5])

        elif isinstance(trajectory, np.ndarray):
            # Numpy array [N×6] or [N×7] - assume already Keplerian [a, e, i, Ω, ω, ν]
            if trajectory.shape[1] >= 6:
                if trajectory.shape[1] == 7:
                    # Has time column
                    times = trajectory[:, 0]
                    a_list = trajectory[:, 1]
                    e_list = trajectory[:, 2]
                    i_list = trajectory[:, 3]
                    raan_list = trajectory[:, 4]
                    argp_list = trajectory[:, 5]
                    anom_list = trajectory[:, 6]
                else:
                    # No time column
                    times = np.arange(len(trajectory))
                    a_list = trajectory[:, 0]
                    e_list = trajectory[:, 1]
                    i_list = trajectory[:, 2]
                    raan_list = trajectory[:, 3]
                    argp_list = trajectory[:, 4]
                    anom_list = trajectory[:, 5]
            else:
                continue
        else:
            continue

        # Convert to numpy arrays for unit conversion
        a_arr = np.array(a_list)
        e_arr = np.array(e_list)
        i_arr = np.array(i_list)
        raan_arr = np.array(raan_list)
        argp_arr = np.array(argp_list)
        anom_arr = np.array(anom_list)

        # Apply unit conversions
        if sma_units == "km":
            a_arr = a_arr * 1e-3

        if angle_units == "deg":
            i_arr = np.degrees(i_arr)
            raan_arr = np.degrees(raan_arr)
            argp_arr = np.degrees(argp_arr)
            anom_arr = np.degrees(anom_arr)

        # Plot elements
        fig.add_trace(
            go.Scatter(
                x=times,
                y=a_arr,
                mode="lines",
                name=label,
                line=dict(color=color),
                showlegend=(i == 0),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=e_arr,
                mode="lines",
                name=label,
                line=dict(color=color),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=i_arr,
                mode="lines",
                name=label,
                line=dict(color=color),
                showlegend=False,
            ),
            row=1,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=raan_arr,
                mode="lines",
                name=label,
                line=dict(color=color),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=argp_arr,
                mode="lines",
                name=label,
                line=dict(color=color),
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=anom_arr,
                mode="lines",
                name=label,
                line=dict(color=color),
                showlegend=False,
            ),
            row=2,
            col=3,
        )

    # Update axes labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_xaxes(title_text="Time", row=1, col=3)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    fig.update_xaxes(title_text="Time", row=2, col=3)

    fig.update_yaxes(title_text=f"Semi-major Axis ({sma_units})", row=1, col=1)
    fig.update_yaxes(title_text="Eccentricity", row=1, col=2)
    fig.update_yaxes(title_text=f"Inclination ({angle_units})", row=1, col=3)
    fig.update_yaxes(title_text=f"RAAN ({angle_units})", row=2, col=1)
    fig.update_yaxes(title_text=f"Arg. Periapsis ({angle_units})", row=2, col=2)
    fig.update_yaxes(title_text=f"True Anomaly ({angle_units})", row=2, col=3)

    fig.update_layout(title_text="Keplerian Orbital Elements")
    return fig
