"""
Orbital element time series visualization.

Provides 2D plots of Keplerian and Cartesian orbital elements over time.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

import brahe as bh
from brahe.plots.backend import validate_backend, apply_scienceplots_style


def plot_cartesian_trajectory(
    trajectories,
    time_range=None,
    position_units="km",
    velocity_units="km/s",
    backend="matplotlib",
    show_title=False,
    show_grid=False,
    matplotlib_config=None,
    plotly_config=None,
    width=None,
    height=None,
) -> object:
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
        show_title (bool, optional): Whether to display plot title. Default: False
        show_grid (bool, optional): Whether to display grid lines. Default: False
        matplotlib_config (dict, optional): Matplotlib-specific configuration:
            - legend_subplot (tuple): (row, col) of subplot for legend. Default: (0, 0)
            - legend_loc (str): Legend location. Default: 'best'
              Options: 'best', 'upper right', 'upper left', 'lower left', 'lower right',
                       'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
            - dark_mode (bool): Apply dark mode styling. Default: False
            - ylabel_pad (float): Padding for y-axis labels. Default: 10
            - figsize (tuple): Figure size (width, height). Default: (15, 10)
        plotly_config (dict, optional): Plotly-specific configuration (reserved for future use)
        width (int, optional): Figure width in pixels (plotly only). Default: None (responsive)
        height (int, optional): Figure height in pixels (plotly only). Default: None (responsive)

    Returns:
        Generated figure object

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

        # Plot Cartesian elements with legend in upper right of Z position subplot
        fig = bh.plot_cartesian_trajectory(
            [{"trajectory": traj, "label": "LEO Orbit"}],
            position_units='km',
            velocity_units='km/s',
            backend='matplotlib',
            matplotlib_config={'legend_subplot': (0, 2), 'legend_loc': 'upper right'}
        )
        ```
    """
    start_time = time.time()
    logger.info(f"Plotting Cartesian trajectory with backend={backend}")
    logger.debug(f"Units: position={position_units}, velocity={velocity_units}")

    validate_backend(backend)

    # Normalize inputs
    traj_groups = _normalize_trajectory_groups(trajectories)

    # Extract backend-specific config with defaults
    if matplotlib_config is None:
        matplotlib_config = {}
    if plotly_config is None:
        plotly_config = {}

    # Dispatch to backend
    if backend == "matplotlib":
        legend_subplot = matplotlib_config.get("legend_subplot", (0, 0))
        legend_loc = matplotlib_config.get("legend_loc", "best")
        dark_mode = matplotlib_config.get("dark_mode", False)
        ylabel_pad = matplotlib_config.get("ylabel_pad", 10)
        figsize = matplotlib_config.get("figsize", (15, 10))
        result = _cartesian_elements_matplotlib(
            traj_groups,
            time_range,
            position_units,
            velocity_units,
            show_title,
            show_grid,
            legend_subplot,
            legend_loc,
            dark_mode,
            ylabel_pad,
            figsize,
        )
    else:  # plotly
        result = _cartesian_elements_plotly(
            traj_groups,
            time_range,
            position_units,
            velocity_units,
            show_title,
            show_grid,
            width,
            height,
        )

    elapsed = time.time() - start_time
    logger.info(f"Cartesian trajectory plot completed in {elapsed:.2f}s")
    return result


def plot_keplerian_trajectory(
    trajectories,
    time_range=None,
    angle_units="deg",
    sma_units="km",
    normalize_angles=False,
    backend="matplotlib",
    show_title=False,
    show_grid=False,
    matplotlib_config=None,
    plotly_config=None,
    width=None,
    height=None,
) -> object:
    """Plot Keplerian orbital elements vs time.

    Creates a 2x3 subplot layout:
    - Row 1: a (semi-major axis), e (eccentricity), i (inclination)
    - Row 2: Ω (RAAN), ω (argument of periapsis), M (mean anomaly)

    Args:
        trajectories (list of dict): List of trajectory groups, each with:
            - trajectory: OrbitTrajectory or numpy array [N×6] or [N×7] as [a, e, i, Ω, ω, M]
            - times (np.ndarray, optional): Time array if trajectory is numpy array without time column
            - color (str, optional): Line/marker color
            - marker (str, optional): Marker style
            - label (str, optional): Legend label

        time_range (tuple, optional): (start_epoch, end_epoch) to filter data
        angle_units (str, optional): 'rad' or 'deg'. Default: 'deg'
        sma_units (str, optional): 'm' or 'km'. Default: 'km'
        normalize_angles (bool, optional): If True, wrap angles to [0, 2π) or [0, 360°). Default: False
        backend (str, optional): 'matplotlib' or 'plotly'. Default: 'matplotlib'
        show_title (bool, optional): Whether to display plot title. Default: False
        show_grid (bool, optional): Whether to display grid lines. Default: False
        matplotlib_config (dict, optional): Matplotlib-specific configuration:
            - legend_subplot (tuple): (row, col) of subplot for legend. Default: (0, 0)
            - legend_loc (str): Legend location. Default: 'best'
              Options: 'best', 'upper right', 'upper left', 'lower left', 'lower right',
                       'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
            - dark_mode (bool): Apply dark mode styling. Default: False
            - ylabel_pad (float): Padding for y-axis labels. Default: 10
            - figsize (tuple): Figure size (width, height). Default: (15, 10)
            - set_angle_ylim (bool): Set y-axis limits to [0, 360°] or [0, 2π]. Default: False
        plotly_config (dict, optional): Plotly-specific configuration:
            - set_angle_ylim (bool): Set y-axis limits to [0, 360°] or [0, 2π]. Default: False
        width (int, optional): Figure width in pixels (plotly only). Default: None (responsive)
        height (int, optional): Figure height in pixels (plotly only). Default: None (responsive)

    Returns:
        Generated figure object

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Create trajectory
        eop = bh.FileEOPProvider.from_default_standard(bh.EarthOrientationFileType.STANDARD, True)
        bh.set_global_eop_provider(eop)

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        oe = np.array([bh.R_EARTH + 500e3, 0.01, 97.8, 0.0, 0.0, 0.0])

        prop = bh.KeplerianPropagator.from_eci(epoch, oe, bh.AngleFormat.DEGREES, 60.0)
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
    start_time = time.time()
    logger.info(f"Plotting Keplerian trajectory with backend={backend}")
    logger.debug(f"Units: angle={angle_units}, sma={sma_units}")

    validate_backend(backend)

    # Normalize inputs
    traj_groups = _normalize_trajectory_groups(trajectories)

    # Extract backend-specific config with defaults
    if matplotlib_config is None:
        matplotlib_config = {}
    if plotly_config is None:
        plotly_config = {}

    # Dispatch to backend
    if backend == "matplotlib":
        legend_subplot = matplotlib_config.get("legend_subplot", (0, 0))
        legend_loc = matplotlib_config.get("legend_loc", "best")
        dark_mode = matplotlib_config.get("dark_mode", False)
        ylabel_pad = matplotlib_config.get("ylabel_pad", 10)
        figsize = matplotlib_config.get("figsize", (15, 10))
        set_angle_ylim = matplotlib_config.get("set_angle_ylim", False)
        result = _keplerian_elements_matplotlib(
            traj_groups,
            time_range,
            angle_units,
            sma_units,
            normalize_angles,
            show_title,
            show_grid,
            legend_subplot,
            legend_loc,
            dark_mode,
            ylabel_pad,
            figsize,
            set_angle_ylim,
        )
    else:  # plotly
        set_angle_ylim = plotly_config.get("set_angle_ylim", False)
        result = _keplerian_elements_plotly(
            traj_groups,
            time_range,
            angle_units,
            sma_units,
            normalize_angles,
            show_title,
            show_grid,
            set_angle_ylim,
            width,
            height,
        )

    elapsed = time.time() - start_time
    logger.info(f"Keplerian trajectory plot completed in {elapsed:.2f}s")
    return result


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
    traj_groups,
    time_range,
    position_units,
    velocity_units,
    show_title,
    show_grid,
    legend_subplot,
    legend_loc,
    dark_mode,
    ylabel_pad,
    figsize,
):
    """Matplotlib implementation of Cartesian elements plot."""
    # Apply scienceplots if available
    apply_scienceplots_style()

    # Apply dark mode if requested
    if dark_mode:
        plt.style.use("dark_background")

    # Create 2x3 subplot layout with better spacing
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    if show_title:
        fig.suptitle("Cartesian Orbital Elements", y=0.995)
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    # Track first epoch and max time for relative time calculation
    first_epoch = None
    max_time = 0

    # Plot each trajectory
    for i, group in enumerate(traj_groups):
        trajectory = group.get("trajectory")
        color = group.get("color")
        marker = group.get("marker", "-")
        label = group.get("label")

        if trajectory is None:
            continue

        # Extract times and states
        if hasattr(trajectory, "to_matrix"):
            # OrbitTrajectory - use to_matrix() which returns (N, 6) array
            epochs = trajectory.epochs()
            states = trajectory.to_matrix()

            # Convert epochs to elapsed time in seconds
            if first_epoch is None:
                first_epoch = epochs[0]
            times = np.array([e - first_epoch for e in epochs])
            max_time = max(max_time, times[-1])

            # Extract positions and velocities from (N, 6) array
            pos_x = states[:, 0]
            pos_y = states[:, 1]
            pos_z = states[:, 2]
            vel_x = states[:, 3]
            vel_y = states[:, 4]
            vel_z = states[:, 5]

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

        # Determine time unit and convert if needed
        if first_epoch is not None:
            if max_time > 7200:  # > 2 hours, use hours
                times_plot = times / 3600
                time_label = "Time (hours)"
            elif max_time > 120:  # > 2 minutes, use minutes
                times_plot = times / 60
                time_label = "Time (minutes)"
            else:
                times_plot = times
                time_label = "Time (seconds)"
        else:
            times_plot = times
            time_label = "Time"

        # Apply unit conversions
        if position_units == "km":
            pos_x = np.array(pos_x) * 1e-3
            pos_y = np.array(pos_y) * 1e-3
            pos_z = np.array(pos_z) * 1e-3

        if velocity_units == "km/s":
            vel_x = np.array(vel_x) * 1e-3
            vel_y = np.array(vel_y) * 1e-3
            vel_z = np.array(vel_z) * 1e-3

        # Construct plot kwargs
        plot_kwargs = {}
        if color is not None:
            plot_kwargs["color"] = color
        if label is not None:
            plot_kwargs["label"] = label
        if marker is not None:
            plot_kwargs["linestyle"] = marker

        # Plot positions
        axes[0, 0].plot(times_plot, pos_x, **plot_kwargs)
        axes[0, 1].plot(times_plot, pos_y, **plot_kwargs)
        axes[0, 2].plot(times_plot, pos_z, **plot_kwargs)

        # Plot velocities
        axes[1, 0].plot(times_plot, vel_x, **plot_kwargs)
        axes[1, 1].plot(times_plot, vel_y, **plot_kwargs)
        axes[1, 2].plot(times_plot, vel_z, **plot_kwargs)

    # Set time label
    if first_epoch is None:
        time_label = "Time"
    # time_label already set in the loop

    # Set consistent x-axis limits across all subplots
    # Use actual max time value (without padding)
    if len(traj_groups) > 0:
        # Find the actual maximum time value from the data
        max_time_value = 0
        for ax in axes.flat:
            if ax.has_data():
                lines = ax.get_lines()
                for line in lines:
                    xdata = line.get_xdata()
                    if len(xdata) > 0:
                        max_time_value = max(max_time_value, xdata[-1])

        # Set x-limits to end exactly at the last data point
        for ax in axes.flat:
            if ax.has_data():
                ax.set_xlim(0, max_time_value)

    # Configure subplots
    axes[0, 0].set_ylabel(f"X Position ({position_units})")
    axes[0, 1].set_ylabel(f"Y Position ({position_units})")
    axes[0, 2].set_ylabel(f"Z Position ({position_units})")
    axes[1, 0].set_ylabel(f"VX Velocity ({velocity_units})")
    axes[1, 1].set_ylabel(f"VY Velocity ({velocity_units})")
    axes[1, 2].set_ylabel(f"VZ Velocity ({velocity_units})")

    # Apply ylabel padding for better alignment
    for ax in axes.flat:
        ax.set_xlabel(time_label)
        ax.yaxis.labelpad = ylabel_pad
        if show_grid:
            ax.grid(True)

    # Add legend only to specified subplot
    axes[legend_subplot].legend(loc=legend_loc)

    plt.tight_layout()
    return fig


def _cartesian_elements_plotly(
    traj_groups,
    time_range,
    position_units,
    velocity_units,
    show_title,
    show_grid,
    width,
    height,
):
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
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )

    # Track first epoch and max time for relative time calculation
    first_epoch = None
    max_time = 0

    # Plot each trajectory
    for i, group in enumerate(traj_groups):
        trajectory = group.get("trajectory")
        color = group.get("color")
        label = group.get("label")

        if trajectory is None:
            continue

        # Extract times and states
        if hasattr(trajectory, "to_matrix"):
            # OrbitTrajectory - use to_matrix() which returns (N, 6) array
            epochs = trajectory.epochs()
            states = trajectory.to_matrix()

            # Convert epochs to elapsed time in seconds
            if first_epoch is None:
                first_epoch = epochs[0]
            times = np.array([e - first_epoch for e in epochs])
            max_time = max(max_time, times[-1])

            # Extract positions and velocities from (N, 6) array
            pos_x = states[:, 0]
            pos_y = states[:, 1]
            pos_z = states[:, 2]
            vel_x = states[:, 3]
            vel_y = states[:, 4]
            vel_z = states[:, 5]

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

        # Determine time unit and convert if needed
        if first_epoch is not None:
            if max_time > 7200:  # > 2 hours, use hours
                times_plot = times / 3600
                time_label = "Time (hours)"
            elif max_time > 120:  # > 2 minutes, use minutes
                times_plot = times / 60
                time_label = "Time (minutes)"
            else:
                times_plot = times
                time_label = "Time (seconds)"
        else:
            times_plot = times
            time_label = "Time"

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
                x=times_plot,
                y=pos_x,
                mode="lines",
                name=label,
                line=dict(color=color),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times_plot,
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
                x=times_plot,
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
                x=times_plot,
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
                x=times_plot,
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
                x=times_plot,
                y=vel_z,
                mode="lines",
                name=label,
                line=dict(color=color),
                showlegend=False,
            ),
            row=2,
            col=3,
        )

    # Update axes labels with appropriate time unit
    if first_epoch is None:
        time_label = "Time"
    # time_label already set in the loop

    fig.update_xaxes(title_text=time_label, row=1, col=1)
    fig.update_xaxes(title_text=time_label, row=1, col=2)
    fig.update_xaxes(title_text=time_label, row=1, col=3)
    fig.update_xaxes(title_text=time_label, row=2, col=1)
    fig.update_xaxes(title_text=time_label, row=2, col=2)
    fig.update_xaxes(title_text=time_label, row=2, col=3)

    fig.update_yaxes(title_text=f"X Position ({position_units})", row=1, col=1)
    fig.update_yaxes(title_text=f"Y Position ({position_units})", row=1, col=2)
    fig.update_yaxes(title_text=f"Z Position ({position_units})", row=1, col=3)
    fig.update_yaxes(title_text=f"VX Velocity ({velocity_units})", row=2, col=1)
    fig.update_yaxes(title_text=f"VY Velocity ({velocity_units})", row=2, col=2)
    fig.update_yaxes(title_text=f"VZ Velocity ({velocity_units})", row=2, col=3)

    # Update layout with optional title and grid settings
    layout_config = {}
    if show_title:
        layout_config["title_text"] = "Cartesian Orbital Elements"

    # Only set width/height if explicitly provided
    if width is not None:
        layout_config["width"] = width
    if height is not None:
        layout_config["height"] = height

    fig.update_layout(**layout_config)

    # Configure grid and axis styling
    axis_config = {
        "showgrid": show_grid,
        "title_font": {"size": 11},
        "tickfont": {"size": 10},
    }
    fig.update_xaxes(**axis_config)
    fig.update_yaxes(**axis_config)

    # Make subplot titles smaller
    for annotation in fig.layout.annotations:
        annotation.font.size = 11

    return fig


def _keplerian_elements_matplotlib(
    traj_groups,
    time_range,
    angle_units,
    sma_units,
    normalize_angles,
    show_title,
    show_grid,
    legend_subplot,
    legend_loc,
    dark_mode,
    ylabel_pad,
    figsize,
    set_angle_ylim,
):
    """Matplotlib implementation of Keplerian elements plot."""
    # Apply scienceplots if available
    apply_scienceplots_style()

    # Apply dark mode if requested
    if dark_mode:
        plt.style.use("dark_background")

    # Create 2x3 subplot layout with better spacing
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    if show_title:
        fig.suptitle("Keplerian Orbital Elements", y=0.995)
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    # Track first epoch and max time for relative time calculation
    first_epoch = None
    max_time = 0

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
        if hasattr(trajectory, "to_matrix"):
            # OrbitTrajectory - need to convert Cartesian to Keplerian
            epochs = trajectory.epochs()
            states = trajectory.to_matrix()  # Returns (N, 6) array

            # Convert epochs to elapsed time in seconds
            if first_epoch is None:
                first_epoch = epochs[0]
            times = np.array([e - first_epoch for e in epochs])
            max_time = max(max_time, times[-1])

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
                argp_list.append(oe[4])  # omega (argument of periapsis)
                anom_list.append(oe[5])  # M (mean anomaly)

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

        # Determine time unit and convert if needed
        if first_epoch is not None:
            if max_time > 7200:  # > 2 hours, use hours
                times_plot = times / 3600
                time_label = "Time (hours)"
            elif max_time > 120:  # > 2 minutes, use minutes
                times_plot = times / 60
                time_label = "Time (minutes)"
            else:
                times_plot = times
                time_label = "Time (seconds)"
        else:
            times_plot = times
            time_label = "Time"

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

        # Normalize angles to [0, 2π) or [0, 360°) if requested
        if normalize_angles:
            if angle_units == "deg":
                i_arr = np.mod(i_arr, 360.0)
                raan_arr = np.mod(raan_arr, 360.0)
                argp_arr = np.mod(argp_arr, 360.0)
                anom_arr = np.mod(anom_arr, 360.0)
            else:  # radians
                i_arr = np.mod(i_arr, 2.0 * np.pi)
                raan_arr = np.mod(raan_arr, 2.0 * np.pi)
                argp_arr = np.mod(argp_arr, 2.0 * np.pi)
                anom_arr = np.mod(anom_arr, 2.0 * np.pi)

        # Construct plot kwargs
        plot_kwargs = {}
        if color is not None:
            plot_kwargs["color"] = color
        if label is not None:
            plot_kwargs["label"] = label
        if marker is not None:
            plot_kwargs["linestyle"] = marker

        # Plot elements
        axes[0, 0].plot(times_plot, a_arr, **plot_kwargs)
        axes[0, 1].plot(times_plot, e_arr, **plot_kwargs)
        axes[0, 2].plot(times_plot, i_arr, **plot_kwargs)
        axes[1, 0].plot(times_plot, raan_arr, **plot_kwargs)
        axes[1, 1].plot(times_plot, argp_arr, **plot_kwargs)
        axes[1, 2].plot(times_plot, anom_arr, **plot_kwargs)

    # Set time label
    if first_epoch is None:
        time_label = "Time"
    # time_label already set in the loop

    # Set consistent x-axis limits across all subplots
    # Use actual max time value (without padding)
    if len(traj_groups) > 0:
        # Find the actual maximum time value from the data
        max_time_value = 0
        for ax in axes.flat:
            if ax.has_data():
                lines = ax.get_lines()
                for line in lines:
                    xdata = line.get_xdata()
                    if len(xdata) > 0:
                        max_time_value = max(max_time_value, xdata[-1])

        # Set x-limits to end exactly at the last data point
        for ax in axes.flat:
            if ax.has_data():
                ax.set_xlim(0, max_time_value)

    # Configure subplots
    axes[0, 0].set_ylabel(f"Semi-major Axis ({sma_units})")
    axes[0, 1].set_ylabel("Eccentricity")
    axes[0, 2].set_ylabel(f"Inclination ({angle_units})")
    axes[1, 0].set_ylabel(f"RAAN ({angle_units})")
    axes[1, 1].set_ylabel(f"Arg. Periapsis ({angle_units})")
    axes[1, 2].set_ylabel(f"Mean Anomaly ({angle_units})")

    # Set y-axis limits for angle plots if requested
    if set_angle_ylim:
        if angle_units == "deg":
            angle_ylim = (0, 360)
        else:  # radians
            angle_ylim = (0, 2 * np.pi)

        # Apply to inclination, RAAN, argument of periapsis, and mean anomaly
        axes[0, 2].set_ylim(angle_ylim)  # Inclination
        axes[1, 0].set_ylim(angle_ylim)  # RAAN
        axes[1, 1].set_ylim(angle_ylim)  # Arg. Periapsis
        axes[1, 2].set_ylim(angle_ylim)  # Mean Anomaly

    # Apply ylabel padding for better alignment
    for ax in axes.flat:
        ax.set_xlabel(time_label)
        ax.yaxis.labelpad = ylabel_pad
        if show_grid:
            ax.grid(True)

    # Add legend only to specified subplot
    axes[legend_subplot].legend(loc=legend_loc)

    plt.tight_layout()
    return fig


def _keplerian_elements_plotly(
    traj_groups,
    time_range,
    angle_units,
    sma_units,
    normalize_angles,
    show_title,
    show_grid,
    set_angle_ylim,
    width,
    height,
):
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
            "Mean Anomaly",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )

    # Track first epoch and max time for relative time calculation
    first_epoch = None
    max_time = 0

    # Plot each trajectory
    for i, group in enumerate(traj_groups):
        trajectory = group.get("trajectory")
        color = group.get("color")
        label = group.get("label")

        if trajectory is None:
            continue

        # Extract times and elements
        if hasattr(trajectory, "to_matrix"):
            # OrbitTrajectory - need to convert Cartesian to Keplerian
            epochs = trajectory.epochs()
            states = trajectory.to_matrix()  # Returns (N, 6) array

            # Convert epochs to elapsed time in seconds
            if first_epoch is None:
                first_epoch = epochs[0]
            times = np.array([e - first_epoch for e in epochs])
            max_time = max(max_time, times[-1])

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
                argp_list.append(oe[4])  # omega (argument of periapsis)
                anom_list.append(oe[5])  # M (mean anomaly)

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

        # Normalize angles to [0, 2π) or [0, 360°) if requested
        if normalize_angles:
            if angle_units == "deg":
                i_arr = np.mod(i_arr, 360.0)
                raan_arr = np.mod(raan_arr, 360.0)
                argp_arr = np.mod(argp_arr, 360.0)
                anom_arr = np.mod(anom_arr, 360.0)
            else:  # radians
                i_arr = np.mod(i_arr, 2.0 * np.pi)
                raan_arr = np.mod(raan_arr, 2.0 * np.pi)
                argp_arr = np.mod(argp_arr, 2.0 * np.pi)
                anom_arr = np.mod(anom_arr, 2.0 * np.pi)

        # Determine time unit and convert if needed
        if first_epoch is not None:
            if max_time > 7200:  # > 2 hours, use hours
                times_plot = times / 3600
                time_label = "Time (hours)"
            elif max_time > 120:  # > 2 minutes, use minutes
                times_plot = times / 60
                time_label = "Time (minutes)"
            else:
                times_plot = times
                time_label = "Time (seconds)"
        else:
            times_plot = times
            time_label = "Time"

        # Plot elements
        fig.add_trace(
            go.Scatter(
                x=times_plot,
                y=a_arr,
                mode="lines",
                name=label,
                line=dict(color=color),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times_plot,
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
                x=times_plot,
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
                x=times_plot,
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
                x=times_plot,
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
                x=times_plot,
                y=anom_arr,
                mode="lines",
                name=label,
                line=dict(color=color),
                showlegend=False,
            ),
            row=2,
            col=3,
        )

    # Update axes labels with appropriate time unit
    if first_epoch is None:
        time_label = "Time"
    # time_label already set in the loop

    fig.update_xaxes(title_text=time_label, row=1, col=1)
    fig.update_xaxes(title_text=time_label, row=1, col=2)
    fig.update_xaxes(title_text=time_label, row=1, col=3)
    fig.update_xaxes(title_text=time_label, row=2, col=1)
    fig.update_xaxes(title_text=time_label, row=2, col=2)
    fig.update_xaxes(title_text=time_label, row=2, col=3)

    fig.update_yaxes(title_text=f"Semi-major Axis ({sma_units})", row=1, col=1)
    fig.update_yaxes(title_text="Eccentricity", row=1, col=2)
    fig.update_yaxes(title_text=f"Inclination ({angle_units})", row=1, col=3)
    fig.update_yaxes(title_text=f"RAAN ({angle_units})", row=2, col=1)
    fig.update_yaxes(title_text=f"Arg. Periapsis ({angle_units})", row=2, col=2)
    fig.update_yaxes(title_text=f"Mean Anomaly ({angle_units})", row=2, col=3)

    # Set y-axis limits for angle plots if requested
    if set_angle_ylim:
        if angle_units == "deg":
            angle_range = [0, 360]
        else:  # radians
            angle_range = [0, 2 * np.pi]

        # Apply to inclination, RAAN, argument of periapsis, and mean anomaly
        fig.update_yaxes(range=angle_range, row=1, col=3)  # Inclination
        fig.update_yaxes(range=angle_range, row=2, col=1)  # RAAN
        fig.update_yaxes(range=angle_range, row=2, col=2)  # Arg. Periapsis
        fig.update_yaxes(range=angle_range, row=2, col=3)  # Mean Anomaly

    # Update layout with optional title and grid settings
    layout_config = {}
    if show_title:
        layout_config["title_text"] = "Keplerian Orbital Elements"

    # Only set width/height if explicitly provided
    if width is not None:
        layout_config["width"] = width
    if height is not None:
        layout_config["height"] = height

    fig.update_layout(**layout_config)

    # Configure grid and axis styling
    axis_config = {
        "showgrid": show_grid,
        "title_font": {"size": 11},
        "tickfont": {"size": 10},
    }
    fig.update_xaxes(**axis_config)
    fig.update_yaxes(**axis_config)

    # Make subplot titles smaller
    for annotation in fig.layout.annotations:
        annotation.font.size = 11

    return fig
