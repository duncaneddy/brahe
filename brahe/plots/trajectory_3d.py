"""
3D trajectory visualization in ECI frame.

Provides 3D plots of orbital trajectories with optional Earth sphere.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from loguru import logger

import brahe as bh
from brahe.plots.backend import validate_backend, apply_scienceplots_style
from brahe._brahe import OrbitTrajectory, OrbitFrame, OrbitRepresentation


def plot_trajectory_3d(
    trajectories,
    time_range=None,
    units="km",
    normalize=False,
    view_azimuth=45.0,
    view_elevation=30.0,
    view_distance=None,
    show_earth=True,
    earth_texture="simple",
    backend="matplotlib",
    width=None,
    height=None,
) -> object:
    """Plot 3D trajectory in ECI frame.

    Args:
        trajectories (list of dict): List of trajectory groups, each with:
            - trajectory: OrbitTrajectory or numpy array [N×3] or [N×6] (positions in ECI)
            - color (str, optional): Line color
            - line_width (float, optional): Line width
            - label (str, optional): Legend label

        time_range (tuple, optional): (start_epoch, end_epoch) to filter data
        units (str, optional): 'm' or 'km'. Default: 'km'
        normalize (bool, optional): Normalize to Earth radii. Default: False
        view_azimuth (float, optional): Camera azimuth angle (degrees). Default: 45.0
        view_elevation (float, optional): Camera elevation angle (degrees). Default: 30.0
        view_distance (float, optional): Camera distance multiplier. Default: 2.5 (larger = further out)
        show_earth (bool, optional): Show Earth sphere at origin. Default: True
        earth_texture (str, optional): 'blue_marble', 'simple', or None. Default: 'simple'
        backend (str, optional): 'matplotlib' or 'plotly'. Default: 'matplotlib'
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
        traj = prop.propagate(epoch, epoch + bh.orbital_period(oe[0]), 60.0)

        # Plot 3D trajectory
        fig = bh.plot_trajectory_3d(
            [{"trajectory": traj, "color": "red", "label": "LEO Orbit"}],
            units='km',
            show_earth=True,
            backend='matplotlib'
        )
        ```
    """
    start_time = time.time()
    logger.info(f"Plotting 3D trajectory with backend={backend}")
    logger.debug(f"Units: {units}, normalize={normalize}, show_earth={show_earth}")

    validate_backend(backend)

    # Normalize inputs
    traj_groups = _normalize_trajectory_groups(trajectories)

    # Dispatch to backend
    if backend == "matplotlib":
        result = _trajectory_3d_matplotlib(
            traj_groups,
            time_range,
            units,
            normalize,
            view_azimuth,
            view_elevation,
            view_distance,
            show_earth,
            earth_texture,
        )
    else:  # plotly
        result = _trajectory_3d_plotly(
            traj_groups,
            time_range,
            units,
            normalize,
            view_azimuth,
            view_elevation,
            view_distance,
            show_earth,
            earth_texture,
            width,
            height,
        )

    elapsed = time.time() - start_time
    logger.info(f"3D trajectory plot completed in {elapsed:.2f}s")
    return result


def _normalize_trajectory_groups(trajectories):
    """Normalize trajectory input to list of dicts with defaults."""
    defaults = {
        "color": None,
        "line_width": 2.0,
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


def _trajectory_3d_matplotlib(
    traj_groups,
    time_range,
    units,
    normalize,
    view_azimuth,
    view_elevation,
    view_distance,
    show_earth,
    earth_texture,
):
    """Matplotlib implementation of 3D trajectory plot."""
    # Apply scienceplots if available
    apply_scienceplots_style()

    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Unit conversion
    scale = 1.0
    if normalize:
        scale = 1.0 / bh.R_EARTH
        unit_label = "Earth Radii"
    elif units == "km":
        scale = 1e-3
        unit_label = "km"
    else:
        unit_label = "m"

    # Plot Earth sphere if requested
    if show_earth:
        _plot_earth_sphere_matplotlib(ax, scale, earth_texture)

    # Plot each trajectory
    for i, group in enumerate(traj_groups):
        trajectory = group.get("trajectory")
        color = group.get("color")
        line_width_group = group.get("line_width", 2.0)
        label = group.get("label")

        if trajectory is None:
            continue

        # Validate that trajectory is an OrbitTrajectory
        if not isinstance(trajectory, OrbitTrajectory):
            raise TypeError(
                f"Trajectory must be an OrbitTrajectory object, got {type(trajectory)}"
            )

        # Convert to ECI Cartesian if needed
        if (
            trajectory.frame != OrbitFrame.ECI
            or trajectory.representation != OrbitRepresentation.CARTESIAN
        ):
            logger.debug(
                f"Converting trajectory from {trajectory.frame}/{trajectory.representation} to ECI/CARTESIAN"
            )
            trajectory = trajectory.to_eci()

        # Extract positions using to_matrix() which returns (N, 6) array
        states = trajectory.to_matrix()
        pos_x = states[:, 0]
        pos_y = states[:, 1]
        pos_z = states[:, 2]

        # Scale positions
        pos_x = pos_x * scale
        pos_y = pos_y * scale
        pos_z = pos_z * scale

        # Plot 3D line
        ax.plot(
            pos_x, pos_y, pos_z, color=color, linewidth=line_width_group, label=label
        )

    # Set viewing angle
    ax.view_init(elev=view_elevation, azim=view_azimuth)

    # Set labels
    ax.set_xlabel(f"X ({unit_label})")
    ax.set_ylabel(f"Y ({unit_label})")
    ax.set_zlabel(f"Z ({unit_label})")
    ax.set_title("3D Trajectory (ECI Frame)")

    # Equal aspect ratio
    _set_axes_equal(ax)

    ax.legend()

    return fig


def _trajectory_3d_plotly(
    traj_groups,
    time_range,
    units,
    normalize,
    view_azimuth,
    view_elevation,
    view_distance,
    show_earth,
    earth_texture,
    width,
    height,
):
    """Plotly implementation of 3D trajectory plot."""
    fig = go.Figure()

    # Unit conversion
    scale = 1.0
    if normalize:
        scale = 1.0 / bh.R_EARTH
        unit_label = "Earth Radii"
    elif units == "km":
        scale = 1e-3
        unit_label = "km"
    else:
        unit_label = "m"

    # Plot Earth sphere if requested
    if show_earth:
        _plot_earth_sphere_plotly(fig, scale, earth_texture)

    # Plot each trajectory
    for i, group in enumerate(traj_groups):
        trajectory = group.get("trajectory")
        color = group.get("color")
        line_width_group = group.get("line_width", 2.0)
        label = group.get("label")

        if trajectory is None:
            continue

        # Validate that trajectory is an OrbitTrajectory
        if not isinstance(trajectory, OrbitTrajectory):
            raise TypeError(
                f"Trajectory must be an OrbitTrajectory object, got {type(trajectory)}"
            )

        # Convert to ECI Cartesian if needed
        if (
            trajectory.frame != OrbitFrame.ECI
            or trajectory.representation != OrbitRepresentation.CARTESIAN
        ):
            logger.debug(
                f"Converting trajectory from {trajectory.frame}/{trajectory.representation} to ECI/CARTESIAN"
            )
            trajectory = trajectory.to_eci()

        # Extract positions using to_matrix() which returns (N, 6) array
        states = trajectory.to_matrix()
        pos_x = states[:, 0]
        pos_y = states[:, 1]
        pos_z = states[:, 2]

        # Scale positions
        pos_x = pos_x * scale
        pos_y = pos_y * scale
        pos_z = pos_z * scale

        # Add 3D scatter trace
        fig.add_trace(
            go.Scatter3d(
                x=pos_x,
                y=pos_y,
                z=pos_z,
                mode="lines",
                line=dict(color=color, width=line_width_group),
                name=label,
            )
        )

    # Set default view distance if not specified (larger = further out)
    if view_distance is None:
        view_distance = 2.5  # Default: zoom out to 2.5x distance

    # Configure layout
    layout_config = {
        "title": "3D Trajectory (ECI Frame)",
        "scene": dict(
            xaxis_title=f"X ({unit_label})",
            yaxis_title=f"Y ({unit_label})",
            zaxis_title=f"Z ({unit_label})",
            aspectmode="data",
            xaxis=dict(showgrid=True, showbackground=False),
            yaxis=dict(showgrid=True, showbackground=False),
            zaxis=dict(showgrid=True, showbackground=False),
            camera=dict(
                eye=dict(
                    x=view_distance
                    * np.cos(np.radians(view_azimuth))
                    * np.cos(np.radians(view_elevation)),
                    y=view_distance
                    * np.sin(np.radians(view_azimuth))
                    * np.cos(np.radians(view_elevation)),
                    z=view_distance * np.sin(np.radians(view_elevation)),
                )
            ),
        ),
    }

    # Only set width/height if explicitly provided
    if width is not None:
        layout_config["width"] = width
    if height is not None:
        layout_config["height"] = height

    fig.update_layout(**layout_config)

    return fig


def _plot_earth_sphere_matplotlib(ax, scale, texture):
    """Plot Earth sphere on matplotlib 3D axis."""
    # Create sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = bh.R_EARTH * scale * np.outer(np.cos(u), np.sin(v))
    y = bh.R_EARTH * scale * np.outer(np.sin(u), np.sin(v))
    z = bh.R_EARTH * scale * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot sphere
    if texture == "simple":
        ax.plot_surface(x, y, z, color="lightblue", alpha=0.6, edgecolor="none")
    else:
        # TODO: Add blue marble texture support
        ax.plot_surface(x, y, z, color="lightblue", alpha=0.6, edgecolor="none")


def _plot_earth_sphere_plotly(fig, scale, texture):
    """Plot Earth sphere on plotly 3D figure."""
    # Create sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = bh.R_EARTH * scale * np.outer(np.cos(u), np.sin(v))
    y = bh.R_EARTH * scale * np.outer(np.sin(u), np.sin(v))
    z = bh.R_EARTH * scale * np.outer(np.ones(np.size(u)), np.cos(v))

    # Add sphere surface with higher opacity for better visibility
    # Note: Plotly Surface doesn't support gradient opacity, so using uniform value
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale="Blues",
            showscale=False,
            opacity=0.7,
            name="Earth",
        )
    )


def _set_axes_equal(ax):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Args:
        ax: matplotlib 3D axis
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
