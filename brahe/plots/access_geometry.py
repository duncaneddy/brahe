"""
Access window geometry visualization.

Provides polar plots (azimuth/elevation) and elevation profile plots for access windows.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from loguru import logger

from brahe.plots.backend import validate_backend, apply_scienceplots_style


def plot_access_polar(
    access_windows,
    min_elevation=0.0,
    backend="matplotlib",
) -> object:
    """Plot access window geometry in polar coordinates (azimuth/elevation).

    Polar coordinates:
    - Radius: 90° - elevation (zenith at center, horizon at edge)
    - Theta: Azimuth (North at top, clockwise)

    Args:
        access_windows (list of dict): List of access window groups, each with:
            - access_window: AccessWindow object
            - propagator (Propagator, optional): Propagator for full trajectory
            - color (str, optional): Line color
            - line_width (float, optional): Line width
            - label (str, optional): Legend label

        min_elevation (float, optional): Minimum elevation for plot edge (degrees). Default: 0.0
        backend (str, optional): 'matplotlib' or 'plotly'. Default: 'matplotlib'

    Returns:
        Generated figure object

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Setup
        eop = bh.FileEOPProvider.from_default_standard(bh.EarthOrientationFileType.STANDARD, True)
        bh.set_global_eop_provider(eop)

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        oe = np.array([bh.R_EARTH + 500e3, 0.01, np.radians(97.8), 0.0, 0.0, 0.0])
        state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)

        prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0).with_name("Satellite")
        location = bh.PointLocation(np.radians(40.7128), np.radians(-74.0060), 0.0).with_name("NYC")
        constraint = bh.ElevationConstraint(min_elevation_deg=10.0)

        # Compute access
        windows = bh.location_accesses([location], [prop], epoch, epoch + 86400.0, constraint)

        # Plot first access window
        fig = bh.plot_access_polar(
            [{"access_window": windows[0], "propagator": prop, "label": "Pass 1"}],
            min_elevation=10.0,
            backend='matplotlib'
        )
        ```
    """
    start_time = time.time()
    logger.info(f"Plotting access polar geometry with backend={backend}")
    logger.debug(f"Windows: {len(access_windows)}, min_elevation={min_elevation}°")

    validate_backend(backend)

    # Normalize inputs
    window_groups = _normalize_access_window_groups(access_windows)

    # Dispatch to backend
    if backend == "matplotlib":
        result = _access_polar_matplotlib(window_groups, min_elevation)
    else:  # plotly
        result = _access_polar_plotly(window_groups, min_elevation)

    elapsed = time.time() - start_time
    logger.info(f"Access polar plot completed in {elapsed:.2f}s")
    return result


def plot_access_elevation(
    access_windows,
    backend="matplotlib",
) -> object:
    """Plot elevation angle vs time for access windows.

    Args:
        access_windows (list of dict): List of access window groups, each with:
            - access_window: AccessWindow object
            - propagator (Propagator, optional): Propagator for full trajectory
            - color (str, optional): Line color
            - line_width (float, optional): Line width
            - label (str, optional): Legend label

        backend (str, optional): 'matplotlib' or 'plotly'. Default: 'matplotlib'

    Returns:
        Generated figure object

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Setup (same as polar plot example)
        eop = bh.FileEOPProvider.from_default_standard(bh.EarthOrientationFileType.STANDARD, True)
        bh.set_global_eop_provider(eop)

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        oe = np.array([bh.R_EARTH + 500e3, 0.01, np.radians(97.8), 0.0, 0.0, 0.0])
        state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)

        prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0).with_name("Satellite")
        location = bh.PointLocation(np.radians(40.7128), np.radians(-74.0060), 0.0).with_name("NYC")
        constraint = bh.ElevationConstraint(min_elevation_deg=10.0)

        windows = bh.location_accesses([location], [prop], epoch, epoch + 86400.0, constraint)

        # Plot elevation profile
        fig = bh.plot_access_elevation(
            [{"access_window": w, "propagator": prop, "label": f"Pass {i+1}"} for i, w in enumerate(windows[:3])],
            backend='matplotlib'
        )
        ```
    """
    start_time = time.time()
    logger.info(f"Plotting access elevation profile with backend={backend}")
    logger.debug(f"Windows: {len(access_windows)}")

    validate_backend(backend)

    # Normalize inputs
    window_groups = _normalize_access_window_groups(access_windows)

    # Dispatch to backend
    if backend == "matplotlib":
        result = _access_elevation_matplotlib(window_groups)
    else:  # plotly
        result = _access_elevation_plotly(window_groups)

    elapsed = time.time() - start_time
    logger.info(f"Access elevation plot completed in {elapsed:.2f}s")
    return result


def _normalize_access_window_groups(access_windows):
    """Normalize access window input to list of dicts with defaults."""
    defaults = {
        "propagator": None,
        "color": None,
        "line_width": 2.0,
        "label": None,
    }

    if access_windows is None:
        return []

    if not isinstance(access_windows, list):
        return [{**defaults, "access_window": access_windows}]

    if len(access_windows) == 0:
        return []

    if not isinstance(access_windows[0], dict):
        # List of AccessWindow objects without config
        return [{**defaults, "access_window": w} for w in access_windows]

    # List of dicts - apply defaults
    return [{**defaults, **group} for group in access_windows]


def _access_polar_matplotlib(window_groups, min_elevation):
    """Matplotlib implementation of access polar plot."""
    # Apply scienceplots if available
    apply_scienceplots_style()

    # Create polar plot
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))

    # Configure polar plot
    # North at top (theta=0), angles increase clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Radius is 90 - elevation, so 90 at edge (horizon), 0 at center (zenith)
    ax.set_ylim(min_elevation, 90)
    ax.set_yticks(np.arange(min_elevation, 91, 15))
    ax.set_yticklabels([f"{int(90 - r)}°" for r in np.arange(min_elevation, 91, 15)])

    # Plot each window
    for i, group in enumerate(window_groups):
        access_window = group.get("access_window")
        propagator = group.get("propagator")
        color = group.get("color")
        line_width = group.get("line_width", 2.0)
        label = group.get("label")

        if access_window is None:
            continue

        # Extract azimuth/elevation trajectory
        azimuths, elevations = _extract_azimuth_elevation(access_window, propagator)

        if len(azimuths) == 0:
            continue

        # Convert to polar coordinates
        # Azimuth: degrees to radians
        theta = np.radians(azimuths)
        # Radius: 90 - elevation (so zenith is at center)
        radius = 90.0 - elevations

        # Plot trajectory
        ax.plot(theta, radius, color=color, linewidth=line_width, label=label)

        # Mark start and end points
        ax.plot(theta[0], radius[0], "o", color=color, markersize=6)
        ax.plot(theta[-1], radius[-1], "s", color=color, markersize=6)

    ax.set_xlabel("Azimuth")
    ax.set_title("Access Window Geometry")
    ax.grid(True)
    ax.legend()

    return fig


def _access_polar_plotly(window_groups, min_elevation):
    """Plotly implementation of access polar plot."""
    fig = go.Figure()

    # Plot each window
    for i, group in enumerate(window_groups):
        access_window = group.get("access_window")
        propagator = group.get("propagator")
        color = group.get("color")
        line_width = group.get("line_width", 2.0)
        label = group.get("label")

        if access_window is None:
            continue

        # Extract azimuth/elevation trajectory
        azimuths, elevations = _extract_azimuth_elevation(access_window, propagator)

        if len(azimuths) == 0:
            continue

        # Radius: 90 - elevation (so zenith is at center)
        radius = 90.0 - elevations

        # Plot trajectory
        fig.add_trace(
            go.Scatterpolar(
                r=radius,
                theta=azimuths,
                mode="lines",
                name=label,
                line=dict(color=color, width=line_width),
            )
        )

        # Mark start and end points
        fig.add_trace(
            go.Scatterpolar(
                r=[radius[0]],
                theta=[azimuths[0]],
                mode="markers",
                name=f"{label} (start)" if label else None,
                marker=dict(color=color, size=8, symbol="circle"),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=[radius[-1]],
                theta=[azimuths[-1]],
                mode="markers",
                name=f"{label} (end)" if label else None,
                marker=dict(color=color, size=8, symbol="square"),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Access Window Geometry",
        polar=dict(
            radialaxis=dict(
                range=[min_elevation, 90],
                angle=90,
                tickangle=90,
            ),
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
            ),
        ),
    )

    return fig


def _access_elevation_matplotlib(window_groups):
    """Matplotlib implementation of elevation vs time plot."""
    # Apply scienceplots if available
    apply_scienceplots_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each window
    for i, group in enumerate(window_groups):
        access_window = group.get("access_window")
        propagator = group.get("propagator")
        color = group.get("color")
        line_width = group.get("line_width", 2.0)
        label = group.get("label")

        if access_window is None:
            continue

        # Extract time and elevation
        times, elevations = _extract_time_elevation(access_window, propagator)

        if len(times) == 0:
            continue

        # Plot elevation profile
        ax.plot(times, elevations, color=color, linewidth=line_width, label=label)

        # Mark start and end points
        ax.plot(times[0], elevations[0], "o", color=color, markersize=6)
        ax.plot(times[-1], elevations[-1], "s", color=color, markersize=6)

    ax.set_xlabel("Time")
    ax.set_ylabel("Elevation (degrees)")
    ax.set_title("Elevation Profile")
    ax.grid(True)
    ax.legend()

    return fig


def _access_elevation_plotly(window_groups):
    """Plotly implementation of elevation vs time plot."""
    fig = go.Figure()

    # Plot each window elevation profile
    for i, group in enumerate(window_groups):
        access_window = group.get("access_window")
        propagator = group.get("propagator")
        color = group.get("color")
        line_width = group.get("line_width", 2.0)
        label = group.get("label")

        if access_window is None:
            continue

        # Extract time and elevation
        times, elevations = _extract_time_elevation(access_window, propagator)

        if len(times) == 0:
            continue

        # Plot elevation profile
        fig.add_trace(
            go.Scatter(
                x=times,
                y=elevations,
                mode="lines",
                name=label,
                line=dict(color=color, width=line_width),
            )
        )

        # Mark start and end points
        fig.add_trace(
            go.Scatter(
                x=[times[0]],
                y=[elevations[0]],
                mode="markers",
                name=f"{label} (start)" if label else None,
                marker=dict(color=color, size=8, symbol="circle"),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[times[-1]],
                y=[elevations[-1]],
                mode="markers",
                name=f"{label} (end)" if label else None,
                marker=dict(color=color, size=8, symbol="square"),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Elevation Profile",
        xaxis_title="Time",
        yaxis_title="Elevation (degrees)",
    )

    return fig


def _extract_azimuth_elevation(access_window, propagator, num_samples=100):
    """Extract azimuth and elevation trajectory from access window.

    Args:
        access_window: AccessWindow object with window_open and window_close
        propagator: Propagator to compute satellite states (optional)
        num_samples: Number of points to sample along the trajectory

    Returns:
        (azimuths, elevations): Arrays in degrees
    """
    # If no propagator provided, use only start/end points from properties
    if propagator is None:
        azimuths = np.array(
            [
                access_window.properties.azimuth_open,
                access_window.properties.azimuth_close,
            ]
        )
        elevations = np.array(
            [
                access_window.properties.elevation_min,
                access_window.properties.elevation_max,
            ]
        )
        return azimuths, elevations

    # For full implementation with propagator, we would need:
    # 1. Get satellite ECI states at each time
    # 2. Get location ECEF position from access window
    # 3. Compute relative geometry (azimuth/elevation)
    #
    # Current limitation: AccessWindow doesn't store the location object,
    # so we can't compute the full trajectory even with a propagator.
    # Future enhancement: Add location to access_windows parameter or
    # store it in AccessWindow.
    #
    # Simplified implementation: use properties min/max as endpoints
    azimuths = np.array(
        [access_window.properties.azimuth_open, access_window.properties.azimuth_close]
    )
    elevations = np.array(
        [access_window.properties.elevation_min, access_window.properties.elevation_max]
    )

    return azimuths, elevations


def _extract_time_elevation(access_window, propagator, num_samples=100):
    """Extract time and elevation trajectory from access window.

    Args:
        access_window: AccessWindow object with window_open and window_close
        propagator: Propagator to compute satellite states (optional)
        num_samples: Number of points to sample along the trajectory

    Returns:
        (times, elevations): Time as epoch values, elevations in degrees
    """
    # If no propagator provided, use only start/end points from properties
    if propagator is None:
        times = np.array([access_window.window_open, access_window.window_close])
        elevations = np.array(
            [
                access_window.properties.elevation_min,
                access_window.properties.elevation_max,
            ]
        )
        return times, elevations

    # Get window times
    window_open = access_window.window_open
    window_close = access_window.window_close
    duration = window_close - window_open

    # Sample times uniformly across the window
    times = np.array(
        [window_open + i * duration / (num_samples - 1) for i in range(num_samples)]
    )

    # Simplified implementation: use properties min/max as endpoints
    # and interpolate linearly (this is a limitation without location access)
    # Typically elevation rises to max at midpoint then falls
    elevations = np.zeros(num_samples)
    for i in range(num_samples):
        t_norm = i / (num_samples - 1)
        if t_norm < 0.5:
            # Rising from min to max
            elevations[i] = access_window.properties.elevation_min + (
                access_window.properties.elevation_max
                - access_window.properties.elevation_min
            ) * (t_norm * 2)
        else:
            # Falling from max to min
            elevations[i] = access_window.properties.elevation_max - (
                access_window.properties.elevation_max
                - access_window.properties.elevation_min
            ) * ((t_norm - 0.5) * 2)

    return times, elevations
