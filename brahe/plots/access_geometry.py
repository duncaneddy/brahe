"""
Access window geometry visualization.

Provides polar plots (azimuth/elevation) and elevation profile plots for access windows.
"""

import time
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from loguru import logger

from brahe.plots.backend import validate_backend, apply_scienceplots_style
from brahe._brahe import KeplerianPropagator, SGPPropagator


def plot_access_polar(
    access_windows,
    propagator: Union[KeplerianPropagator, SGPPropagator],
    min_elevation=0.0,
    num_samples=None,
    time_step=5.0,
    elevation_mask=None,
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
        propagator: Propagator object for computing interpolated trajectories
        min_elevation (float, optional): Minimum elevation for plot edge (degrees). Default: 0.0
        num_samples (int, optional): Number of samples for interpolation. If None, uses time_step.
        time_step (float, optional): Time step for interpolation (seconds). Default: 5.0.
                                     Ignored if num_samples is specified.
        elevation_mask (float, callable, or array, optional): Elevation mask to visualize.
            Can be:
            - float: Constant elevation angle (degrees)
            - callable: Function taking azimuth (degrees) returning elevation (degrees)
            - array: Elevation values at each azimuth (evaluated at 360 points around horizon)
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
        result = _access_polar_matplotlib(
            window_groups,
            propagator,
            min_elevation,
            num_samples,
            time_step,
            elevation_mask,
        )
    else:  # plotly
        result = _access_polar_plotly(
            window_groups,
            propagator,
            min_elevation,
            num_samples,
            time_step,
            elevation_mask,
        )

    elapsed = time.time() - start_time
    logger.info(f"Access polar plot completed in {elapsed:.2f}s")
    return result


def plot_access_elevation(
    access_windows,
    propagator: Union[KeplerianPropagator, SGPPropagator],
    num_samples=None,
    time_step=5.0,
    backend="matplotlib",
) -> object:
    """Plot elevation angle vs time for access windows.

    Args:
        access_windows (list of dict): List of access window groups, each with:
            - access_window: AccessWindow object
            - propagator (Propagator, optional): Propagator for full trajectory
            - color (str, optional): Line color
            - line_width (float, optional): Line width
        propagator: Propagator object for computing interpolated trajectories
        num_samples (int, optional): Number of samples for interpolation. If None, uses time_step.
        time_step (float, optional): Time step for interpolation (seconds). Default: 5.0.
                                     Ignored if num_samples is specified.
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
        result = _access_elevation_matplotlib(
            window_groups, propagator, num_samples, time_step
        )
    else:  # plotly
        result = _access_elevation_plotly(
            window_groups, propagator, num_samples, time_step
        )

    elapsed = time.time() - start_time
    logger.info(f"Access elevation plot completed in {elapsed:.2f}s")
    return result


def plot_access_elevation_azimuth(
    access_windows,
    propagator: Union[KeplerianPropagator, SGPPropagator],
    num_samples=None,
    time_step=5.0,
    elevation_mask=None,
    backend="matplotlib",
) -> object:
    """Plot elevation vs azimuth for access windows (observed horizon plot).

    Shows the satellite's path across the sky as elevation angle vs azimuth angle,
    providing a "view from the ground" perspective of the satellite's trajectory.

    Args:
        access_windows (list of dict): List of access window groups, each with:
            - access_window: AccessWindow object
            - propagator (Propagator, optional): Propagator for full trajectory
            - color (str, optional): Line color
            - line_width (float, optional): Line width
            - label (str, optional): Legend label
        propagator: Propagator object for computing interpolated trajectories
        num_samples (int, optional): Number of samples for interpolation. If None, uses time_step.
        time_step (float, optional): Time step for interpolation (seconds). Default: 5.0.
                                     Ignored if num_samples is specified.
        elevation_mask (float, callable, or array, optional): Elevation mask to visualize.
            Can be:
            - float: Constant elevation angle (degrees)
            - callable: Function taking azimuth (degrees) returning elevation (degrees)
            - array: Elevation values at each azimuth (evaluated at 360 points)
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

        windows = bh.location_accesses([location], [prop], epoch, epoch + 86400.0, constraint)

        # Sinusoidal elevation mask
        mask_fn = lambda az: 15.0 + 10.0 * np.sin(np.radians(2 * az))

        # Plot elevation vs azimuth
        fig = bh.plot_access_elevation_azimuth(
            [{"access_window": w, "propagator": prop, "label": f"Pass {i+1}"} for i, w in enumerate(windows[:3])],
            elevation_mask=mask_fn,
            backend='matplotlib'
        )
        ```
    """
    start_time = time.time()
    logger.info(f"Plotting access elevation vs azimuth with backend={backend}")
    logger.debug(f"Windows: {len(access_windows)}")

    validate_backend(backend)

    # Normalize inputs
    window_groups = _normalize_access_window_groups(access_windows)

    # Dispatch to backend
    if backend == "matplotlib":
        result = _access_elevation_azimuth_matplotlib(
            window_groups, propagator, num_samples, time_step, elevation_mask
        )
    else:  # plotly
        result = _access_elevation_azimuth_plotly(
            window_groups, propagator, num_samples, time_step, elevation_mask
        )

    elapsed = time.time() - start_time
    logger.info(f"Access elevation vs azimuth plot completed in {elapsed:.2f}s")
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


def _access_polar_matplotlib(
    window_groups,
    propagator: Union[KeplerianPropagator, SGPPropagator],
    min_elevation,
    num_samples,
    time_step,
    elevation_mask,
):
    """Matplotlib implementation of access polar plot."""
    # Apply scienceplots if available
    apply_scienceplots_style()

    # Create polar plot
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))

    # Configure polar plot
    # North at top (theta=0), angles increase clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Radius is 90 - elevation, so 0 at center (zenith), 90 at edge (horizon)
    # Always show full range from zenith to horizon
    ax.set_ylim(0, 90)
    ax.set_yticks(np.arange(0, 91, 15))
    ax.set_yticklabels([f"{int(90 - r)}°" for r in np.arange(0, 91, 15)])

    # Plot elevation mask if provided
    if elevation_mask is not None:
        # Create azimuth samples around full horizon
        mask_azimuths = np.linspace(0, 360, 361)  # Include 360 to close the circle
        mask_elevations = _evaluate_elevation_mask(elevation_mask, mask_azimuths)

        # Convert to polar coordinates
        mask_theta = np.radians(mask_azimuths)
        mask_radius = 90.0 - mask_elevations

        # Fill region from horizon to mask
        ax.fill_between(
            mask_theta, 90, mask_radius, alpha=0.2, color="gray", label="Elevation Mask"
        )

    # Plot each window
    for i, group in enumerate(window_groups):
        access_window = group.get("access_window")
        color = group.get("color")
        line_width = group.get("line_width", 2.0)
        label = group.get("label")

        if access_window is None:
            continue

        # Extract azimuth/elevation trajectory
        azimuths, elevations = _extract_azimuth_elevation(
            access_window, propagator, num_samples, time_step
        )

        if len(azimuths) == 0:
            continue

        # Split at discontinuities to avoid drawing lines across plot
        segments = _split_azimuth_discontinuities(azimuths, elevations)

        # Plot each continuous segment
        for seg_idx, (az_seg, el_seg) in enumerate(segments):
            # Convert to polar coordinates
            # Azimuth: degrees to radians
            theta = np.radians(az_seg)
            # Radius: 90 - elevation (so zenith is at center)
            radius = 90.0 - el_seg

            # Only show label on first segment
            seg_label = label if seg_idx == 0 else None
            ax.plot(theta, radius, color=color, linewidth=line_width, label=seg_label)

        # Mark start and end points (using original full arrays)
        theta_start = np.radians(azimuths[0])
        theta_end = np.radians(azimuths[-1])
        radius_start = 90.0 - elevations[0]
        radius_end = 90.0 - elevations[-1]
        ax.plot(theta_start, radius_start, "o", color=color, markersize=6)
        ax.plot(theta_end, radius_end, "s", color=color, markersize=6)

    ax.set_xlabel("Azimuth")
    ax.set_title("Access Window Geometry")
    ax.grid(True)
    ax.legend()

    return fig


def _access_polar_plotly(
    window_groups,
    propagator: Union[KeplerianPropagator, SGPPropagator],
    min_elevation,
    num_samples,
    time_step,
    elevation_mask,
):
    """Plotly implementation of access polar plot."""
    fig = go.Figure()

    # Plot elevation mask if provided
    if elevation_mask is not None:
        # Create azimuth samples around full horizon
        mask_azimuths = np.linspace(0, 360, 361)  # Include 360 to close the circle
        mask_elevations = _evaluate_elevation_mask(elevation_mask, mask_azimuths)

        # Convert to polar coordinates (radius = 90 - elevation)
        mask_radius = 90.0 - mask_elevations

        # Create filled region from horizon (90) to mask
        # Need to trace: horizon edge -> mask edge -> back to start
        fill_theta = np.concatenate(
            [mask_azimuths, mask_azimuths[::-1], [mask_azimuths[0]]]
        )
        fill_radius = np.concatenate(
            [np.full_like(mask_azimuths, 90), mask_radius[::-1], [90]]
        )

        fig.add_trace(
            go.Scatterpolar(
                r=fill_radius,
                theta=fill_theta,
                fill="toself",
                fillcolor="rgba(128, 128, 128, 0.2)",
                line=dict(width=0),
                name="Elevation Mask",
                showlegend=True,
            )
        )

    # Plot each window
    for i, group in enumerate(window_groups):
        access_window = group.get("access_window")
        color = group.get("color")
        line_width = group.get("line_width", 2.0)
        label = group.get("label")

        # Use default label if none specified
        if label is None:
            label = f"Access {i + 1}"

        if access_window is None:
            continue

        # Extract azimuth/elevation trajectory
        azimuths, elevations = _extract_azimuth_elevation(
            access_window, propagator, num_samples, time_step
        )

        if len(azimuths) == 0:
            continue

        # Split at discontinuities to avoid drawing lines across plot
        segments = _split_azimuth_discontinuities(azimuths, elevations)

        # Plot each continuous segment
        for seg_idx, (az_seg, el_seg) in enumerate(segments):
            # Radius: 90 - elevation (so zenith is at center)
            radius_seg = 90.0 - el_seg

            # Only show label on first segment
            seg_label = label if seg_idx == 0 else None
            seg_showlegend = seg_idx == 0

            fig.add_trace(
                go.Scatterpolar(
                    r=radius_seg,
                    theta=az_seg,
                    mode="lines",
                    name=seg_label,
                    line=dict(color=color, width=line_width),
                    showlegend=seg_showlegend,
                )
            )

        # Mark start and end points (using original full arrays)
        radius_start = 90.0 - elevations[0]
        radius_end = 90.0 - elevations[-1]
        fig.add_trace(
            go.Scatterpolar(
                r=[radius_start],
                theta=[azimuths[0]],
                mode="markers",
                name=f"{label} (start)" if label else None,
                marker=dict(color=color, size=8, symbol="circle"),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=[radius_end],
                theta=[azimuths[-1]],
                mode="markers",
                name=f"{label} (end)" if label else None,
                marker=dict(color=color, size=8, symbol="square"),
                showlegend=False,
            )
        )

    # Create tick values and labels for radial axis
    # Radius represents 90 - elevation, so we need to reverse the labels
    tick_radii = np.arange(0, 91, 15)  # [0, 15, 30, 45, 60, 75, 90]
    tick_labels = [f"{int(90 - r)}°" for r in tick_radii]  # ["90°", "75°", ..., "0°"]

    fig.update_layout(
        title="Access Window Geometry",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                range=[0, 90],  # Always show full range from zenith to horizon
                angle=90,
                tickangle=90,
                tickmode="array",
                tickvals=tick_radii.tolist(),
                ticktext=tick_labels,
            ),
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def _access_elevation_matplotlib(
    window_groups,
    propagator: Union[KeplerianPropagator, SGPPropagator],
    num_samples,
    time_step,
):
    """Matplotlib implementation of elevation vs time plot."""
    # Apply scienceplots if available
    apply_scienceplots_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each window
    for i, group in enumerate(window_groups):
        access_window = group.get("access_window")
        color = group.get("color")
        line_width = group.get("line_width", 2.0)
        label = group.get("label")

        if access_window is None:
            continue

        # Extract time and elevation
        times, elevations = _extract_time_elevation(
            access_window, propagator, num_samples, time_step
        )

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
    ax.set_ylim(0, 90)
    ax.set_yticks(np.arange(0, 91, 15))
    ax.grid(True)
    ax.legend()

    return fig


def _access_elevation_plotly(
    window_groups,
    propagator: Union[KeplerianPropagator, SGPPropagator],
    num_samples,
    time_step,
):
    """Plotly implementation of elevation vs time plot."""
    fig = go.Figure()

    # Plot each window elevation profile
    for i, group in enumerate(window_groups):
        access_window = group.get("access_window")
        color = group.get("color")
        line_width = group.get("line_width", 2.0)
        label = group.get("label")

        # Use default label if none specified
        if label is None:
            label = f"Access {i + 1}"

        if access_window is None:
            continue

        # Extract time and elevation
        times, elevations = _extract_time_elevation(
            access_window, propagator, num_samples, time_step
        )

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
        yaxis=dict(range=[0, 90], tickmode="linear", tick0=0, dtick=15),
    )

    return fig


def _access_elevation_azimuth_matplotlib(
    window_groups,
    propagator: Union[KeplerianPropagator, SGPPropagator],
    num_samples,
    time_step,
    elevation_mask,
):
    """Matplotlib implementation of elevation vs azimuth plot."""
    # Apply scienceplots if available
    apply_scienceplots_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot elevation mask if provided
    if elevation_mask is not None:
        # Create azimuth samples across full range
        mask_azimuths = np.linspace(0, 360, 361)
        mask_elevations = _evaluate_elevation_mask(elevation_mask, mask_azimuths)

        # Fill region from 0 to mask elevation
        ax.fill_between(
            mask_azimuths,
            0,
            mask_elevations,
            alpha=0.2,
            color="gray",
            label="Elevation Mask",
        )

    # Plot each window
    for i, group in enumerate(window_groups):
        access_window = group.get("access_window")
        color = group.get("color")
        line_width = group.get("line_width", 2.0)
        label = group.get("label")

        if access_window is None:
            continue

        # Extract azimuth/elevation trajectory
        azimuths, elevations = _extract_azimuth_elevation(
            access_window, propagator, num_samples, time_step
        )

        if len(azimuths) == 0:
            continue

        # Split at discontinuities to avoid drawing lines across plot
        segments = _split_azimuth_discontinuities(azimuths, elevations)

        # Plot each continuous segment
        for seg_idx, (az_seg, el_seg) in enumerate(segments):
            # Only show label on first segment
            seg_label = label if seg_idx == 0 else None
            ax.plot(az_seg, el_seg, color=color, linewidth=line_width, label=seg_label)

        # Mark start and end points (using original full arrays)
        ax.plot(azimuths[0], elevations[0], "o", color=color, markersize=6)
        ax.plot(azimuths[-1], elevations[-1], "s", color=color, markersize=6)

    ax.set_xlabel("Azimuth (degrees)")
    ax.set_ylabel("Elevation (degrees)")
    ax.set_title("Elevation vs Azimuth")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 90)
    ax.set_yticks(np.arange(0, 91, 15))
    ax.grid(True)
    ax.legend()

    return fig


def _access_elevation_azimuth_plotly(
    window_groups,
    propagator: Union[KeplerianPropagator, SGPPropagator],
    num_samples,
    time_step,
    elevation_mask,
):
    """Plotly implementation of elevation vs azimuth plot."""
    fig = go.Figure()

    # Plot elevation mask if provided
    if elevation_mask is not None:
        # Create azimuth samples across full range
        mask_azimuths = np.linspace(0, 360, 361)
        mask_elevations = _evaluate_elevation_mask(elevation_mask, mask_azimuths)

        # Add filled region from 0 to mask elevation
        fig.add_trace(
            go.Scatter(
                x=mask_azimuths,
                y=mask_elevations,
                fill="tozeroy",
                fillcolor="rgba(128, 128, 128, 0.2)",
                line=dict(width=0),
                name="Elevation Mask",
                showlegend=True,
            )
        )

    # Plot each window
    for i, group in enumerate(window_groups):
        access_window = group.get("access_window")
        color = group.get("color")
        line_width = group.get("line_width", 2.0)
        label = group.get("label")

        # Use default label if none specified
        if label is None:
            label = f"Access {i + 1}"

        if access_window is None:
            continue

        # Extract azimuth/elevation trajectory
        azimuths, elevations = _extract_azimuth_elevation(
            access_window, propagator, num_samples, time_step
        )

        if len(azimuths) == 0:
            continue

        # Split at discontinuities to avoid drawing lines across plot
        segments = _split_azimuth_discontinuities(azimuths, elevations)

        # Plot each continuous segment
        for seg_idx, (az_seg, el_seg) in enumerate(segments):
            # Only show label on first segment
            seg_label = label if seg_idx == 0 else None
            seg_showlegend = seg_idx == 0

            fig.add_trace(
                go.Scatter(
                    x=az_seg,
                    y=el_seg,
                    mode="lines",
                    name=seg_label,
                    line=dict(color=color, width=line_width),
                    showlegend=seg_showlegend,
                )
            )

        # Mark start and end points (using original full arrays)
        fig.add_trace(
            go.Scatter(
                x=[azimuths[0]],
                y=[elevations[0]],
                mode="markers",
                name=f"{label} (start)" if label else None,
                marker=dict(color=color, size=8, symbol="circle"),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[azimuths[-1]],
                y=[elevations[-1]],
                mode="markers",
                name=f"{label} (end)" if label else None,
                marker=dict(color=color, size=8, symbol="square"),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Elevation vs Azimuth",
        xaxis_title="Azimuth (degrees)",
        yaxis_title="Elevation (degrees)",
        xaxis=dict(range=[0, 360]),
        yaxis=dict(range=[0, 90], tickmode="linear", tick0=0, dtick=15),
    )

    return fig


def _evaluate_elevation_mask(elevation_mask, azimuths):
    """Evaluate elevation mask at given azimuth values.

    Args:
        elevation_mask: Can be:
            - float: Constant elevation (degrees)
            - callable: Function taking azimuth (degrees) and returning elevation
            - array-like: Elevation values at azimuth samples (must match length)
        azimuths: Array of azimuth values in degrees [0, 360)

    Returns:
        Array of elevation mask values in degrees, same length as azimuths
    """
    if elevation_mask is None:
        return None

    # Constant elevation
    if isinstance(elevation_mask, (int, float)):
        return np.full_like(azimuths, float(elevation_mask))

    # Callable function
    if callable(elevation_mask):
        return np.array([elevation_mask(az) for az in azimuths])

    # Array-like
    mask_array = np.asarray(elevation_mask)
    if len(mask_array) != len(azimuths):
        raise ValueError(
            f"Elevation mask array length ({len(mask_array)}) must match "
            f"azimuth array length ({len(azimuths)})"
        )
    return mask_array


def _split_azimuth_discontinuities(azimuths, elevations):
    """Split azimuth/elevation data at 0°/360° discontinuities.

    When satellite trajectories cross North (azimuth transitions between 0° and 360°),
    plotting libraries will draw a straight line across the plot. This function detects
    such discontinuities and splits the data into separate segments.

    Args:
        azimuths: Array of azimuth values in degrees [0, 360)
        elevations: Array of elevation values in degrees

    Returns:
        List of (azimuth_segment, elevation_segment) tuples, one per continuous segment
    """
    if len(azimuths) < 2:
        return [(azimuths, elevations)]

    segments = []
    start_idx = 0

    # Find discontinuities (jumps > 180 degrees)
    for i in range(1, len(azimuths)):
        azimuth_diff = abs(azimuths[i] - azimuths[i - 1])

        # Check for discontinuity
        if azimuth_diff > 180.0:
            # Add segment up to (but not including) this point
            segments.append((azimuths[start_idx:i], elevations[start_idx:i]))
            start_idx = i

    # Add final segment
    segments.append((azimuths[start_idx:], elevations[start_idx:]))

    return segments


def _extract_azimuth_elevation(
    access_window, propagator, num_samples=None, time_step=5.0
):
    """Extract azimuth and elevation trajectory from access window.

    Args:
        access_window: AccessWindow object with window_open, window_close, and location data
        propagator: Propagator to compute satellite states
        num_samples: Number of points to sample (overrides time_step if provided)
        time_step: Time step in seconds for sampling (default: 5.0)

    Returns:
        (azimuths, elevations): Arrays in degrees
    """
    import brahe as bh

    # Calculate number of samples
    duration = access_window.duration
    if num_samples is None:
        num_samples = max(2, int(duration / time_step) + 1)
    else:
        num_samples = max(2, num_samples)

    # Generate time samples
    times = np.linspace(
        access_window.window_open.mjd(), access_window.window_close.mjd(), num_samples
    )

    # Extract location from access window properties
    location_ecef = np.array(access_window.properties.center_ecef)

    azimuths = []
    elevations = []

    # Compute azimuth/elevation at each time
    for mjd in times:
        epoch = bh.Epoch.from_mjd(mjd, bh.TimeSystem.UTC)

        # Get satellite position in ECEF
        state_ecef = propagator.state_ecef(epoch)
        sat_pos_ecef = state_ecef[:3]

        # Compute relative position in ENZ frame
        rel_pos_enz = bh.relative_position_ecef_to_enz(
            location_ecef, sat_pos_ecef, bh.EllipsoidalConversionType.GEODETIC
        )

        # Compute azimuth and elevation
        east, north, zenith = rel_pos_enz[0], rel_pos_enz[1], rel_pos_enz[2]
        azimuth = np.degrees(np.arctan2(east, north)) % 360.0
        elevation = np.degrees(np.arctan2(zenith, np.sqrt(east**2 + north**2)))

        azimuths.append(azimuth)
        elevations.append(elevation)

    return np.array(azimuths), np.array(elevations)


def _extract_time_elevation(access_window, propagator, num_samples=None, time_step=5.0):
    """Extract time and elevation trajectory from access window.

    Args:
        access_window: AccessWindow object with window_open, window_close, and location data
        propagator: Propagator to compute satellite states
        num_samples: Number of points to sample (overrides time_step if provided)
        time_step: Time step in seconds for sampling (default: 5.0)

    Returns:
        (times, elevations): Time as Python datetime objects, elevations in degrees
    """
    import brahe as bh

    # Calculate number of samples
    duration = access_window.duration
    if num_samples is None:
        num_samples = max(2, int(duration / time_step) + 1)
    else:
        num_samples = max(2, num_samples)

    # Generate time samples
    mjd_times = np.linspace(
        access_window.window_open.mjd(), access_window.window_close.mjd(), num_samples
    )

    # Extract location from access window properties
    location_ecef = np.array(access_window.properties.center_ecef)

    times = []
    elevations = []

    # Compute elevation at each time
    for mjd in mjd_times:
        epoch = bh.Epoch.from_mjd(mjd, bh.TimeSystem.UTC)
        times.append(epoch.to_pydatetime())

        # Get satellite position in ECEF
        state_ecef = propagator.state_ecef(epoch)
        sat_pos_ecef = state_ecef[:3]

        # Compute relative position in ENZ frame
        rel_pos_enz = bh.relative_position_ecef_to_enz(
            location_ecef, sat_pos_ecef, bh.EllipsoidalConversionType.GEODETIC
        )

        # Compute elevation
        east, north, zenith = rel_pos_enz[0], rel_pos_enz[1], rel_pos_enz[2]
        elevation = np.degrees(np.arctan2(zenith, np.sqrt(east**2 + north**2)))

        elevations.append(elevation)

    return np.array(times), np.array(elevations)
