"""
Gabbard diagram visualization.

Provides scatter plots of orbital period vs apogee/perigee altitude, commonly
used for visualizing satellite populations and debris clouds from breakup events.
"""

import time
from typing import Union

import matplotlib.figure
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from loguru import logger

import brahe as bh
from brahe.plots.backend import validate_backend, apply_scienceplots_style


def plot_gabbard_diagram(
    objects,
    epoch=None,
    altitude_units: str = "km",
    period_units: str = "min",
    backend: str = "matplotlib",
    width=None,
    height=None,
) -> Union[matplotlib.figure.Figure, go.Figure]:
    """Plot Gabbard diagram showing orbital period vs apogee/perigee altitude.

    A Gabbard diagram is a scatter plot used to visualize satellite populations
    and debris clouds, plotting each object's apogee and perigee altitudes
    against its orbital period. This visualization is particularly useful for
    analyzing satellite breakup events and orbital debris distributions.

    Args:
        objects (list): List of objects to plot. Can be:
            - List of Propagator objects (SGPPropagator or KeplerianPropagator)
            - List of numpy arrays (state vectors or Keplerian elements) with format specified
            - List of dicts with keys:
                - objects: List of propagators or states
                - format (str, optional): 'ECI', 'ECEF', or 'Keplerian' (required for state arrays)
                - color (str, optional): Marker color
                - marker (str, optional): Marker style
                - label (str, optional): Legend label

        epoch (Epoch, optional): Epoch to evaluate propagator states. If None, uses current state.
        altitude_units (str, optional): 'km' or 'm'. Default: 'km'
        period_units (str, optional): 'min' or 's'. Default: 'min'
        backend (str, optional): 'matplotlib' or 'plotly'. Default: 'matplotlib'
        width (int, optional): Figure width in pixels (plotly only). Default: None (responsive)
        height (int, optional): Figure height in pixels (plotly only). Default: None (responsive)

    Returns:
        Union[matplotlib.figure.Figure, plotly.graph_objects.Figure]: Figure object

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Set up EOP
        eop = bh.FileEOPProvider.from_default_standard(bh.EarthOrientationFileType.STANDARD, True)
        bh.set_global_eop_provider(eop)

        # Create propagators for debris cloud
        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # Parent orbit
        oe_parent = np.array([bh.R_EARTH + 215e3, 0.1, np.radians(97.8), 0.0, 0.0, 0.0])

        # Simulate debris with various delta-v
        debris = []
        for dv in np.linspace(-100, 100, 50):  # m/s delta-v
            oe = oe_parent.copy()
            # Simplified: adjust semi-major axis based on delta-v
            oe[0] += dv * 1000  # rough approximation
            oe[1] = max(0.001, min(0.3, oe[1] + np.random.normal(0, 0.05)))
            state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
            prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)
            debris.append(prop)

        # Plot Gabbard diagram
        fig = bh.plot_gabbard_diagram(
            debris,
            epoch=epoch,
            altitude_units='km',
            period_units='min',
            backend='matplotlib'
        )
        ```
    """
    start_time = time.time()
    logger.info(f"Plotting Gabbard diagram with backend={backend}")
    logger.debug(f"Units: altitude={altitude_units}, period={period_units}")

    validate_backend(backend)

    # Normalize inputs
    object_groups = _normalize_object_groups(objects)

    # Dispatch to backend
    if backend == "matplotlib":
        result = _gabbard_matplotlib(object_groups, epoch, altitude_units, period_units)
    else:  # plotly
        result = _gabbard_plotly(
            object_groups, epoch, altitude_units, period_units, width, height
        )

    elapsed = time.time() - start_time
    logger.info(f"Gabbard diagram plot completed in {elapsed:.2f}s")
    return result


def _normalize_object_groups(objects):
    """Normalize object input to list of dicts with defaults."""
    defaults = {
        "format": None,
        "color": None,
        "marker": None,
        "label": None,
    }

    if objects is None:
        return []

    if not isinstance(objects, list):
        return [{**defaults, "objects": [objects]}]

    if len(objects) == 0:
        return []

    # Check if first element is a dict (grouped input)
    if isinstance(objects[0], dict):
        # List of dicts - apply defaults
        return [{**defaults, **group} for group in objects]
    else:
        # List of objects without grouping
        return [{**defaults, "objects": objects}]


def _extract_orbital_elements(obj, epoch, obj_format):
    """Extract Keplerian elements from various object types.

    Returns:
        tuple: (semi_major_axis, eccentricity) in meters and dimensionless
    """

    # Check if it's a propagator
    if hasattr(obj, "state_as_osculating_elements"):
        # It's a propagator
        if epoch is not None:
            oe = obj.state_as_osculating_elements(epoch, bh.AngleFormat.RADIANS)
        else:
            # Convert current state to osculating elements
            state = obj.current_state()
            oe = bh.state_cartesian_to_osculating(state, bh.AngleFormat.RADIANS)

        return oe[0], oe[1]

    # It's a state vector or Keplerian elements
    if isinstance(obj, np.ndarray):
        if len(obj) != 6:
            raise ValueError(f"State vector must have 6 elements, got {len(obj)}")

        if obj_format is None:
            raise ValueError(
                "format must be specified for state vector inputs (ECI, ECEF, or Keplerian)"
            )

        if obj_format.upper() == "KEPLERIAN":
            # Already Keplerian [a, e, i, raan, argp, anom]
            return obj[0], obj[1]
        elif obj_format.upper() == "ECI":
            # Cartesian ECI state
            oe = bh.state_cartesian_to_osculating(obj, bh.AngleFormat.RADIANS)
            return oe[0], oe[1]
        elif obj_format.upper() == "ECEF":
            # Need to convert ECEF to ECI first
            if epoch is None:
                raise ValueError("epoch must be provided to convert ECEF states to ECI")
            state_eci = bh.state_ecef_to_eci(epoch, obj)
            oe = bh.state_cartesian_to_osculating(state_eci, bh.AngleFormat.RADIANS)
            return oe[0], oe[1]
        else:
            raise ValueError(
                f"Unknown format: {obj_format}. Must be 'ECI', 'ECEF', or 'Keplerian'"
            )

    raise TypeError(f"Unknown object type: {type(obj)}")


def _gabbard_matplotlib(object_groups, epoch, altitude_units, period_units):
    """Matplotlib implementation of Gabbard diagram."""
    # Apply scienceplots if available
    apply_scienceplots_style()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Gabbard Diagram")
    ax.set_xlabel(f"Orbital Period ({period_units})")
    ax.set_ylabel(f"Apogee / Perigee Altitude ({altitude_units})")
    ax.grid(True, alpha=0.3)

    # Unit conversion factors
    alt_scale = 1e-3 if altitude_units == "km" else 1.0
    period_scale = 1.0 / 60.0 if period_units == "min" else 1.0

    # Track if any data was plotted
    has_data = False

    # Plot each group
    for i, group in enumerate(object_groups):
        objects = group.get("objects", [])
        obj_format = group.get("format")
        color = group.get("color")
        label = group.get("label")

        if not objects:
            continue

        # Extract data
        periods = []
        apogees = []
        perigees = []

        for obj in objects:
            try:
                a, e = _extract_orbital_elements(obj, epoch, obj_format)

                # Calculate apogee and perigee altitudes
                apogee_alt = bh.apogee_altitude(a, e) * alt_scale
                perigee_alt = bh.perigee_altitude(a, e) * alt_scale

                # Calculate orbital period
                period = bh.orbital_period(a) * period_scale

                periods.append(period)
                periods.append(period)  # Same period for both points
                apogees.append(apogee_alt)
                perigees.append(perigee_alt)
            except Exception as e:
                logger.warning(f"Failed to extract elements from object: {e}")
                continue

        if not periods:
            continue

        has_data = True

        # Plot apogee points (red circles)
        apogee_label = f"{label} (Apogee)" if label else "Apogee"
        ax.scatter(
            periods[::2],  # Every other period (paired with apogee)
            apogees,
            c=color if color else "red",
            marker="o",
            label=apogee_label if i == 0 or label else None,
            alpha=0.6,
            s=50,
        )

        # Plot perigee points (blue diamonds)
        perigee_label = f"{label} (Perigee)" if label else "Perigee"
        ax.scatter(
            periods[1::2],  # Every other period (paired with perigee)
            perigees,
            c=color if color else "blue",
            marker="D",
            label=perigee_label if i == 0 or label else None,
            alpha=0.6,
            s=50,
        )

    # Only show legend if there's data to display
    if has_data:
        ax.legend()

    plt.tight_layout()
    return fig


def _gabbard_plotly(object_groups, epoch, altitude_units, period_units, width, height):
    """Plotly implementation of Gabbard diagram."""
    # Create figure
    fig = go.Figure()

    # Unit conversion factors
    alt_scale = 1e-3 if altitude_units == "km" else 1.0
    period_scale = 1.0 / 60.0 if period_units == "min" else 1.0

    # Plot each group
    for i, group in enumerate(object_groups):
        objects = group.get("objects", [])
        obj_format = group.get("format")
        color = group.get("color")
        label = group.get("label")

        if not objects:
            continue

        # Extract data
        periods_apogee = []
        altitudes_apogee = []
        periods_perigee = []
        altitudes_perigee = []
        hover_text_apogee = []
        hover_text_perigee = []

        for obj in objects:
            try:
                a, e = _extract_orbital_elements(obj, epoch, obj_format)

                # Calculate apogee and perigee altitudes
                apogee_alt = bh.apogee_altitude(a, e) * alt_scale
                perigee_alt = bh.perigee_altitude(a, e) * alt_scale

                # Calculate orbital period
                period = bh.orbital_period(a) * period_scale

                # Check if object has get_id() method and use it for hover text
                obj_id = None
                if hasattr(obj, "get_id"):
                    obj_id = obj.get_id()

                if obj_id is not None:
                    hover_text_apogee.append(f"ID: {obj_id}")
                    hover_text_perigee.append(f"ID: {obj_id}")
                else:
                    hover_text_apogee.append("Apogee")
                    hover_text_perigee.append("Perigee")

                periods_apogee.append(period)
                altitudes_apogee.append(apogee_alt)
                periods_perigee.append(period)
                altitudes_perigee.append(perigee_alt)
            except Exception as e:
                logger.warning(f"Failed to extract elements from object: {e}")
                continue

        if not periods_apogee:
            continue

        # Plot apogee points (red circles)
        apogee_label = f"{label} (Apogee)" if label else "Apogee"
        fig.add_trace(
            go.Scatter(
                x=periods_apogee,
                y=altitudes_apogee,
                mode="markers",
                name=apogee_label,
                marker=dict(
                    color=color if color else "red",
                    size=8,
                    symbol="circle",
                    opacity=0.6,
                ),
                customdata=hover_text_apogee,
                hovertemplate=(
                    "<b>%{customdata}</b><br>"
                    f"Period: %{{x:.2f}} {period_units}<br>"
                    f"Altitude: %{{y:.1f}} {altitude_units}<br>"
                    "<extra></extra>"
                ),
                showlegend=(i == 0 or label is not None),
            )
        )

        # Plot perigee points (blue diamonds)
        perigee_label = f"{label} (Perigee)" if label else "Perigee"
        fig.add_trace(
            go.Scatter(
                x=periods_perigee,
                y=altitudes_perigee,
                mode="markers",
                name=perigee_label,
                marker=dict(
                    color=color if color else "blue",
                    size=8,
                    symbol="diamond",
                    opacity=0.6,
                ),
                customdata=hover_text_perigee,
                hovertemplate=(
                    "<b>%{customdata}</b><br>"
                    f"Period: %{{x:.2f}} {period_units}<br>"
                    f"Altitude: %{{y:.1f}} {altitude_units}<br>"
                    "<extra></extra>"
                ),
                showlegend=(i == 0 or label is not None),
            )
        )

    # Update layout
    layout_config = {
        "title": "Gabbard Diagram",
        "xaxis_title": f"Orbital Period ({period_units})",
        "yaxis_title": f"Apogee / Perigee Altitude ({altitude_units})",
        "hovermode": "closest",
        "showlegend": True,
    }

    # Only set width/height if explicitly provided
    if width is not None:
        layout_config["width"] = width
    if height is not None:
        layout_config["height"] = height

    fig.update_layout(**layout_config)

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    return fig
