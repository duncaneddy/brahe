"""
Plots Module

Visualization tools for astrodynamics data with support for matplotlib and plotly backends.

This module provides high-level plotting functions for common astrodynamics visualizations:
- Ground track plots with communication cones and polygon zones
- Access window geometry (polar plots and elevation profiles)
- Orbital element time series (Keplerian and Cartesian)
- 3D trajectory visualization around arbitrary central bodies, including synodic (rotating) frames
- Gabbard diagrams (orbital period vs apogee/perigee altitude)
- Estimation state errors, values, residuals, and marginal distributions

All plot functions support both matplotlib and plotly backends, selected via the `backend` parameter.

Plotting requires the optional ``plots`` dependencies (matplotlib, cartopy, plotly,
shapely, pillow, httpx, kaleido, pyshp, scienceplots). Importing this module does not
import those libraries; they are loaded lazily when a plot function is first accessed.
Install them with ``pip install "brahe[plots]"`` (or ``"brahe[all]"``).
"""

import importlib
from typing import TYPE_CHECKING

# Map each public plotting symbol to the submodule that defines it. Submodules are
# imported lazily on first attribute access so that importing ``brahe`` / ``brahe.plots``
# never pulls in the heavy (and optional) visualization dependency stack.
_PLOT_EXPORTS = {
    "plot_groundtrack": "groundtrack",
    "split_ground_track_at_antimeridian": "groundtrack",
    "plot_access_polar": "access_geometry",
    "plot_access_elevation": "access_geometry",
    "plot_access_elevation_azimuth": "access_geometry",
    "plot_cartesian_trajectory": "trajectories",
    "plot_keplerian_trajectory": "trajectories",
    "plot_trajectory_3d": "trajectory_3d",
    "plot_synodic_3d": "synodic_3d",
    "plot_earth_moon_rotating_3d": "synodic_3d",
    "plot_gabbard_diagram": "gabbard",
    "plot_estimator_state_error": "estimation_state",
    "plot_estimator_state_value": "estimation_state",
    "plot_estimator_state_error_grid": "estimation_state",
    "plot_estimator_state_value_grid": "estimation_state",
    "plot_estimator_state_error_from_arrays": "estimation_state",
    "plot_estimator_state_value_from_arrays": "estimation_state",
    "plot_estimator_state_error_grid_from_arrays": "estimation_state",
    "plot_estimator_state_value_grid_from_arrays": "estimation_state",
    "plot_measurement_residual": "estimation_residuals",
    "plot_measurement_residual_grid": "estimation_residuals",
    "plot_measurement_residual_rms": "estimation_residuals",
    "plot_measurement_residual_from_arrays": "estimation_residuals",
    "plot_measurement_residual_grid_from_arrays": "estimation_residuals",
    "plot_measurement_residual_rms_from_arrays": "estimation_residuals",
    "plot_estimator_marginal": "estimation_marginal",
    "plot_estimator_marginal_from_arrays": "estimation_marginal",
}

__all__ = list(_PLOT_EXPORTS.keys())

# Static re-exports for the benefit of type checkers, IDEs, and the docs builder
# (griffe/mkdocstrings). This block never executes at runtime (TYPE_CHECKING is
# False), so the optional visualization stack stays unimported — but it lets
# static tooling resolve the symbols that ``__getattr__`` forwards lazily below.
# Keep in sync with ``_PLOT_EXPORTS``.
if TYPE_CHECKING:
    from brahe.plots.access_geometry import (
        plot_access_elevation as plot_access_elevation,
        plot_access_elevation_azimuth as plot_access_elevation_azimuth,
        plot_access_polar as plot_access_polar,
    )
    from brahe.plots.estimation_marginal import (
        plot_estimator_marginal as plot_estimator_marginal,
        plot_estimator_marginal_from_arrays as plot_estimator_marginal_from_arrays,
    )
    from brahe.plots.estimation_residuals import (
        plot_measurement_residual as plot_measurement_residual,
        plot_measurement_residual_from_arrays as plot_measurement_residual_from_arrays,
        plot_measurement_residual_grid as plot_measurement_residual_grid,
        plot_measurement_residual_grid_from_arrays as plot_measurement_residual_grid_from_arrays,
        plot_measurement_residual_rms as plot_measurement_residual_rms,
        plot_measurement_residual_rms_from_arrays as plot_measurement_residual_rms_from_arrays,
    )
    from brahe.plots.estimation_state import (
        plot_estimator_state_error as plot_estimator_state_error,
        plot_estimator_state_error_from_arrays as plot_estimator_state_error_from_arrays,
        plot_estimator_state_error_grid as plot_estimator_state_error_grid,
        plot_estimator_state_error_grid_from_arrays as plot_estimator_state_error_grid_from_arrays,
        plot_estimator_state_value as plot_estimator_state_value,
        plot_estimator_state_value_from_arrays as plot_estimator_state_value_from_arrays,
        plot_estimator_state_value_grid as plot_estimator_state_value_grid,
        plot_estimator_state_value_grid_from_arrays as plot_estimator_state_value_grid_from_arrays,
    )
    from brahe.plots.gabbard import plot_gabbard_diagram as plot_gabbard_diagram
    from brahe.plots.groundtrack import (
        plot_groundtrack as plot_groundtrack,
        split_ground_track_at_antimeridian as split_ground_track_at_antimeridian,
    )
    from brahe.plots.trajectories import (
        plot_cartesian_trajectory as plot_cartesian_trajectory,
        plot_keplerian_trajectory as plot_keplerian_trajectory,
    )
    from brahe.plots.trajectory_3d import (
        plot_trajectory_3d as plot_trajectory_3d,
    )
    from brahe.plots.synodic_3d import (
        plot_earth_moon_rotating_3d as plot_earth_moon_rotating_3d,
        plot_synodic_3d as plot_synodic_3d,
    )

_INSTALL_HINT = (
    "Plotting support requires the optional 'plots' dependencies. "
    'Install with: pip install "brahe[plots]" or "brahe[all]"'
)


def _import_plot_submodule(submodule):
    """Import a ``brahe.plots`` submodule, reframing a missing optional plotting
    dependency into an actionable install hint while preserving the original error."""
    try:
        return importlib.import_module(f"brahe.plots.{submodule}")
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


def __getattr__(name):
    submodule = _PLOT_EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(_import_plot_submodule(submodule), name)


def __dir__():
    return sorted(__all__)
