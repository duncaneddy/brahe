"""
Plots Module

Visualization tools for astrodynamics data with support for matplotlib and plotly backends.

This module provides high-level plotting functions for common astrodynamics visualizations:
- Ground track plots with communication cones and polygon zones
- Access window geometry (polar plots and elevation profiles)
- Orbital element time series (Keplerian and Cartesian)
- 3D trajectory visualization in ECI frame
- Gabbard diagrams (orbital period vs apogee/perigee altitude)
- Estimation state errors, values, residuals, and marginal distributions

All plot functions support both matplotlib and plotly backends, selected via the `backend` parameter.

Plotting requires the optional ``plots`` dependencies (matplotlib, cartopy, plotly,
shapely, pillow, httpx, kaleido, pyshp, scienceplots). Importing this module does not
import those libraries; they are loaded lazily when a plot function is first accessed.
Install them with ``pip install "brahe[plots]"`` (or ``"brahe[all]"``).
"""

import importlib

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
