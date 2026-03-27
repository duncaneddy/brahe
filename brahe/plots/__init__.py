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
"""

from brahe.plots.groundtrack import plot_groundtrack, split_ground_track_at_antimeridian
from brahe.plots.access_geometry import (
    plot_access_polar,
    plot_access_elevation,
    plot_access_elevation_azimuth,
)
from brahe.plots.trajectories import (
    plot_cartesian_trajectory,
    plot_keplerian_trajectory,
)
from brahe.plots.trajectory_3d import plot_trajectory_3d
from brahe.plots.gabbard import plot_gabbard_diagram

from brahe.plots.estimation_state import (
    plot_estimator_state_error,
    plot_estimator_state_value,
    plot_estimator_state_error_grid,
    plot_estimator_state_value_grid,
    plot_estimator_state_error_from_arrays,
    plot_estimator_state_value_from_arrays,
    plot_estimator_state_error_grid_from_arrays,
    plot_estimator_state_value_grid_from_arrays,
)
from brahe.plots.estimation_residuals import (
    plot_measurement_residual,
    plot_measurement_residual_grid,
    plot_measurement_residual_rms,
    plot_measurement_residual_from_arrays,
    plot_measurement_residual_grid_from_arrays,
    plot_measurement_residual_rms_from_arrays,
)
from brahe.plots.estimation_marginal import (
    plot_estimator_marginal,
    plot_estimator_marginal_from_arrays,
)

__all__ = [
    "plot_groundtrack",
    "split_ground_track_at_antimeridian",
    "plot_access_polar",
    "plot_access_elevation",
    "plot_access_elevation_azimuth",
    "plot_cartesian_trajectory",
    "plot_keplerian_trajectory",
    "plot_trajectory_3d",
    "plot_gabbard_diagram",
    # Estimation state plots
    "plot_estimator_state_error",
    "plot_estimator_state_value",
    "plot_estimator_state_error_grid",
    "plot_estimator_state_value_grid",
    "plot_estimator_state_error_from_arrays",
    "plot_estimator_state_value_from_arrays",
    "plot_estimator_state_error_grid_from_arrays",
    "plot_estimator_state_value_grid_from_arrays",
    # Measurement residual plots
    "plot_measurement_residual",
    "plot_measurement_residual_grid",
    "plot_measurement_residual_rms",
    "plot_measurement_residual_from_arrays",
    "plot_measurement_residual_grid_from_arrays",
    "plot_measurement_residual_rms_from_arrays",
    # Marginal distribution plots
    "plot_estimator_marginal",
    "plot_estimator_marginal_from_arrays",
]
