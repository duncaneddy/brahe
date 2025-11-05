"""
Plots Module

Visualization tools for astrodynamics data with support for matplotlib and plotly backends.

This module provides high-level plotting functions for common astrodynamics visualizations:
- Ground track plots with communication cones and polygon zones
- Access window geometry (polar plots and elevation profiles)
- Orbital element time series (Keplerian and Cartesian)
- 3D trajectory visualization in ECI frame
- Gabbard diagrams (orbital period vs apogee/perigee altitude)

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
]
