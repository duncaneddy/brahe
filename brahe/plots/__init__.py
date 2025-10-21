"""
Plots Module

Visualization tools for astrodynamics data with support for matplotlib and plotly backends.

This module provides high-level plotting functions for common astrodynamics visualizations:
- Ground track plots with communication cones and polygon zones
- Access window geometry (polar plots and elevation profiles)
- Orbital element time series (Keplerian and Cartesian)
- 3D trajectory visualization in ECI frame

All plot functions support both matplotlib and plotly backends, selected via the `backend` parameter.
"""

from brahe.plots.groundtrack import plot_groundtrack
from brahe.plots.access_geometry import plot_access_polar, plot_access_elevation
from brahe.plots.orbital_elements import (
    plot_cartesian_elements,
    plot_keplerian_elements,
)
from brahe.plots.trajectory_3d import plot_trajectory_3d

__all__ = [
    "plot_groundtrack",
    "plot_access_polar",
    "plot_access_elevation",
    "plot_cartesian_elements",
    "plot_keplerian_elements",
    "plot_trajectory_3d",
]
