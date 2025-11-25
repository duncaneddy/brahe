"""
Trajectories Module

Trajectory containers and interpolation for orbit propagation.

This module provides containers for storing and interpolating spacecraft trajectories:

**Trajectory Types:**
- Trajectory: Dynamic trajectory container for arbitrary state vectors
- OrbitTrajectory: Orbital trajectory with frame/representation conversions

**Trajectory Features:**
- Time-series state storage
- Interpolation methods (linear, cubic spline)
- Support for multiple reference frames (ECI, ECEF)
- Support for multiple orbit representations (Cartesian, Keplerian)
- Angle format handling (radians, degrees)

**Enumerations:**
- OrbitFrame: Reference frame specification
- OrbitRepresentation: State representation format
- AngleFormat: Angle unit specification
- InterpolationMethod: Interpolation algorithm selection
- CovarianceInterpolationMethod: Covariance interpolation algorithm selection
"""

from brahe._brahe import (
    # Trajectory classes
    Trajectory,
    OrbitTrajectory,
    # Configuration enums
    OrbitFrame,
    OrbitRepresentation,
    AngleFormat,
    InterpolationMethod,
    CovarianceInterpolationMethod,
)

__all__ = [
    # Trajectory classes
    "Trajectory",
    "OrbitTrajectory",
    # Configuration enums
    "OrbitFrame",
    "OrbitRepresentation",
    "AngleFormat",
    "InterpolationMethod",
    "CovarianceInterpolationMethod",
]
