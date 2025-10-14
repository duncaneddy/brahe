"""
Trajectories Module

Trajectory containers and interpolation for orbit propagation.

This module provides containers for storing and interpolating spacecraft trajectories:

**Trajectory Types:**
- Trajectory: Generic trajectory container for arbitrary state vectors
- OrbitalTrajectory: Specialized trajectory for orbital states with interpolation
- STrajectory6: Static 6-DOF trajectory container

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
"""

from brahe._brahe import (
    # Trajectory classes
    DTrajectory,
    OrbitTrajectory,
    STrajectory6,
    # Configuration enums
    OrbitFrame,
    OrbitRepresentation,
    AngleFormat,
    InterpolationMethod,
)

__all__ = [
    # Trajectory classes
    "DTrajectory",
    "OrbitTrajectory",
    "STrajectory6",
    # Configuration enums
    "OrbitFrame",
    "OrbitRepresentation",
    "AngleFormat",
    "InterpolationMethod",
]
