"""
Trajectories Module

Trajectory containers and interpolation for orbit propagation.

This module provides containers for storing and interpolating spacecraft trajectories:

**Trajectory Types:**
- Trajectory: Dynamic trajectory container for arbitrary state vectors
- OrbitTrajectory: Orbital trajectory with frame/representation conversions
- AttitudeTrajectory: Chronologically sorted attitude (quaternion + optional rate) samples

**Trajectory Features:**
- Time-series state storage
- Interpolation methods (linear, cubic spline)
- Support for multiple reference frames (ECI, ECEF)
- Support for multiple orbit representations (Cartesian, Keplerian)
- Angle format handling (radians, degrees)
- Attitude interpolation (slerp, linear, Lagrange) and provider access (quaternion, Euler
  angles, Euler axis, rotation matrix, angular velocity)

**Enumerations:**
- OrbitFrame: Reference frame specification
- OrbitRepresentation: State representation format
- AngleFormat: Angle unit specification
- InterpolationMethod: Interpolation algorithm selection
- CovarianceInterpolationMethod: Covariance interpolation algorithm selection
"""

from brahe._brahe import (
    AngleFormat,
    AttitudeState,
    AttitudeTrajectory,
    CovarianceInterpolationMethod,
    InterpolationMethod,
    # Configuration enums
    OrbitFrame,
    OrbitRepresentation,
    OrbitTrajectory,
    # Trajectory classes
    Trajectory,
)

__all__ = [
    "AngleFormat",
    "AttitudeState",
    "AttitudeTrajectory",
    "CovarianceInterpolationMethod",
    "InterpolationMethod",
    # Configuration enums
    "OrbitFrame",
    "OrbitRepresentation",
    "OrbitTrajectory",
    # Trajectory classes
    "Trajectory",
]
