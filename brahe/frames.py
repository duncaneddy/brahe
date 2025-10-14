"""
Reference Frames Module

Reference frame transformations between ECI and ECEF coordinate systems.

This module provides transformations between:
- ECI (Earth-Centered Inertial): J2000/GCRF frame
- ECEF (Earth-Centered Earth-Fixed): ITRF frame

The transformations implement the IAU 2006/2000A precession-nutation model
and use Earth Orientation Parameters (EOP) for high-precision conversions.

Functions are provided for:
- Rotation matrices (bias-precession-nutation, Earth rotation, polar motion)
- Position vector transformations
- State vector (position + velocity) transformations
"""

from brahe._brahe import (
    # Rotation matrix components
    bias_precession_nutation,
    earth_rotation,
    polar_motion,
    # Complete rotation matrices
    rotation_eci_to_ecef,
    rotation_ecef_to_eci,
    # Position transformations
    position_eci_to_ecef,
    position_ecef_to_eci,
    # State transformations
    state_eci_to_ecef,
    state_ecef_to_eci,
)

__all__ = [
    # Rotation matrix components
    "bias_precession_nutation",
    "earth_rotation",
    "polar_motion",
    # Complete rotation matrices
    "rotation_eci_to_ecef",
    "rotation_ecef_to_eci",
    # Position transformations
    "position_eci_to_ecef",
    "position_ecef_to_eci",
    # State transformations
    "state_eci_to_ecef",
    "state_ecef_to_eci",
]
