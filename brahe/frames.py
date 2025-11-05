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

Naming Conventions:
  Brahe provides two equivalent sets of function names for frame transformations:

  - ECI/ECEF naming: Traditional coordinate system names (e.g., rotation_eci_to_ecef)
  - GCRF/ITRF naming: Explicit reference frame names (e.g., rotation_gcrf_to_itrf)

  Both naming conventions provide identical results. Users can choose whichever
  convention they prefer. The ECI/ECEF names are more intuitive and widely used,
  while the GCRF/ITRF names explicitly identify the specific reference frame
  implementations used. The ECI/ECEF names are provided as the default to get
  the "best" reference frame transformations out-of-the-box, while the
  GCRF/ITRF names are for users who want to be explicit about the
  reference frames they are using.
"""

from brahe._brahe import (
    # Rotation matrix components
    bias_precession_nutation,
    earth_rotation,
    polar_motion,
    rotation_gcrf_to_itrf,
    rotation_itrf_to_gcrf,
    rotation_eci_to_ecef,
    rotation_ecef_to_eci,
    position_gcrf_to_itrf,
    position_itrf_to_gcrf,
    position_eci_to_ecef,
    position_ecef_to_eci,
    state_gcrf_to_itrf,
    state_itrf_to_gcrf,
    state_eci_to_ecef,
    state_ecef_to_eci,
)

__all__ = [
    # Rotation matrix components
    "bias_precession_nutation",
    "earth_rotation",
    "polar_motion",
    "rotation_gcrf_to_itrf",
    "rotation_itrf_to_gcrf",
    "rotation_eci_to_ecef",
    "rotation_ecef_to_eci",
    "position_gcrf_to_itrf",
    "position_itrf_to_gcrf",
    "position_eci_to_ecef",
    "position_ecef_to_eci",
    "state_gcrf_to_itrf",
    "state_itrf_to_gcrf",
    "state_eci_to_ecef",
    "state_ecef_to_eci",
]
