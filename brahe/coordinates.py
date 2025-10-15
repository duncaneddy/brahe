"""
Coordinates Module

Coordinate system transformations for satellite dynamics.

This module provides transformations between various coordinate representations:

**Cartesian Coordinates:**
- State vector (position + velocity) representations
- Conversions to/from osculating Keplerian elements

**Geocentric Coordinates:**
- Spherical geocentric coordinates (latitude, longitude, altitude)
- Conversions to/from ECEF Cartesian coordinates

**Geodetic Coordinates:**
- WGS84 geodetic coordinates (latitude, longitude, altitude)
- Conversions to/from ECEF Cartesian coordinates

**Topocentric Coordinates:**
- East-North-Zenith (ENZ) local coordinate system
- South-East-Zenith (SEZ) local coordinate system
- Azimuth-Elevation transformations
- Station-relative position and velocity
"""

from brahe._brahe import (
    # Coordinate types
    EllipsoidalConversionType,
    # Cartesian conversions
    state_osculating_to_cartesian,
    state_cartesian_to_osculating,
    # Geocentric conversions
    position_geocentric_to_ecef,
    position_ecef_to_geocentric,
    # Geodetic conversions
    position_geodetic_to_ecef,
    position_ecef_to_geodetic,
    # Topocentric ENZ
    rotation_ellipsoid_to_enz,
    rotation_enz_to_ellipsoid,
    relative_position_ecef_to_enz,
    relative_position_enz_to_ecef,
    # Topocentric SEZ
    rotation_ellipsoid_to_sez,
    rotation_sez_to_ellipsoid,
    relative_position_ecef_to_sez,
    relative_position_sez_to_ecef,
    # Azimuth-Elevation
    position_enz_to_azel,
    position_sez_to_azel,
)

__all__ = [
    # Coordinate types
    "EllipsoidalConversionType",
    # Cartesian conversions
    "state_osculating_to_cartesian",
    "state_cartesian_to_osculating",
    # Geocentric conversions
    "position_geocentric_to_ecef",
    "position_ecef_to_geocentric",
    # Geodetic conversions
    "position_geodetic_to_ecef",
    "position_ecef_to_geodetic",
    # Topocentric ENZ
    "rotation_ellipsoid_to_enz",
    "rotation_enz_to_ellipsoid",
    "relative_position_ecef_to_enz",
    "relative_position_enz_to_ecef",
    # Topocentric SEZ
    "rotation_ellipsoid_to_sez",
    "rotation_sez_to_ellipsoid",
    "relative_position_ecef_to_sez",
    "relative_position_sez_to_ecef",
    # Azimuth-Elevation
    "position_enz_to_azel",
    "position_sez_to_azel",
]
