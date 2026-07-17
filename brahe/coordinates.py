"""
Coordinates Module

Coordinate system transformations for satellite dynamics.

This module provides transformations between various coordinate representations:

**Cartesian Coordinates:**
- State vector (position + velocity) representations
- Conversions to/from osculating Keplerian elements
- state_eci_to_koe_for_body / state_koe_to_eci_for_body: Osculating elements about a central body with arbitrary GM

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

**Right Ascension / Declination Coordinates:**
- Conversions to/from Cartesian inertial position and state
- Topocentric right ascension/declination to/from azimuth-elevation
- Proper-motion propagation between epochs
"""

from brahe._brahe import (
    # Coordinate types
    EllipsoidalConversionType,
    # Cartesian conversions
    state_koe_to_eci,
    state_eci_to_koe,
    state_eci_to_koe_for_body,
    state_koe_to_eci_for_body,
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
    # Right Ascension / Declination
    position_radec_to_inertial,
    position_inertial_to_radec,
    state_radec_to_inertial,
    state_inertial_to_radec,
    position_radec_to_azel,
    position_azel_to_radec,
    apply_proper_motion,
)

__all__ = [
    # Coordinate types
    "EllipsoidalConversionType",
    # Cartesian conversions
    "state_koe_to_eci",
    "state_eci_to_koe",
    "state_eci_to_koe_for_body",
    "state_koe_to_eci_for_body",
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
    # Right Ascension / Declination
    "position_radec_to_inertial",
    "position_inertial_to_radec",
    "state_radec_to_inertial",
    "state_inertial_to_radec",
    "position_radec_to_azel",
    "position_azel_to_radec",
    "apply_proper_motion",
]
