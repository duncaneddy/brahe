"""
Coordinates Module

Coordinate system transformations for satellite dynamics.

This module provides transformations between various coordinate representations:

**Cartesian Coordinates:**
- State vector (position + velocity) representations
- Conversions to/from osculating Keplerian elements
- state_inertial_to_koe_for_body / state_koe_to_inertial_for_body: Osculating elements referenced to a central body's mean equator at J2000

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
    apply_proper_motion,
    position_azel_to_radec,
    position_ecef_to_geocentric,
    position_ecef_to_geodetic,
    # Azimuth-Elevation
    position_enz_to_azel,
    # Geocentric conversions
    position_geocentric_to_ecef,
    # Geodetic conversions
    position_geodetic_to_ecef,
    position_inertial_to_radec,
    position_radec_to_azel,
    # Right Ascension / Declination
    position_radec_to_inertial,
    position_sez_to_azel,
    relative_position_ecef_to_enz,
    relative_position_ecef_to_sez,
    relative_position_enz_to_ecef,
    relative_position_sez_to_ecef,
    # Topocentric ENZ
    rotation_ellipsoid_to_enz,
    # Topocentric SEZ
    rotation_ellipsoid_to_sez,
    rotation_enz_to_ellipsoid,
    rotation_sez_to_ellipsoid,
    state_eci_to_koe,
    state_inertial_to_koe_for_body,
    state_inertial_to_radec,
    # Cartesian conversions
    state_koe_to_eci,
    state_koe_to_inertial_for_body,
    state_radec_to_inertial,
)

__all__ = [
    # Coordinate types
    "EllipsoidalConversionType",
    "apply_proper_motion",
    "position_azel_to_radec",
    "position_ecef_to_geocentric",
    "position_ecef_to_geodetic",
    # Azimuth-Elevation
    "position_enz_to_azel",
    # Geocentric conversions
    "position_geocentric_to_ecef",
    # Geodetic conversions
    "position_geodetic_to_ecef",
    "position_inertial_to_radec",
    "position_radec_to_azel",
    # Right Ascension / Declination
    "position_radec_to_inertial",
    "position_sez_to_azel",
    "relative_position_ecef_to_enz",
    "relative_position_ecef_to_sez",
    "relative_position_enz_to_ecef",
    "relative_position_sez_to_ecef",
    # Topocentric ENZ
    "rotation_ellipsoid_to_enz",
    # Topocentric SEZ
    "rotation_ellipsoid_to_sez",
    "rotation_enz_to_ellipsoid",
    "rotation_sez_to_ellipsoid",
    "state_eci_to_koe",
    "state_inertial_to_koe_for_body",
    "state_inertial_to_radec",
    # Cartesian conversions
    "state_koe_to_eci",
    "state_koe_to_inertial_for_body",
    "state_radec_to_inertial",
]
