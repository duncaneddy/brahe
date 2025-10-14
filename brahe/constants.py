"""
Constants Module

Mathematical, physical, and astronomical constants used throughout Brahe.

This module provides:
- Mathematical constants (π conversions, angle conversions)
- Time system constants (offsets, epoch definitions)
- Physical constants (speed of light, astronomical unit)
- Earth constants (radius, GM, shape parameters)
- Planetary constants (GM values for solar system bodies)
"""

from brahe._brahe import (
    # Mathematical constants
    DEG2RAD,
    RAD2DEG,
    AS2RAD,
    RAD2AS,
    # Time constants
    MJD_ZERO,
    MJD2000,
    GPS_TAI,
    TAI_GPS,
    TT_TAI,
    TAI_TT,
    GPS_TT,
    TT_GPS,
    GPS_ZERO,
    # Physical constants
    C_LIGHT,
    AU,
    # Earth constants
    R_EARTH,
    WGS84_A,
    WGS84_F,
    GM_EARTH,
    ECC_EARTH,
    J2_EARTH,
    OMEGA_EARTH,
    # Solar constants
    GM_SUN,
    R_SUN,
    P_SUN,
    # Lunar constants
    R_MOON,
    GM_MOON,
    # Planetary GM values
    GM_MERCURY,
    GM_VENUS,
    GM_MARS,
    GM_JUPITER,
    GM_SATURN,
    GM_URANUS,
    GM_NEPTUNE,
    GM_PLUTO,
)

__all__ = [
    # Mathematical constants
    "DEG2RAD",
    "RAD2DEG",
    "AS2RAD",
    "RAD2AS",
    # Time constants
    "MJD_ZERO",
    "MJD2000",
    "GPS_TAI",
    "TAI_GPS",
    "TT_TAI",
    "TAI_TT",
    "GPS_TT",
    "TT_GPS",
    "GPS_ZERO",
    # Physical constants
    "C_LIGHT",
    "AU",
    # Earth constants
    "R_EARTH",
    "WGS84_A",
    "WGS84_F",
    "GM_EARTH",
    "ECC_EARTH",
    "J2_EARTH",
    "OMEGA_EARTH",
    # Solar constants
    "GM_SUN",
    "R_SUN",
    "P_SUN",
    # Lunar constants
    "R_MOON",
    "GM_MOON",
    # Planetary GM values
    "GM_MERCURY",
    "GM_VENUS",
    "GM_MARS",
    "GM_JUPITER",
    "GM_SATURN",
    "GM_URANUS",
    "GM_NEPTUNE",
    "GM_PLUTO",
]
