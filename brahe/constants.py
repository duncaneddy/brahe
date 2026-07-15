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
    MJD_J2000,
    JD_J2000,
    GPS_TAI,
    TAI_GPS,
    TT_TAI,
    TAI_TT,
    GPS_TT,
    TT_GPS,
    GPS_ZERO,
    BDT_TAI,
    TAI_BDT,
    GST_TAI,
    TAI_GST,
    BDT_ZERO,
    GST_ZERO,
    UNIX_EPOCH_JD,
    UNIX_EPOCH_MJD,
    SECONDS_PER_DAY,
    SECONDS_PER_JULIAN_CENTURY,
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
    J3_EARTH,
    J4_EARTH,
    J5_EARTH,
    J6_EARTH,
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
    GM_MARS_SYSTEM,
    GM_JUPITER,
    GM_JUPITER_SYSTEM,
    GM_SATURN,
    GM_SATURN_SYSTEM,
    GM_URANUS,
    GM_URANUS_SYSTEM,
    GM_NEPTUNE,
    GM_NEPTUNE_SYSTEM,
    GM_PLUTO,
    # Mars constants
    R_MARS,
    OMEGA_MARS,
    # Moon constants
    OMEGA_MOON,
    # Martian moon constants
    GM_PHOBOS,
    GM_DEIMOS,
)

__all__ = [
    # Mathematical constants
    "DEG2RAD",
    "RAD2DEG",
    "AS2RAD",
    "RAD2AS",
    # Time constants
    "MJD_ZERO",
    "MJD_J2000",
    "JD_J2000",
    "GPS_TAI",
    "TAI_GPS",
    "TT_TAI",
    "TAI_TT",
    "GPS_TT",
    "TT_GPS",
    "GPS_ZERO",
    "BDT_TAI",
    "TAI_BDT",
    "GST_TAI",
    "TAI_GST",
    "BDT_ZERO",
    "GST_ZERO",
    "UNIX_EPOCH_JD",
    "SECONDS_PER_DAY",
    "SECONDS_PER_JULIAN_CENTURY",
    "UNIX_EPOCH_MJD",
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
    "J3_EARTH",
    "J4_EARTH",
    "J5_EARTH",
    "J6_EARTH",
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
    "GM_MARS_SYSTEM",
    "GM_JUPITER",
    "GM_JUPITER_SYSTEM",
    "GM_SATURN",
    "GM_SATURN_SYSTEM",
    "GM_URANUS",
    "GM_URANUS_SYSTEM",
    "GM_NEPTUNE",
    "GM_NEPTUNE_SYSTEM",
    "GM_PLUTO",
    # Mars constants
    "R_MARS",
    "OMEGA_MARS",
    # Moon constants
    "OMEGA_MOON",
    # Martian moon constants
    "GM_PHOBOS",
    "GM_DEIMOS",
]
