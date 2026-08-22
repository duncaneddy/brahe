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
    AS2RAD,
    AU,
    BDT_TAI,
    BDT_ZERO,
    # Physical constants
    C_LIGHT,
    # Mathematical constants
    DEG2RAD,
    ECC_EARTH,
    GM_DEIMOS,
    GM_EARTH,
    GM_JUPITER,
    GM_JUPITER_SYSTEM,
    GM_MARS,
    GM_MARS_SYSTEM,
    # Planetary GM values
    GM_MERCURY,
    GM_MOON,
    GM_NEPTUNE,
    GM_NEPTUNE_SYSTEM,
    # Martian moon constants
    GM_PHOBOS,
    GM_PLUTO,
    GM_PLUTO_SYSTEM,
    GM_SATURN,
    GM_SATURN_SYSTEM,
    # Solar constants
    GM_SUN,
    GM_URANUS,
    GM_URANUS_SYSTEM,
    GM_VENUS,
    GPS_TAI,
    GPS_TT,
    GPS_ZERO,
    GST_TAI,
    GST_ZERO,
    J2_EARTH,
    J3_EARTH,
    J4_EARTH,
    J5_EARTH,
    J6_EARTH,
    JD_J2000,
    MJD_J2000,
    # Time constants
    MJD_ZERO,
    OMEGA_EARTH,
    OMEGA_MARS,
    # Moon constants
    OMEGA_MOON,
    P_SUN,
    # Earth constants
    R_EARTH,
    R_JUPITER,
    # Mars constants
    R_MARS,
    # Planetary radii
    R_MERCURY,
    # Lunar constants
    R_MOON,
    R_NEPTUNE,
    R_SATURN,
    R_SUN,
    R_URANUS,
    R_VENUS,
    RAD2AS,
    RAD2DEG,
    SECONDS_PER_DAY,
    SECONDS_PER_JULIAN_CENTURY,
    TAI_BDT,
    TAI_GPS,
    TAI_GST,
    TAI_TT,
    TT_GPS,
    TT_TAI,
    UNIX_EPOCH_JD,
    UNIX_EPOCH_MJD,
    WGS84_A,
    WGS84_F,
)

__all__ = [
    "AS2RAD",
    "AU",
    "BDT_TAI",
    "BDT_ZERO",
    # Physical constants
    "C_LIGHT",
    # Mathematical constants
    "DEG2RAD",
    "ECC_EARTH",
    "GM_DEIMOS",
    "GM_EARTH",
    "GM_JUPITER",
    "GM_JUPITER_SYSTEM",
    "GM_MARS",
    "GM_MARS_SYSTEM",
    # Planetary GM values
    "GM_MERCURY",
    "GM_MOON",
    "GM_NEPTUNE",
    "GM_NEPTUNE_SYSTEM",
    # Martian moon constants
    "GM_PHOBOS",
    "GM_PLUTO",
    "GM_PLUTO_SYSTEM",
    "GM_SATURN",
    "GM_SATURN_SYSTEM",
    # Solar constants
    "GM_SUN",
    "GM_URANUS",
    "GM_URANUS_SYSTEM",
    "GM_VENUS",
    "GPS_TAI",
    "GPS_TT",
    "GPS_ZERO",
    "GST_TAI",
    "GST_ZERO",
    "J2_EARTH",
    "J3_EARTH",
    "J4_EARTH",
    "J5_EARTH",
    "J6_EARTH",
    "JD_J2000",
    "MJD_J2000",
    # Time constants
    "MJD_ZERO",
    "OMEGA_EARTH",
    "OMEGA_MARS",
    # Moon constants
    "OMEGA_MOON",
    "P_SUN",
    "RAD2AS",
    "RAD2DEG",
    # Earth constants
    "R_EARTH",
    "R_JUPITER",
    # Mars constants
    "R_MARS",
    # Planetary radii
    "R_MERCURY",
    # Lunar constants
    "R_MOON",
    "R_NEPTUNE",
    "R_SATURN",
    "R_SUN",
    "R_URANUS",
    "R_VENUS",
    "SECONDS_PER_DAY",
    "SECONDS_PER_JULIAN_CENTURY",
    "TAI_BDT",
    "TAI_GPS",
    "TAI_GST",
    "TAI_TT",
    "TT_GPS",
    "TT_TAI",
    "UNIX_EPOCH_JD",
    "UNIX_EPOCH_MJD",
    "WGS84_A",
    "WGS84_F",
]
