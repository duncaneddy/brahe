"""
Orbit Dynamics Module

This module provides functions for computing celestial body ephemerides and other
orbit dynamics calculations.

Ephemerides:
-----------
- sun_position: Low-precision analytical solar position
- moon_position: Low-precision analytical lunar position
- sun_position_de440s: High-precision solar position using NAIF DE440s
- moon_position_de440s: High-precision lunar position using NAIF DE440s
- mercury_position_de440s: High-precision Mercury position using NAIF DE440s
- venus_position_de440s: High-precision Venus position using NAIF DE440s
- mars_position_de440s: High-precision Mars position using NAIF DE440s
- jupiter_position_de440s: High-precision Jupiter position using NAIF DE440s
- saturn_position_de440s: High-precision Saturn position using NAIF DE440s
- uranus_position_de440s: High-precision Uranus position using NAIF DE440s
- neptune_position_de440s: High-precision Neptune position using NAIF DE440s
- solar_system_barycenter_position_de440s: High-precision SSB position using NAIF DE440s
- ssb_position_de440s: Convenience alias for solar_system_barycenter_position_de440s
- initialize_ephemeris: Pre-initialize the DE440s ephemeris kernel

The DE440s functions use a global, thread-safe ephemeris context that is loaded
once and shared across all calls for performance.
"""

from brahe._brahe import (
    sun_position,
    moon_position,
    sun_position_de440s,
    moon_position_de440s,
    mercury_position_de440s,
    venus_position_de440s,
    mars_position_de440s,
    jupiter_position_de440s,
    saturn_position_de440s,
    uranus_position_de440s,
    neptune_position_de440s,
    solar_system_barycenter_position_de440s,
    ssb_position_de440s,
    initialize_ephemeris,
)

__all__ = [
    "sun_position",
    "moon_position",
    "sun_position_de440s",
    "moon_position_de440s",
    "mercury_position_de440s",
    "venus_position_de440s",
    "mars_position_de440s",
    "jupiter_position_de440s",
    "saturn_position_de440s",
    "uranus_position_de440s",
    "neptune_position_de440s",
    "solar_system_barycenter_position_de440s",
    "ssb_position_de440s",
    "initialize_ephemeris",
]
