"""
Orbit Dynamics Module

This module provides functions for computing celestial body ephemerides and
acceleration models for orbital dynamics.

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

Acceleration Models:
-------------------
Third-Body Perturbations:
- accel_third_body_sun: Sun perturbation using analytical ephemerides
- accel_third_body_moon: Moon perturbation using analytical ephemerides
- accel_third_body_sun_de440s: Sun perturbation using DE440s ephemerides
- accel_third_body_moon_de440s: Moon perturbation using DE440s ephemerides
- accel_third_body_mercury_de440s: Mercury perturbation using DE440s ephemerides
- accel_third_body_venus_de440s: Venus perturbation using DE440s ephemerides
- accel_third_body_mars_de440s: Mars perturbation using DE440s ephemerides
- accel_third_body_jupiter_de440s: Jupiter perturbation using DE440s ephemerides
- accel_third_body_saturn_de440s: Saturn perturbation using DE440s ephemerides
- accel_third_body_uranus_de440s: Uranus perturbation using DE440s ephemerides
- accel_third_body_neptune_de440s: Neptune perturbation using DE440s ephemerides

Gravity:
- accel_point_mass_gravity: Point-mass gravity acceleration
- GravityModelType: Enum for gravity model types (packaged or from file)
- GravityModelTideSystem: Enum for tide system conventions
- GravityModelErrors: Enum for error estimation types
- GravityModelNormalization: Enum for coefficient normalization conventions
- GravityModel: Spherical harmonic gravity model class
- accel_gravity_spherical_harmonics: Spherical harmonic gravity acceleration

Drag and SRP:
- accel_drag: Atmospheric drag acceleration
- accel_solar_radiation_pressure: Solar radiation pressure acceleration
- eclipse_conical: Conical (penumbral) shadow model for eclipse detection
- eclipse_cylindrical: Cylindrical shadow model for eclipse detection

Relativity:
- accel_relativity: Relativistic acceleration effects

The DE440s functions use a global, thread-safe ephemeris context that is loaded
once and shared across all calls for performance.
"""

from brahe._brahe import (
    # Ephemerides
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
    # Third-Body Accelerations
    accel_third_body_sun,
    accel_third_body_moon,
    accel_third_body_sun_de440s,
    accel_third_body_moon_de440s,
    accel_third_body_mercury_de440s,
    accel_third_body_venus_de440s,
    accel_third_body_mars_de440s,
    accel_third_body_jupiter_de440s,
    accel_third_body_saturn_de440s,
    accel_third_body_uranus_de440s,
    accel_third_body_neptune_de440s,
    # Gravity
    accel_point_mass_gravity,
    GravityModelType,
    GravityModelTideSystem,
    GravityModelErrors,
    GravityModelNormalization,
    GravityModel,
    accel_gravity_spherical_harmonics,
    # Atmospheric Density Models
    density_harris_priester,
    density_nrlmsise00,
    density_nrlmsise00_geod,
    # Drag, SRP, and Relativity
    accel_drag,
    accel_solar_radiation_pressure,
    eclipse_conical,
    eclipse_cylindrical,
    accel_relativity,
)

__all__ = [
    # Ephemerides
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
    # Third-Body Accelerations
    "accel_third_body_sun",
    "accel_third_body_moon",
    "accel_third_body_sun_de440s",
    "accel_third_body_moon_de440s",
    "accel_third_body_mercury_de440s",
    "accel_third_body_venus_de440s",
    "accel_third_body_mars_de440s",
    "accel_third_body_jupiter_de440s",
    "accel_third_body_saturn_de440s",
    "accel_third_body_uranus_de440s",
    "accel_third_body_neptune_de440s",
    # Gravity
    "accel_point_mass_gravity",
    "GravityModelType",
    "GravityModelTideSystem",
    "GravityModelErrors",
    "GravityModelNormalization",
    "GravityModel",
    "accel_gravity_spherical_harmonics",
    # Atmospheric Density Models
    "density_harris_priester",
    "density_nrlmsise00",
    "density_nrlmsise00_geod",
    # Drag, SRP, and Relativity
    "accel_drag",
    "accel_solar_radiation_pressure",
    "eclipse_conical",
    "eclipse_cylindrical",
    "accel_relativity",
]
