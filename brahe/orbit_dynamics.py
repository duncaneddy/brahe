"""
Orbit Dynamics Module

This module provides functions for computing celestial body ephemerides and
acceleration models for orbital dynamics.

Ephemerides:
-----------
- sun_position: Low-precision analytical solar position
- moon_position: Low-precision analytical lunar position
- sun_position_de: High-precision solar position using NAIF DE ephemerides
- moon_position_de: High-precision lunar position using NAIF DE ephemerides
- mercury_position_de: High-precision Mercury position using NAIF DE ephemerides
- venus_position_de: High-precision Venus position using NAIF DE ephemerides
- mars_position_de: High-precision Mars position using NAIF DE ephemerides
- jupiter_position_de: High-precision Jupiter position using NAIF DE ephemerides
- saturn_position_de: High-precision Saturn position using NAIF DE ephemerides
- uranus_position_de: High-precision Uranus position using NAIF DE ephemerides
- neptune_position_de: High-precision Neptune position using NAIF DE ephemerides
- solar_system_barycenter_position_de: High-precision SSB position using NAIF DE ephemerides
- ssb_position_de: Convenience alias for solar_system_barycenter_position_de
- initialize_ephemeris: Pre-initialize the DE ephemeris kernel

Acceleration Models:
-------------------
Third-Body Perturbations:
- accel_third_body_sun: Sun perturbation using analytical ephemerides
- accel_third_body_moon: Moon perturbation using analytical ephemerides
- accel_third_body_sun_de: Sun perturbation using DE ephemerides
- accel_third_body_moon_de: Moon perturbation using DE ephemerides
- accel_third_body_mercury_de: Mercury perturbation using DE ephemerides
- accel_third_body_venus_de: Venus perturbation using DE ephemerides
- accel_third_body_mars_de: Mars perturbation using DE ephemerides
- accel_third_body_jupiter_de: Jupiter perturbation using DE ephemerides
- accel_third_body_saturn_de: Saturn perturbation using DE ephemerides
- accel_third_body_uranus_de: Uranus perturbation using DE ephemerides
- accel_third_body_neptune_de: Neptune perturbation using DE ephemerides

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

The DE ephemeris functions use a global, thread-safe ephemeris context that is loaded
once and shared across all calls for performance. Choose between DE440s or DE440 kernels
via the EphemerisSource parameter.
"""

from brahe._brahe import (
    # Enums
    EphemerisSource,
    # Ephemerides
    sun_position,
    moon_position,
    sun_position_de,
    moon_position_de,
    mercury_position_de,
    venus_position_de,
    mars_position_de,
    jupiter_position_de,
    saturn_position_de,
    uranus_position_de,
    neptune_position_de,
    solar_system_barycenter_position_de,
    ssb_position_de,
    initialize_ephemeris,
    # Third-Body Accelerations
    accel_third_body_sun,
    accel_third_body_moon,
    accel_third_body_sun_de,
    accel_third_body_moon_de,
    accel_third_body_mercury_de,
    accel_third_body_venus_de,
    accel_third_body_mars_de,
    accel_third_body_jupiter_de,
    accel_third_body_saturn_de,
    accel_third_body_uranus_de,
    accel_third_body_neptune_de,
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
    # Enums
    "EphemerisSource",
    # Ephemerides
    "sun_position",
    "moon_position",
    "sun_position_de",
    "moon_position_de",
    "mercury_position_de",
    "venus_position_de",
    "mars_position_de",
    "jupiter_position_de",
    "saturn_position_de",
    "uranus_position_de",
    "neptune_position_de",
    "solar_system_barycenter_position_de",
    "ssb_position_de",
    "initialize_ephemeris",
    # Third-Body Accelerations
    "accel_third_body_sun",
    "accel_third_body_moon",
    "accel_third_body_sun_de",
    "accel_third_body_moon_de",
    "accel_third_body_mercury_de",
    "accel_third_body_venus_de",
    "accel_third_body_mars_de",
    "accel_third_body_jupiter_de",
    "accel_third_body_saturn_de",
    "accel_third_body_uranus_de",
    "accel_third_body_neptune_de",
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
