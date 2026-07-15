"""
Orbit Dynamics Module

This module provides functions for computing celestial body ephemerides and
acceleration models for orbital dynamics.

Ephemerides:
-----------
- sun_position: Low-precision analytical solar position
- moon_position: Low-precision analytical lunar position
- sun_position_spice: High-precision solar position using NAIF DE ephemerides
- sun_velocity_spice: High-precision solar velocity using NAIF DE ephemerides
- sun_state_spice: High-precision solar state using NAIF DE ephemerides
- moon_position_spice: High-precision lunar position using NAIF DE ephemerides
- moon_velocity_spice: High-precision lunar velocity using NAIF DE ephemerides
- moon_state_spice: High-precision lunar state using NAIF DE ephemerides
- mercury_position_spice: High-precision Mercury position using NAIF DE ephemerides
- mercury_velocity_spice: High-precision Mercury velocity using NAIF DE ephemerides
- mercury_state_spice: High-precision Mercury state using NAIF DE ephemerides
- venus_position_spice: High-precision Venus position using NAIF DE ephemerides
- venus_velocity_spice: High-precision Venus velocity using NAIF DE ephemerides
- venus_state_spice: High-precision Venus state using NAIF DE ephemerides
- mars_position_spice: High-precision Mars position using NAIF DE ephemerides
- mars_velocity_spice: High-precision Mars velocity using NAIF DE ephemerides
- mars_state_spice: High-precision Mars state using NAIF DE ephemerides
- jupiter_position_spice: High-precision Jupiter position using NAIF DE ephemerides
- jupiter_velocity_spice: High-precision Jupiter velocity using NAIF DE ephemerides
- jupiter_state_spice: High-precision Jupiter state using NAIF DE ephemerides
- saturn_position_spice: High-precision Saturn position using NAIF DE ephemerides
- saturn_velocity_spice: High-precision Saturn velocity using NAIF DE ephemerides
- saturn_state_spice: High-precision Saturn state using NAIF DE ephemerides
- uranus_position_spice: High-precision Uranus position using NAIF DE ephemerides
- uranus_velocity_spice: High-precision Uranus velocity using NAIF DE ephemerides
- uranus_state_spice: High-precision Uranus state using NAIF DE ephemerides
- neptune_position_spice: High-precision Neptune position using NAIF DE ephemerides
- neptune_velocity_spice: High-precision Neptune velocity using NAIF DE ephemerides
- neptune_state_spice: High-precision Neptune state using NAIF DE ephemerides
- mars_barycenter_position_spice: Mars-system barycenter position (DE kernel only)
- mars_barycenter_velocity_spice: Mars-system barycenter velocity (DE kernel only)
- mars_barycenter_state_spice: Mars-system barycenter state (DE kernel only)
- jupiter_barycenter_position_spice: Jupiter-system barycenter position (DE kernel only)
- jupiter_barycenter_velocity_spice: Jupiter-system barycenter velocity (DE kernel only)
- jupiter_barycenter_state_spice: Jupiter-system barycenter state (DE kernel only)
- saturn_barycenter_position_spice: Saturn-system barycenter position (DE kernel only)
- saturn_barycenter_velocity_spice: Saturn-system barycenter velocity (DE kernel only)
- saturn_barycenter_state_spice: Saturn-system barycenter state (DE kernel only)
- uranus_barycenter_position_spice: Uranus-system barycenter position (DE kernel only)
- uranus_barycenter_velocity_spice: Uranus-system barycenter velocity (DE kernel only)
- uranus_barycenter_state_spice: Uranus-system barycenter state (DE kernel only)
- neptune_barycenter_position_spice: Neptune-system barycenter position (DE kernel only)
- neptune_barycenter_velocity_spice: Neptune-system barycenter velocity (DE kernel only)
- neptune_barycenter_state_spice: Neptune-system barycenter state (DE kernel only)
- solar_system_barycenter_position_spice: High-precision SSB position using NAIF DE ephemerides
- solar_system_barycenter_velocity_spice: High-precision SSB velocity using NAIF DE ephemerides
- solar_system_barycenter_state_spice: High-precision SSB state using NAIF DE ephemerides
- ssb_position_spice: Convenience alias for solar_system_barycenter_position_spice
- ssb_velocity_spice: Convenience alias for solar_system_barycenter_velocity_spice
- ssb_state_spice: Convenience alias for solar_system_barycenter_state_spice

Acceleration Models:
-------------------
Third-Body Perturbations:
- accel_third_body_sun: Sun perturbation using analytical ephemerides
- accel_third_body_moon: Moon perturbation using analytical ephemerides
- accel_third_body_sun_spice: Sun perturbation using DE ephemerides
- accel_third_body_moon_spice: Moon perturbation using DE ephemerides
- accel_third_body_mercury_spice: Mercury perturbation using DE ephemerides
- accel_third_body_venus_spice: Venus perturbation using DE ephemerides
- accel_third_body_mars_spice: Mars perturbation using DE ephemerides
- accel_third_body_jupiter_spice: Jupiter perturbation using DE ephemerides
- accel_third_body_saturn_spice: Saturn perturbation using DE ephemerides
- accel_third_body_uranus_spice: Uranus perturbation using DE ephemerides
- accel_third_body_neptune_spice: Neptune perturbation using DE ephemerides
- accel_third_body_for_body: Central-body-aware third-body acceleration (Moon, Mars, EMB, SSB, Custom)
- accel_third_body_field_for_body: Third-body acceleration with a configured gravity model (point-mass, spherical-harmonic, Earth-zonal)

Gravity:
- accel_point_mass_gravity: Point-mass gravity acceleration
- accel_earth_zonal_gravity: Earth zonal harmonics (J2-J6) acceleration
- GravityModelType: Enum for gravity model types (packaged or from file)
- GravityModelTideSystem: Enum for tide system conventions
- GravityModelErrors: Enum for error estimation types
- GravityModelNormalization: Enum for coefficient normalization conventions
- GravityModelCoefficients: Enum for which precomputed coefficient sets a model builds
- GravityModel: Spherical harmonic gravity model class
- accel_gravity_spherical_harmonics: Spherical harmonic gravity acceleration (auto-dispatched)
- accel_gravity_spherical_harmonics_clenshaw: Spherical harmonic gravity acceleration (Clenshaw kernel)
- accel_gravity_spherical_harmonics_cunningham: Spherical harmonic gravity acceleration (Cunningham kernel)
- set_global_gravity_model: Set the process-wide global gravity model
- set_global_gravity_model_to_tide_system: Convert a model to a tide system, then set it as global
- get_global_gravity_model: Get a copy of the process-wide global gravity model
- clear_gravity_model_cache: Clear the process-wide gravity model cache

Drag and SRP:
- accel_drag: Atmospheric drag acceleration
- accel_drag_for_body: Atmospheric drag acceleration about an arbitrary central body
- accel_solar_radiation_pressure: Solar radiation pressure acceleration
- eclipse_conical: Conical (penumbral) shadow model for eclipse detection
- eclipse_conical_for_body: Conical shadow model with an arbitrary occulting body
- eclipse_cylindrical: Cylindrical shadow model for eclipse detection
- eclipse_cylindrical_for_body: Cylindrical shadow model with an arbitrary occulting body

Relativity:
- accel_relativity: Relativistic acceleration effects
- accel_relativity_for_body: Relativistic acceleration about a central body with arbitrary GM

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
    sun_position_spice,
    sun_velocity_spice,
    sun_state_spice,
    moon_position_spice,
    moon_velocity_spice,
    moon_state_spice,
    mercury_position_spice,
    mercury_velocity_spice,
    mercury_state_spice,
    venus_position_spice,
    venus_velocity_spice,
    venus_state_spice,
    mars_position_spice,
    mars_velocity_spice,
    mars_state_spice,
    jupiter_position_spice,
    jupiter_velocity_spice,
    jupiter_state_spice,
    saturn_position_spice,
    saturn_velocity_spice,
    saturn_state_spice,
    uranus_position_spice,
    uranus_velocity_spice,
    uranus_state_spice,
    neptune_position_spice,
    neptune_velocity_spice,
    neptune_state_spice,
    mars_barycenter_position_spice,
    mars_barycenter_velocity_spice,
    mars_barycenter_state_spice,
    jupiter_barycenter_position_spice,
    jupiter_barycenter_velocity_spice,
    jupiter_barycenter_state_spice,
    saturn_barycenter_position_spice,
    saturn_barycenter_velocity_spice,
    saturn_barycenter_state_spice,
    uranus_barycenter_position_spice,
    uranus_barycenter_velocity_spice,
    uranus_barycenter_state_spice,
    neptune_barycenter_position_spice,
    neptune_barycenter_velocity_spice,
    neptune_barycenter_state_spice,
    solar_system_barycenter_position_spice,
    solar_system_barycenter_velocity_spice,
    solar_system_barycenter_state_spice,
    ssb_position_spice,
    ssb_velocity_spice,
    ssb_state_spice,
    # Third-Body Accelerations
    accel_third_body_sun,
    accel_third_body_moon,
    accel_third_body_sun_spice,
    accel_third_body_moon_spice,
    accel_third_body_mercury_spice,
    accel_third_body_venus_spice,
    accel_third_body_mars_spice,
    accel_third_body_jupiter_spice,
    accel_third_body_saturn_spice,
    accel_third_body_uranus_spice,
    accel_third_body_neptune_spice,
    accel_third_body_for_body,
    accel_third_body_field_for_body,
    # Gravity
    accel_point_mass_gravity,
    accel_earth_zonal_gravity,
    GravityModelType,
    GravityModelTideSystem,
    GravityModelErrors,
    GravityModelNormalization,
    GravityModelCoefficients,
    GravityModel,
    accel_gravity_spherical_harmonics,
    accel_gravity_spherical_harmonics_clenshaw,
    accel_gravity_spherical_harmonics_cunningham,
    set_global_gravity_model,
    set_global_gravity_model_to_tide_system,
    get_global_gravity_model,
    clear_gravity_model_cache,
    # Atmospheric Density Models
    density_harris_priester,
    density_nrlmsise00,
    density_nrlmsise00_geod,
    # Magnetic Field Models
    igrf_geodetic_enz,
    igrf_geocentric_enz,
    igrf_ecef,
    wmmhr_geodetic_enz,
    wmmhr_geocentric_enz,
    wmmhr_ecef,
    # Drag, SRP, and Relativity
    accel_drag,
    accel_drag_for_body,
    accel_solar_radiation_pressure,
    eclipse_conical,
    eclipse_conical_for_body,
    eclipse_cylindrical,
    eclipse_cylindrical_for_body,
    accel_relativity,
    accel_relativity_for_body,
)

__all__ = [
    # Enums
    "EphemerisSource",
    # Ephemerides
    "sun_position",
    "moon_position",
    "sun_position_spice",
    "sun_velocity_spice",
    "sun_state_spice",
    "moon_position_spice",
    "moon_velocity_spice",
    "moon_state_spice",
    "mercury_position_spice",
    "mercury_velocity_spice",
    "mercury_state_spice",
    "venus_position_spice",
    "venus_velocity_spice",
    "venus_state_spice",
    "mars_position_spice",
    "mars_velocity_spice",
    "mars_state_spice",
    "jupiter_position_spice",
    "jupiter_velocity_spice",
    "jupiter_state_spice",
    "saturn_position_spice",
    "saturn_velocity_spice",
    "saturn_state_spice",
    "uranus_position_spice",
    "uranus_velocity_spice",
    "uranus_state_spice",
    "neptune_position_spice",
    "neptune_velocity_spice",
    "neptune_state_spice",
    "mars_barycenter_position_spice",
    "mars_barycenter_velocity_spice",
    "mars_barycenter_state_spice",
    "jupiter_barycenter_position_spice",
    "jupiter_barycenter_velocity_spice",
    "jupiter_barycenter_state_spice",
    "saturn_barycenter_position_spice",
    "saturn_barycenter_velocity_spice",
    "saturn_barycenter_state_spice",
    "uranus_barycenter_position_spice",
    "uranus_barycenter_velocity_spice",
    "uranus_barycenter_state_spice",
    "neptune_barycenter_position_spice",
    "neptune_barycenter_velocity_spice",
    "neptune_barycenter_state_spice",
    "solar_system_barycenter_position_spice",
    "solar_system_barycenter_velocity_spice",
    "solar_system_barycenter_state_spice",
    "ssb_position_spice",
    "ssb_velocity_spice",
    "ssb_state_spice",
    # Third-Body Accelerations
    "accel_third_body_sun",
    "accel_third_body_moon",
    "accel_third_body_sun_spice",
    "accel_third_body_moon_spice",
    "accel_third_body_mercury_spice",
    "accel_third_body_venus_spice",
    "accel_third_body_mars_spice",
    "accel_third_body_jupiter_spice",
    "accel_third_body_saturn_spice",
    "accel_third_body_uranus_spice",
    "accel_third_body_neptune_spice",
    "accel_third_body_for_body",
    "accel_third_body_field_for_body",
    # Gravity
    "accel_point_mass_gravity",
    "accel_earth_zonal_gravity",
    "GravityModelType",
    "GravityModelTideSystem",
    "GravityModelErrors",
    "GravityModelNormalization",
    "GravityModelCoefficients",
    "GravityModel",
    "accel_gravity_spherical_harmonics",
    "accel_gravity_spherical_harmonics_clenshaw",
    "accel_gravity_spherical_harmonics_cunningham",
    "set_global_gravity_model",
    "set_global_gravity_model_to_tide_system",
    "get_global_gravity_model",
    "clear_gravity_model_cache",
    # Atmospheric Density Models
    "density_harris_priester",
    "density_nrlmsise00",
    "density_nrlmsise00_geod",
    # Magnetic Field Models
    "igrf_geodetic_enz",
    "igrf_geocentric_enz",
    "igrf_ecef",
    "wmmhr_geodetic_enz",
    "wmmhr_geocentric_enz",
    "wmmhr_ecef",
    # Drag, SRP, and Relativity
    "accel_drag",
    "accel_drag_for_body",
    "accel_solar_radiation_pressure",
    "eclipse_conical",
    "eclipse_conical_for_body",
    "eclipse_cylindrical",
    "eclipse_cylindrical_for_body",
    "accel_relativity",
    "accel_relativity_for_body",
]
