"""
Earth Orientation Parameters (EOP) Module

Earth orientation parameter data management and access.

This module provides:
- EOP data providers (static built-in data and file-based data)
- EOP data download utilities (IERS C04 and standard files)
- Global EOP provider management
- Access to UT1-UTC, polar motion, and length-of-day data

EOP data is required for high-precision reference frame transformations between
ECI (Earth-Centered Inertial) and ECEF (Earth-Centered Earth-Fixed) frames.
"""

from brahe._brahe import (
    # Provider classes
    StaticEOPProvider,
    FileEOPProvider,
    CachingEOPProvider,
    # Download functions
    download_c04_eop_file,
    download_standard_eop_file,
    # Global provider management
    set_global_eop_provider_from_static_provider,
    set_global_eop_provider_from_file_provider,
    set_global_eop_provider_from_caching_provider,
    # Global EOP data access
    get_global_ut1_utc,
    get_global_pm,
    get_global_dxdy,
    get_global_lod,
    get_global_eop,
    # Global provider information
    get_global_eop_initialization,
    get_global_eop_len,
    get_global_eop_type,
    get_global_eop_extrapolation,
    get_global_eop_interpolation,
    get_global_eop_mjd_min,
    get_global_eop_mjd_max,
    get_global_eop_mjd_last_lod,
    get_global_eop_mjd_last_dxdy,
)

__all__ = [
    # Provider classes
    "StaticEOPProvider",
    "FileEOPProvider",
    "CachingEOPProvider",
    # Download functions
    "download_c04_eop_file",
    "download_standard_eop_file",
    # Global provider management
    "set_global_eop_provider_from_static_provider",
    "set_global_eop_provider_from_file_provider",
    "set_global_eop_provider_from_caching_provider",
    # Global EOP data access
    "get_global_ut1_utc",
    "get_global_pm",
    "get_global_dxdy",
    "get_global_lod",
    "get_global_eop",
    # Global provider information
    "get_global_eop_initialization",
    "get_global_eop_len",
    "get_global_eop_type",
    "get_global_eop_extrapolation",
    "get_global_eop_interpolation",
    "get_global_eop_mjd_min",
    "get_global_eop_mjd_max",
    "get_global_eop_mjd_last_lod",
    "get_global_eop_mjd_last_dxdy",
]
