"""
Space Weather Module

Space weather data management and access for atmospheric drag calculations.

This module provides:
- Space weather data providers (static, file-based, and caching)
- Global space weather provider management
- Access to Kp/Ap geomagnetic indices and F10.7 solar flux

Space weather data is required for atmospheric density models used in satellite
drag calculations and orbit propagation with atmospheric perturbations.
"""

from brahe._brahe import (
    # Provider classes
    StaticSpaceWeatherProvider,
    FileSpaceWeatherProvider,
    CachingSpaceWeatherProvider,
    # Global provider management
    set_global_space_weather_provider,
    initialize_sw,
    # Global Kp data access
    get_global_kp,
    get_global_kp_all,
    get_global_kp_daily,
    # Global Ap data access
    get_global_ap,
    get_global_ap_all,
    get_global_ap_daily,
    # Global F10.7 data access
    get_global_f107_observed,
    get_global_f107_adjusted,
    get_global_f107_obs_avg81,
    get_global_f107_adj_avg81,
    # Global sunspot data
    get_global_sunspot_number,
    # Global historical data access
    get_global_last_kp,
    get_global_last_ap,
    get_global_last_daily_kp,
    get_global_last_daily_ap,
    get_global_last_f107,
    get_global_last_kpap_epochs,
    get_global_last_daily_epochs,
    # Global provider information
    get_global_sw_initialization,
    get_global_sw_len,
    get_global_sw_type,
    get_global_sw_extrapolation,
    get_global_sw_mjd_min,
    get_global_sw_mjd_max,
    get_global_sw_mjd_last_observed,
    get_global_sw_mjd_last_daily_predicted,
    get_global_sw_mjd_last_monthly_predicted,
)

__all__ = [
    # Provider classes
    "StaticSpaceWeatherProvider",
    "FileSpaceWeatherProvider",
    "CachingSpaceWeatherProvider",
    # Global provider management
    "set_global_space_weather_provider",
    "initialize_sw",
    # Global Kp data access
    "get_global_kp",
    "get_global_kp_all",
    "get_global_kp_daily",
    # Global Ap data access
    "get_global_ap",
    "get_global_ap_all",
    "get_global_ap_daily",
    # Global F10.7 data access
    "get_global_f107_observed",
    "get_global_f107_adjusted",
    "get_global_f107_obs_avg81",
    "get_global_f107_adj_avg81",
    # Global sunspot data
    "get_global_sunspot_number",
    # Global historical data access
    "get_global_last_kp",
    "get_global_last_ap",
    "get_global_last_daily_kp",
    "get_global_last_daily_ap",
    "get_global_last_f107",
    "get_global_last_kpap_epochs",
    "get_global_last_daily_epochs",
    # Global provider information
    "get_global_sw_initialization",
    "get_global_sw_len",
    "get_global_sw_type",
    "get_global_sw_extrapolation",
    "get_global_sw_mjd_min",
    "get_global_sw_mjd_max",
    "get_global_sw_mjd_last_observed",
    "get_global_sw_mjd_last_daily_predicted",
    "get_global_sw_mjd_last_monthly_predicted",
]
