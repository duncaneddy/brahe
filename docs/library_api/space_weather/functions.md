# Space Weather Functions

Global space weather management and query functions.

**Module**: `brahe.space_weather`

## Managing Global Provider

::: brahe.initialize_sw
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.set_global_space_weather_provider
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.get_global_sw_initialization
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.get_global_sw_len
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.get_global_sw_type
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.get_global_sw_extrapolation
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.get_global_sw_mjd_min
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.get_global_sw_mjd_max
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.get_global_sw_mjd_last_observed
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.get_global_sw_mjd_last_daily_predicted
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.get_global_sw_mjd_last_monthly_predicted
    options:
      show_root_heading: true
      show_root_full_path: false

## Querying Kp Index

::: brahe.get_global_kp
    options:
      show_root_heading: true
      show_root_full_path: false


::: brahe.get_global_kp_all
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.get_global_kp_daily
    options:
      show_root_heading: true
      show_root_full_path: false

## Querying Ap Index

::: brahe.get_global_ap
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.get_global_ap_all
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.get_global_ap_daily
    options:
      show_root_heading: true
      show_root_full_path: false

## Querying F10.7 Solar Flux

::: brahe.get_global_f107_observed
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.get_global_f107_adjusted
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.get_global_f107_obs_avg81
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.get_global_f107_adj_avg81
    options:
      show_root_heading: true
      show_root_full_path: false

## Querying Sunspot Number

::: brahe.get_global_sunspot_number
    options:
      show_root_heading: true
      show_root_full_path: false

## Range Queries

::: brahe.get_global_last_kp
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.get_global_last_ap
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.get_global_last_daily_kp
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.get_global_last_daily_ap
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.get_global_last_f107
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.get_global_last_kpap_epochs
    options:
      show_root_heading: true
      show_root_full_path: false

::: brahe.get_global_last_daily_epochs
    options:
      show_root_heading: true
      show_root_full_path: false

## See Also

- [FileSpaceWeatherProvider](file_provider.md)
- [CachingSpaceWeatherProvider](caching_provider.md)
- [StaticSpaceWeatherProvider](static_provider.md)
