# /// script
# dependencies = ["brahe"]
# ///
"""
Initialize Static Space Weather Providers
"""

import brahe as bh

# Method 1: Static Space Weather Provider - All Zeros
sw_static_zeros = bh.StaticSpaceWeatherProvider.from_zero()
bh.set_global_space_weather_provider(sw_static_zeros)

# Method 2: Static Space Weather Provider - Custom Constant Values
# Parameters: kp, ap, f107_obs, f107_adj, sunspot_number
sw_static_values = bh.StaticSpaceWeatherProvider.from_values(
    3.0, 15.0, 150.0, 150.0, 100
)
bh.set_global_space_weather_provider(sw_static_values)
