# /// script
# dependencies = ["brahe"]
# ///
"""
Initialize Space Weather Providers using caching provider
"""

import brahe as bh

# Method 1: Create with custom settings
# - Downloads to ~/.cache/brahe/
# - Refreshes if file is older than 24 hours
sw_caching = bh.CachingSpaceWeatherProvider(
    max_age_seconds=86400,  # max_age: seconds (86400 = 24 hours)
    auto_refresh=False,  # check on each query
    extrapolate="Hold",  # extrapolation
)
bh.set_global_space_weather_provider(sw_caching)

# Method 2: Use initialize_sw() which creates a caching provider
bh.initialize_sw()
