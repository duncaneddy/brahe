# /// script
# dependencies = ["brahe"]
# ///
"""
Initialize Space Weather Providers from files
"""

import brahe as bh

# Method 1: Default Provider -> Uses packaged data file within Brahe
sw_file_default = bh.FileSpaceWeatherProvider.from_default_file()
bh.set_global_space_weather_provider(sw_file_default)

# Method 2: Custom File Path -> Replace with actual file path
if False:  # Change to True to enable custom file example
    sw_file_custom = bh.FileSpaceWeatherProvider.from_file(
        "/path/to/sw19571001.txt",  # Replace with actual file path
        "Hold",  # Extrapolation: "Zero", "Hold", or "Error"
    )
    bh.set_global_space_weather_provider(sw_file_custom)
