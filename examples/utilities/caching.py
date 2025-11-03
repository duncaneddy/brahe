# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrate cache directory management utilities.

This example shows how to get the cache directory paths used by Brahe
for storing downloaded data such as EOP and TLE files.
"""

import brahe as bh

bh.initialize_eop()

# Get main cache directory
cache_dir = bh.get_brahe_cache_dir()
print(f"Main cache directory: {cache_dir}")

# Get cache subdirectory for EOP data
eop_cache = bh.get_eop_cache_dir()
print(f"EOP cache directory: {eop_cache}")

# Get cache subdirectory for CelesTrak data
celestrak_cache = bh.get_celestrak_cache_dir()
print(f"CelesTrak cache directory: {celestrak_cache}")

# Get a custom subdirectory within the cache
custom_cache = bh.get_brahe_cache_dir_with_subdir("custom_data")
print(f"Custom cache subdirectory: {custom_cache}")

# Note: All directories are automatically created if they don't exist
# You can override the default location by setting the BRAHE_CACHE
# environment variable

# Expected output (paths will vary by system):
# Main cache directory: /home/USER/.cache/brahe
# EOP cache directory: /home/USER/.cache/brahe/eop
# CelesTrak cache directory: /home/USER/.cache/brahe/celestrak
# Custom cache subdirectory: /home/USER/.cache/brahe/custom_data
