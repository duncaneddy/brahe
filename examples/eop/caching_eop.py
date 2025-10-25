# /// script
# dependencies = ["brahe"]
# ///
"""
This example demonstrates ways to initialize EOP from files using Brahe.
"""

from pathlib import Path
import brahe as bh

# Method 1: Initialize from Caching EOP Provider -> Internally caches data to ~/.cache/brahe/eop
provider = bh.CachingEOPProvider(
    eop_type="StandardBulletinA",
    max_age_seconds=7 * 86400,  # Maximum age of file before refreshing
    auto_refresh=False,  # Check staleness of every access
    interpolate=True,
    extrapolate="Hold",
)
bh.set_global_eop_provider(provider)

# Method 2: Initialize from Caching EOP Provider with custom location
provider_custom = bh.CachingEOPProvider(
    filepath=str(
        Path(bh.get_brahe_cache_dir()) / "my_eop.txt"
    ),  # Replace with desired file path to load / save from
    eop_type="StandardBulletinA",
    max_age_seconds=7 * 86400,  # Maximum age of file before refreshing
    auto_refresh=False,  # Check staleness of every access
    interpolate=True,
    extrapolate="Hold",
)
bh.set_global_eop_provider(provider_custom)
