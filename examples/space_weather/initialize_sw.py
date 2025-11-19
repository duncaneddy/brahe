# /// script
# dependencies = ["brahe"]
# ///
"""
Initialize Space Weather Providers with simplest way possible
"""

import brahe as bh

# Initialize with default caching provider (will download data as needed)
bh.initialize_sw()
