# /// script
# dependencies = ["brahe"]
# ///
"""
Create a range constraint capping the maximum slant range to a satellite.
"""

import brahe as bh

# Require satellite slant range to be no more than 5000 km
constraint = bh.RangeConstraint(min_range_m=None, max_range_m=5_000_000.0)

print(f"Created: {constraint}")
