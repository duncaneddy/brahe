# /// script
# dependencies = ["brahe"]
# ///
"""
Create a basic elevation constraint requiring satellites to be at least 10 degrees above the horizon.
"""

import brahe as bh

# Require satellite to be at least 10 degrees above horizon
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)

print(f"Created: {constraint}")
# Created:  ElevationConstraint(>= 10.00°)
