# /// script
# dependencies = ["brahe"]
# ///
"""
Create an elevation constraint with both minimum and maximum elevation limits for side-looking sensors.
"""

import brahe as bh

# Side-looking sensor with elevation range
constraint = bh.ElevationConstraint(min_elevation_deg=10.0, max_elevation_deg=80.0)

print(f"Created: {constraint}")
# Created: ElevationConstraint(10.00° - 80.00°)
