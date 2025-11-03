# /// script
# dependencies = ["brahe"]
# ///
"""
Create a composite constraint using AND logic (all constraints must be satisfied).
"""

import brahe as bh

# Elevation > 10° AND daylight hours
elev = bh.ElevationConstraint(min_elevation_deg=10.0)
daytime = bh.LocalTimeConstraint(time_windows=[(800, 1800)])

constraint = bh.ConstraintAll(constraints=[elev, daytime])

print(f"Created: {constraint}")
# Created: ElevationConstraint(>= 60.00°) || LookDirectionConstraint(Right)
