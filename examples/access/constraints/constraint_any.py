# /// script
# dependencies = ["brahe"]
# ///
"""
Create a composite constraint using OR logic (at least one constraint must be satisfied).
"""

import brahe as bh

# High elevation OR right-looking geometry
high_elev = bh.ElevationConstraint(min_elevation_deg=60.0)
right_look = bh.LookDirectionConstraint(allowed=bh.LookDirection.RIGHT)

constraint = bh.ConstraintAny(constraints=[high_elev, right_look])

print(f"Created: {constraint}")
# Created: ConstraintAny(constraints: [ElevationConstraint(...), LookDirectionConstraint(...)])
