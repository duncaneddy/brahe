# /// script
# dependencies = ["brahe"]
# ///
"""
Create complex nested constraint logic by combining composition operators.
"""

import brahe as bh

# Complex constraint: (High elevation AND daylight)
# Note: Python API currently supports single-level composition
# For nested constraints, use Rust API with Box<dyn AccessConstraint>

# High elevation AND daylight
high_elev = bh.ElevationConstraint(min_elevation_deg=60.0)
daytime = bh.LocalTimeConstraint(time_windows=[(800, 1800)])
look_right = bh.LookDirectionConstraint(allowed=bh.LookDirection.RIGHT)

# Combine multiple constraints with AND
constraint = bh.ConstraintAll(constraints=[high_elev, daytime, look_right])

print(f"Created: {constraint}")
# Created: ElevationConstraint(>= 60.00°) && LocalTimeConstraint(08:00-18:00) && LookDirectionConstraint(Right)
