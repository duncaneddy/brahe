# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Create a custom constraint by implementing the AccessConstraintComputer interface.
"""

import brahe as bh
import numpy as np


class MaxRangeConstraint(bh.AccessConstraintComputer):
    """Limit access to satellites within a maximum range."""

    def __init__(self):
        self.max_range_m = 2000.0 * 1000.0  # 2000 km in meters

    def evaluate(self, epoch, satellite_state_ecef, location_ecef):
        """Return True when constraint is satisfied"""
        # Compute range vector from location to satellite
        range_vec = satellite_state_ecef[:3] - location_ecef
        range_m = np.linalg.norm(range_vec)

        return range_m <= self.max_range_m

    def name(self):
        return f"MaxRange({self.max_range_m / 1000:.0f}km)"


# Use custom constraint
constraint = MaxRangeConstraint()

print(f"Created: {constraint.name()}")
# Created: MaxRange(2000km)
