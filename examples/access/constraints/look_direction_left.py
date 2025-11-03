# /// script
# dependencies = ["brahe"]
# ///
"""
Create a look direction constraint requiring left-looking geometry.
"""

import brahe as bh
from brahe import LookDirection

# Require left-looking geometry
constraint = bh.LookDirectionConstraint(allowed=LookDirection.LEFT)

print(f"Created: {constraint}")
# Created: LookDirectionConstraint(Left)
