# /// script
# dependencies = ["brahe"]
# ///
"""
Create a negation constraint (access when child constraint is NOT satisfied).
"""

import brahe as bh

# Avoid daylight (e.g., for night-time astronomy)
daytime = bh.LocalTimeConstraint(time_windows=[(600, 2000)])
night_only = bh.ConstraintNot(constraint=daytime)

print(f"Created: {night_only}")
# Created: !LocalTimeConstraint(06:00-20:00)
