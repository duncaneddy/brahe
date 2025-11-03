# /// script
# dependencies = ["brahe"]
# ///
"""
Create a local time constraint with multiple windows for dawn and dusk passes.
"""

import brahe as bh

# Multiple windows: dawn and dusk passes
constraint = bh.LocalTimeConstraint(time_windows=[(600, 800), (1800, 2000)])

print(f"Created: {constraint}")
# Created: LocalTimeConstraint(06:00-08:00, 18:00-20:00)
