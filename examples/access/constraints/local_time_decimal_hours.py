# /// script
# dependencies = ["brahe"]
# ///
"""
Create a local time constraint using decimal hours format.
"""

import brahe as bh

# Alternative: specify in decimal hours
constraint = bh.LocalTimeConstraint.from_hours(time_windows=[(8.0, 18.0)])

print(f"Created: {constraint}")
# Created: LocalTimeConstraint(08:00-18:00)
