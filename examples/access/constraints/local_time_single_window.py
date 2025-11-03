# /// script
# dependencies = ["brahe"]
# ///
"""
Create a local time constraint for daylight-only imaging (8:00 AM to 6:00 PM local solar time).
"""

import brahe as bh

# Daylight imaging: 8:00 AM to 6:00 PM local solar time
# Times in military format: HHMM
constraint = bh.LocalTimeConstraint(time_windows=[(800, 1800)])

print(f"Created: {constraint}")
# Created: LocalTimeConstraint(08:00-18:00)
