# /// script
# dependencies = ["brahe"]
# ///
"""
Create an azimuth constraint requiring satellites to be within a southeast-facing window.
"""

import brahe as bh

# Require satellite azimuth to be between 90 and 180 degrees (southeast quadrant)
constraint = bh.AzimuthConstraint(90.0, 180.0)

print(f"Created: {constraint}")
