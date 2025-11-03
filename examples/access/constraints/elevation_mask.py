# /// script
# dependencies = ["brahe"]
# ///
"""
Create an elevation mask constraint with azimuth-dependent elevation limits for terrain profiles.
"""

import brahe as bh

# Define elevation mask: [(azimuth_deg, elevation_deg), ...]
# Azimuth clockwise from North (0-360)
mask_points = [
    (0.0, 5.0),  # North: 5° minimum
    (90.0, 15.0),  # East: 15° minimum (mountains)
    (180.0, 8.0),  # South: 8° minimum
    (270.0, 10.0),  # West: 10° minimum
    (360.0, 5.0),  # Back to North
]

constraint = bh.ElevationMaskConstraint(mask_points)

print(f"Created: {constraint}")
# Created: ElevationMaskConstraint(Min: 5.00° at 0.00°, Max: 15.00° at 90.00°)
