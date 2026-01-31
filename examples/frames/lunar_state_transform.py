# /// script
# dependencies = ["brahe"]
# ///
"""
Transform state vectors between lunar reference frames
"""

import brahe as bh
import numpy as np

# Circular orbit state in LCRF [x, y, z, vx, vy, vz] (m, m/s)
state_lcrf = np.array([
    bh.R_MOON + 100e3, 0.0, 0.0,  # Position
    0.0, 1700.0, 0.0               # Velocity (~1.7 km/s orbital speed)
])

# Transform to MOON_J2000
state_j2000 = bh.state_lcrf_to_moon_j2000(state_lcrf)

# Transform back
state_lcrf_back = bh.state_moon_j2000_to_lcrf(state_j2000)

print(f"State in LCRF: {state_lcrf}")
print(f"State in MOON_J2000: {state_j2000}")
print(f"Back to LCRF: {state_lcrf_back}")
print(f"Difference (should be ~0): {np.linalg.norm(state_lcrf - state_lcrf_back)}")
