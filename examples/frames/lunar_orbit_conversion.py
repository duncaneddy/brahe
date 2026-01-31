# /// script
# dependencies = ["brahe"]
# ///
"""
Example: Convert a lunar orbit state between LCRF and MOON_J2000
"""

import brahe as bh
import numpy as np

# Define a lunar orbit state in LCRF
# 100 km circular equatorial orbit
r_orbit = bh.R_MOON + 100e3
v_orbit = np.sqrt(bh.GM_MOON / r_orbit)  # Circular orbit velocity

state_lcrf = np.array([r_orbit, 0.0, 0.0, 0.0, v_orbit, 0.0])

# Convert to MOON_J2000 for comparison with legacy data
state_j2000 = bh.state_lcrf_to_moon_j2000(state_lcrf)

print(f"State in LCRF: {state_lcrf}")
print(f"State in MOON_J2000: {state_j2000}")

# The difference is very small (~23 milliarcseconds rotation)
diff = np.linalg.norm(state_lcrf[:3] - state_j2000[:3])
print(f"Position difference: {diff:.6f} m")  # Sub-meter difference
