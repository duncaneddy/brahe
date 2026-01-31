# /// script
# dependencies = ["brahe"]
# ///
"""
Transform position vectors between lunar reference frames
"""

import brahe as bh
import numpy as np

# Position 100 km above lunar surface in LCRF
r_lcrf = np.array([bh.R_MOON + 100e3, 0.0, 0.0])

# Transform to MOON_J2000
r_j2000 = bh.position_lcrf_to_moon_j2000(r_lcrf)

# Transform back
r_lcrf_back = bh.position_moon_j2000_to_lcrf(r_j2000)

# Using LCI alias
r_j2000_alt = bh.position_lci_to_moon_j2000(r_lcrf)

print(f"Position in LCRF: {r_lcrf}")
print(f"Position in MOON_J2000: {r_j2000}")
print(f"Back to LCRF: {r_lcrf_back}")
print(f"Difference (should be ~0): {np.linalg.norm(r_lcrf - r_lcrf_back)}")
