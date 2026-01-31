# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrate lunar reference frame rotation matrices
"""

import brahe as bh
import numpy as np

# Get constant bias matrix
B = bh.bias_moon_j2000()

# Get rotation matrices (LCRF ↔ MOON_J2000)
R_lcrf_to_j2000 = bh.rotation_lcrf_to_moon_j2000()
R_j2000_to_lcrf = bh.rotation_moon_j2000_to_lcrf()

# Using LCI alias
R_lci_to_j2000 = bh.rotation_lci_to_moon_j2000()  # Same as LCRF version

print("Bias matrix:")
print(B)
print("\nLCRF to MOON_J2000 rotation:")
print(R_lcrf_to_j2000)
print("\nMOON_J2000 to LCRF rotation:")
print(R_j2000_to_lcrf)
print("\nLCI to MOON_J2000 rotation (same as LCRF):")
print(R_lci_to_j2000)
