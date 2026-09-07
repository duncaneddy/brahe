# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Get the MOD to TOD rotation matrix and compute the nutation angle it represents
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

epc = bh.Epoch(2024, 1, 1, 12, 0, 0.0, time_system=bh.UTC)
print(f"Epoch: {epc}")

R_mod_to_tod = bh.rotation_mod_to_tod(epc)

print("\nMOD to TOD rotation matrix:")
print(
    f"  [{R_mod_to_tod[0, 0]:13.10f}, {R_mod_to_tod[0, 1]:13.10f}, {R_mod_to_tod[0, 2]:13.10f}]"
)
print(
    f"  [{R_mod_to_tod[1, 0]:13.10f}, {R_mod_to_tod[1, 1]:13.10f}, {R_mod_to_tod[1, 2]:13.10f}]"
)
print(
    f"  [{R_mod_to_tod[2, 0]:13.10f}, {R_mod_to_tod[2, 1]:13.10f}, {R_mod_to_tod[2, 2]:13.10f}]\n"
)

trace = np.trace(R_mod_to_tod)
nutation_angle = np.degrees(np.arccos((trace - 1.0) / 2.0)) * 3600.0
print(
    f"Nutation angle (rotation angle of the MOD -> TOD matrix): {nutation_angle:.3f} arcsec"
)
