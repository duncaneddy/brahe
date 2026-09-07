# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Get the GCRF to MOD rotation matrix and show that it reduces to the frame bias at J2000
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

epc = bh.Epoch(2024, 1, 1, 12, 0, 0.0, time_system=bh.UTC)
print(f"Epoch: {epc}")

R_gcrf_to_mod = bh.rotation_gcrf_to_mod(epc)

print("\nGCRF to MOD rotation matrix:")
print(
    f"  [{R_gcrf_to_mod[0, 0]:13.10f}, {R_gcrf_to_mod[0, 1]:13.10f}, {R_gcrf_to_mod[0, 2]:13.10f}]"
)
print(
    f"  [{R_gcrf_to_mod[1, 0]:13.10f}, {R_gcrf_to_mod[1, 1]:13.10f}, {R_gcrf_to_mod[1, 2]:13.10f}]"
)
print(
    f"  [{R_gcrf_to_mod[2, 0]:13.10f}, {R_gcrf_to_mod[2, 1]:13.10f}, {R_gcrf_to_mod[2, 2]:13.10f}]\n"
)

epc_j2000 = bh.Epoch(2000, 1, 1, 12, 0, 0.0, time_system=bh.TT)
print(f"J2000 epoch: {epc_j2000}")

R_gcrf_to_mod_j2000 = bh.rotation_gcrf_to_mod(epc_j2000)

print("\nGCRF to MOD rotation matrix at J2000:")
print(
    f"  [{R_gcrf_to_mod_j2000[0, 0]:13.10f}, {R_gcrf_to_mod_j2000[0, 1]:13.10f}, {R_gcrf_to_mod_j2000[0, 2]:13.10f}]"
)
print(
    f"  [{R_gcrf_to_mod_j2000[1, 0]:13.10f}, {R_gcrf_to_mod_j2000[1, 1]:13.10f}, {R_gcrf_to_mod_j2000[1, 2]:13.10f}]"
)
print(
    f"  [{R_gcrf_to_mod_j2000[2, 0]:13.10f}, {R_gcrf_to_mod_j2000[2, 1]:13.10f}, {R_gcrf_to_mod_j2000[2, 2]:13.10f}]\n"
)

B = bh.bias_eme2000()
print("Comparison with the EME2000 frame bias matrix at J2000:")
print(f"  Max absolute difference: {np.max(np.abs(R_gcrf_to_mod_j2000 - B)):.2e}")
print("\nNote: at J2000 the IAU 2000 precession is identity, so MOD reduces")
print("to the constant frame bias between GCRF and EME2000.")
