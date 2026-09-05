# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Get GCRF to TOD rotation matrix and compare it with the CIO-based bias-precession-nutation matrix
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

epc = bh.Epoch(2024, 1, 1, 12, 0, 0.0, time_system=bh.UTC)
print(f"Epoch: {epc}")

R_gcrf_to_tod = bh.rotation_gcrf_to_tod(epc)

print("\nGCRF to TOD rotation matrix:")
print(
    f"  [{R_gcrf_to_tod[0, 0]:13.10f}, {R_gcrf_to_tod[0, 1]:13.10f}, {R_gcrf_to_tod[0, 2]:13.10f}]"
)
print(
    f"  [{R_gcrf_to_tod[1, 0]:13.10f}, {R_gcrf_to_tod[1, 1]:13.10f}, {R_gcrf_to_tod[1, 2]:13.10f}]"
)
print(
    f"  [{R_gcrf_to_tod[2, 0]:13.10f}, {R_gcrf_to_tod[2, 1]:13.10f}, {R_gcrf_to_tod[2, 2]:13.10f}]\n"
)

identity = R_gcrf_to_tod @ R_gcrf_to_tod.T
print("Verify orthonormality (R @ R^T should be identity):")
print(f"  Max deviation from identity: {np.max(np.abs(identity - np.eye(3))):.2e}\n")

R_cio = bh.bias_precession_nutation(epc)
print("Comparison with the CIO-based bias-precession-nutation matrix:")
print(f"  Max element difference: {np.max(np.abs(R_gcrf_to_tod - R_cio)):.6e}")
print("\nNote: rotation_gcrf_to_tod and bias_precession_nutation share the same")
print("Celestial Intermediate Pole direction (third row) and differ only by the")
print("equation of the origins, a rotation within the equatorial plane.")
