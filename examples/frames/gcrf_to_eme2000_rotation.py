# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Get GCRF to EME2000 rotation matrix and use it to transform position vectors
"""

import brahe as bh
import numpy as np

# Get constant rotation matrix from GCRF to EME2000
R_gcrf_to_eme2000 = bh.rotation_gcrf_to_eme2000()

print("GCRF to EME2000 rotation matrix:")
print(
    f"  [{R_gcrf_to_eme2000[0, 0]:13.10f}, {R_gcrf_to_eme2000[0, 1]:13.10f}, {R_gcrf_to_eme2000[0, 2]:13.10f}]"
)
print(
    f"  [{R_gcrf_to_eme2000[1, 0]:13.10f}, {R_gcrf_to_eme2000[1, 1]:13.10f}, {R_gcrf_to_eme2000[1, 2]:13.10f}]"
)
print(
    f"  [{R_gcrf_to_eme2000[2, 0]:13.10f}, {R_gcrf_to_eme2000[2, 1]:13.10f}, {R_gcrf_to_eme2000[2, 2]:13.10f}]\n"
)

R_eme2000_to_gcrf = bh.rotation_eme2000_to_gcrf()
print("Verification: R_gcrf_to_eme2000 = R_eme2000_to_gcrf^T")
print(
    f"  Max difference: {np.max(np.abs(R_gcrf_to_eme2000 - R_eme2000_to_gcrf.T)):.2e}\n"
)
# Verification: R_gcrf_to_eme2000 = R_eme2000_to_gcrf^T

identity = R_gcrf_to_eme2000 @ R_gcrf_to_eme2000.T
print("Verify orthonormality (R @ R^T should be identity):")
print(f"  Max deviation from identity: {np.max(np.abs(identity - np.eye(3))):.2e}\n")

# Define orbital elements for testing transformation
oe = np.array(
    [
        bh.R_EARTH + 500e3,  # Semi-major axis (m)
        0.01,  # Eccentricity
        97.8,  # Inclination (deg)
        15.0,  # RAAN (deg)
        30.0,  # Argument of periapsis (deg)
        45.0,  # Mean anomaly (deg)
    ]
)

# Convert to EME2000, transform to GCRF, and extract position
state_eme2000 = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
state_gcrf = bh.state_eme2000_to_gcrf(state_eme2000)
pos_gcrf = state_gcrf[0:3]

print("Satellite position in GCRF:")
print(f"  [{pos_gcrf[0]:.3f}, {pos_gcrf[1]:.3f}, {pos_gcrf[2]:.3f}] m\n")

# Transform using rotation matrix
pos_eme2000_matrix = R_gcrf_to_eme2000 @ pos_gcrf

print("Satellite position in EME2000 (using rotation matrix):")
print(
    f"  [{pos_eme2000_matrix[0]:.3f}, {pos_eme2000_matrix[1]:.3f}, {pos_eme2000_matrix[2]:.3f}] m"
)

pos_eme2000_direct = bh.position_gcrf_to_eme2000(pos_gcrf)
print("\nSatellite position in EME2000 (using position_gcrf_to_eme2000):")
print(
    f"  [{pos_eme2000_direct[0]:.3f}, {pos_eme2000_direct[1]:.3f}, {pos_eme2000_direct[2]:.3f}] m"
)

diff = np.linalg.norm(pos_eme2000_matrix - pos_eme2000_direct)
print(f"\nDifference between methods: {diff:.6e} m")
print("\nNote: Frame bias is constant (same at all epochs)")
