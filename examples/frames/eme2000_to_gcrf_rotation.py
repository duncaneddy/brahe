# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Get EME2000 to GCRF rotation matrix and use it to transform position vectors
"""

import brahe as bh
import numpy as np

# Get constant rotation matrix from EME2000 to GCRF
R_eme2000_to_gcrf = bh.rotation_eme2000_to_gcrf()

print("EME2000 to GCRF rotation matrix:")
print(
    f"  [{R_eme2000_to_gcrf[0, 0]:13.10f}, {R_eme2000_to_gcrf[0, 1]:13.10f}, {R_eme2000_to_gcrf[0, 2]:13.10f}]"
)
print(
    f"  [{R_eme2000_to_gcrf[1, 0]:13.10f}, {R_eme2000_to_gcrf[1, 1]:13.10f}, {R_eme2000_to_gcrf[1, 2]:13.10f}]"
)
print(
    f"  [{R_eme2000_to_gcrf[2, 0]:13.10f}, {R_eme2000_to_gcrf[2, 1]:13.10f}, {R_eme2000_to_gcrf[2, 2]:13.10f}]\n"
)
# EME2000 to GCRF rotation matrix:
#   [ 1.0000000000,  0.0000000708, -0.0000000806]
#   [-0.0000000708,  1.0000000000, -0.0000000331]
#   [ 0.0000000806,  0.0000000331,  1.0000000000]

# Verify matrix is orthonormal (rotation matrix property)
identity = R_eme2000_to_gcrf @ R_eme2000_to_gcrf.T
print("Verify orthonormality (R @ R^T should be identity):")
print(f"  Max deviation from identity: {np.max(np.abs(identity - np.eye(3))):.2e}\n")
# Verify orthonormality (R @ R^T should be identity):
#   Max deviation from identity: 4.68e-15

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

# Convert to EME2000 Cartesian state and extract position
state_eme2000 = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)
pos_eme2000 = state_eme2000[0:3]

print("Satellite position in EME2000:")
print(f"  [{pos_eme2000[0]:.3f}, {pos_eme2000[1]:.3f}, {pos_eme2000[2]:.3f}] m\n")
# Satellite position in EME2000:
#   [1848964.106, -434937.468, 6560410.530] m

# Transform using rotation matrix
pos_gcrf_matrix = R_eme2000_to_gcrf @ pos_eme2000

print("Satellite position in GCRF (using rotation matrix):")
print(
    f"  [{pos_gcrf_matrix[0]:.3f}, {pos_gcrf_matrix[1]:.3f}, {pos_gcrf_matrix[2]:.3f}] m"
)
# Satellite position in GCRF (using rotation matrix):
#   [1848963.547, -434937.816, 6560410.665] m

# Verify using position transformation function
pos_gcrf_direct = bh.position_eme2000_to_gcrf(pos_eme2000)
print("\nSatellite position in GCRF (using position_eme2000_to_gcrf):")
print(
    f"  [{pos_gcrf_direct[0]:.3f}, {pos_gcrf_direct[1]:.3f}, {pos_gcrf_direct[2]:.3f}] m"
)
# Satellite position in GCRF (using position_eme2000_to_gcrf):
#   [1848963.547, -434937.816, 6560410.665] m

# Verify both methods agree
diff = np.linalg.norm(pos_gcrf_matrix - pos_gcrf_direct)
print(f"\nDifference between methods: {diff:.6e} m")
print("\nNote: Frame bias is constant (same at all epochs)")
# Difference between methods: 0.000000e+00 m
