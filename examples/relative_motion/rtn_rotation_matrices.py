# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Create RTN rotation matrices and verify their orthogonality properties
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define a satellite in LEO orbit
# 700 km altitude, nearly circular, sun-synchronous inclination
oe = np.array(
    [
        bh.R_EARTH + 700e3,  # Semi-major axis (m)
        0.001,  # Eccentricity
        97.8,  # Inclination (deg)
        15.0,  # Right ascension of ascending node (deg)
        30.0,  # Argument of perigee (deg)
        45.0,  # Mean anomaly (deg)
    ]
)

# Convert to Cartesian ECI state
x_eci = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Compute rotation matrices
R_rtn_to_eci = bh.rotation_rtn_to_eci(x_eci)
R_eci_to_rtn = bh.rotation_eci_to_rtn(x_eci)

print("RTN-to-ECI rotation matrix:")
print(
    f"  [{R_rtn_to_eci[0, 0]:8.5f}, {R_rtn_to_eci[0, 1]:8.5f}, {R_rtn_to_eci[0, 2]:8.5f}]"
)
print(
    f"  [{R_rtn_to_eci[1, 0]:8.5f}, {R_rtn_to_eci[1, 1]:8.5f}, {R_rtn_to_eci[1, 2]:8.5f}]"
)
print(
    f"  [{R_rtn_to_eci[2, 0]:8.5f}, {R_rtn_to_eci[2, 1]:8.5f}, {R_rtn_to_eci[2, 2]:8.5f}]\n"
)
#   [ 0.28262, -0.92432,  0.25642]
#   [-0.06004, -0.28384, -0.95699]
#   [ 0.95735,  0.25507, -0.13572]

# Verify orthogonality: R^T × R = I
identity = R_rtn_to_eci.T @ R_rtn_to_eci
print("Orthogonality check (R^T × R):")
print(f"  [{identity[0, 0]:8.5f}, {identity[0, 1]:8.5f}, {identity[0, 2]:8.5f}]")
print(f"  [{identity[1, 0]:8.5f}, {identity[1, 1]:8.5f}, {identity[1, 2]:8.5f}]")
print(f"  [{identity[2, 0]:8.5f}, {identity[2, 1]:8.5f}, {identity[2, 2]:8.5f}]")
print(f"Difference from identity: {np.linalg.norm(identity - np.eye(3)):.15f}\n")
#   [ 1.00000,  0.00000,  0.00000]
#   [ 0.00000,  1.00000,  0.00000]
#   [ 0.00000,  0.00000,  1.00000]
# Difference from identity: 0.000000000000000

# Verify determinant = +1 (proper rotation matrix)
det = np.linalg.det(R_rtn_to_eci)
print(f"Determinant (should be +1): {det:.15f}\n")
# Determinant (should be +1): 1.000000000000000

# Verify ECI-to-RTN is the transpose of RTN-to-ECI
print("Transpose relationship check:")
print(
    f"||R_eci_to_rtn - R_rtn_to_eci^T||: {np.linalg.norm(R_eci_to_rtn - R_rtn_to_eci.T):.15f}\n"
)
# ||R_eci_to_rtn - R_rtn_to_eci^T||: 0.000000000000000

# Example: Transform a vector from RTN to ECI
v_rtn = np.array([1.0, 0.0, 0.0])  # Radial unit vector in RTN frame
v_eci = R_rtn_to_eci @ v_rtn

print("Example transformation:")
print(f"Vector in RTN frame: [{v_rtn[0]:.3f}, {v_rtn[1]:.3f}, {v_rtn[2]:.3f}]")
print(f"Vector in ECI frame: [{v_eci[0]:.5f}, {v_eci[1]:.5f}, {v_eci[2]:.5f}]")
print(f"ECI vector magnitude: {np.linalg.norm(v_eci):.15f}")
# Vector in RTN frame: [1.000, 0.000, 0.000]
# Vector in ECI frame: [0.28262, -0.06004, 0.95735]
# ECI vector magnitude: 1.000000000000000
