# /// script
# dependencies = ["brahe"]
# ///

"""
Demonstrates rotation matrix operations including matrix multiplication
and vector transformations.
"""

import brahe as bh
import numpy as np

# Create two rotation matrices
# 90° rotation about X-axis
rm_x = bh.RotationMatrix(1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)

# 90° rotation about Z-axis
rm_z = bh.RotationMatrix(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)

print("Rotation matrix X (90° about X):")
print(f"  [{rm_x.r11:.3f}, {rm_x.r12:.3f}, {rm_x.r13:.3f}]")
print(f"  [{rm_x.r21:.3f}, {rm_x.r22:.3f}, {rm_x.r23:.3f}]")
print(f"  [{rm_x.r31:.3f}, {rm_x.r32:.3f}, {rm_x.r33:.3f}]")

print("\nRotation matrix Z (90° about Z):")
print(f"  [{rm_z.r11:.3f}, {rm_z.r12:.3f}, {rm_z.r13:.3f}]")
print(f"  [{rm_z.r21:.3f}, {rm_z.r22:.3f}, {rm_z.r23:.3f}]")
print(f"  [{rm_z.r31:.3f}, {rm_z.r32:.3f}, {rm_z.r33:.3f}]")

# Matrix multiplication (compose rotations)
# Apply rm_x first, then rm_z
rm_composed = rm_z * rm_x
print("\nComposed rotation (X then Z):")
print(f"  [{rm_composed.r11:.3f}, {rm_composed.r12:.3f}, {rm_composed.r13:.3f}]")
print(f"  [{rm_composed.r21:.3f}, {rm_composed.r22:.3f}, {rm_composed.r23:.3f}]")
print(f"  [{rm_composed.r31:.3f}, {rm_composed.r32:.3f}, {rm_composed.r33:.3f}]")

# Transform a vector using rotation matrix
# Rotate vector [1, 0, 0] by 90° about Z-axis using matrix multiplication
R_z = rm_z.to_matrix()  # Get 3x3 numpy array
vector = np.array([1.0, 0.0, 0.0])
rotated = R_z @ vector  # Matrix-vector multiplication
print("\nVector transformation:")
print(f"  Original: [{vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f}]")
print(f"  Rotated:  [{rotated[0]:.3f}, {rotated[1]:.3f}, {rotated[2]:.3f}]")

# Transform another vector
vector2 = np.array([0.0, 1.0, 0.0])
rotated2 = R_z @ vector2
print(f"\n  Original: [{vector2[0]:.3f}, {vector2[1]:.3f}, {vector2[2]:.3f}]")
print(f"  Rotated:  [{rotated2[0]:.3f}, {rotated2[1]:.3f}, {rotated2[2]:.3f}]")

# Expected output:
# Rotation matrix X (90° about X):
#   [1.000, 0.000, 0.000]
#   [0.000, 0.000, -1.000]
#   [0.000, 1.000, 0.000]
#
# Rotation matrix Z (90° about Z):
#   [0.000, -1.000, 0.000]
#   [1.000, 0.000, 0.000]
#   [0.000, 0.000, 1.000]
#
# Composed rotation (X then Z):
#   [0.000, 0.000, 1.000]
#   [1.000, 0.000, 0.000]
#   [0.000, 1.000, 0.000]
#
# Vector transformation:
#   Original: [1.000, 0.000, 0.000]
#   Rotated:  [0.000, 1.000, 0.000]
#
#   Original: [0.000, 1.000, 0.000]
#   Rotated:  [-0.000, 0.000, 1.000]
