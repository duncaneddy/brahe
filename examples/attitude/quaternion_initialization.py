# /// script
# dependencies = ["brahe"]
# ///

"""
Demonstrates different ways to initialize quaternions.
"""

import math
import brahe as bh
import numpy as np


# Initialize from individual components (w, x, y, z)
# Always scalar-first in constructor
q1 = bh.Quaternion(0.924, 0.0, 0.0, 0.383)
print("From components (identity):")
print(f"  q = [{q1.w:.3f}, {q1.x:.3f}, {q1.y:.3f}, {q1.z:.3f}]")

# Initialize from vector/array [w, x, y, z]
# Can specify if scalar is first or last
q2 = bh.Quaternion.from_vector(np.array([0.924, 0.0, 0.0, 0.383]), scalar_first=True)
print("\nFrom vector:")
print(f"  q = [{q2.w:.3f}, {q2.x:.3f}, {q2.y:.3f}, {q2.z:.3f}]")

# Initialize from another representation (rotation matrix)
# 90° rotation about Z-axis
rm = bh.RotationMatrix.Rz(45, bh.AngleFormat.DEGREES)
q3 = bh.Quaternion.from_rotation_matrix(rm)
print("\nFrom rotation matrix (45° about Z-axis):")
print(f"  q = [{q3.w:.3f}, {q3.x:.3f}, {q3.y:.3f}, {q3.z:.3f}]")

# Initialize from Euler angles (ZYX sequence)
ea = bh.EulerAngle(
    bh.EulerAngleOrder.ZYX, math.pi / 4, 0.0, 0.0, bh.AngleFormat.RADIANS
)
q4 = bh.Quaternion.from_euler_angle(ea)
print("\nFrom Euler angles (45° yaw, ZYX):")
print(f"  q = [{q4.w:.3f}, {q4.x:.3f}, {q4.y:.3f}, {q4.z:.3f}]")

# Initialize from Euler axis (axis-angle representation)
axis = np.array([0.0, 0.0, 1.0])  # Z-axis
angle = math.pi / 4  # 45°
ea_rep = bh.EulerAxis(axis, angle, bh.AngleFormat.RADIANS)
q5 = bh.Quaternion.from_euler_axis(ea_rep)
print("\nFrom Euler axis (45° about Z-axis):")
print(f"  q = [{q5.w:.3f}, {q5.x:.3f}, {q5.y:.3f}, {q5.z:.3f}]")

# Expected output:
# From components (identity):
#   q = [0.924, 0.000, 0.000, 0.383]

# From vector:
#   q = [0.924, 0.000, 0.000, 0.383]

# From rotation matrix (45° about Z-axis):
#   q = [0.924, 0.000, 0.000, 0.383]

# From Euler angles (45° yaw, ZYX):
#   q = [0.924, 0.000, 0.000, 0.383]

# From Euler axis (45° about Z-axis):
#   q = [0.924, 0.000, 0.000, 0.383]
