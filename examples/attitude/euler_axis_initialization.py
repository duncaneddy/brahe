# /// script
# dependencies = ["brahe"]
# ///

"""
Demonstrates different ways to initialize Euler axis (axis-angle) representations.
"""

import brahe as bh
import numpy as np
import math

# Initialize from axis vector and angle
# 45° rotation about Z-axis
axis_z = np.array([0.0, 0.0, 1.0])
angle = math.radians(45.0)
ea_z = bh.EulerAxis(axis_z, angle, bh.AngleFormat.RADIANS)

print("45° rotation about Z-axis:")
print(f"  Axis: [{ea_z.axis[0]:.3f}, {ea_z.axis[1]:.3f}, {ea_z.axis[2]:.3f}]")
print(f"  Angle: {math.degrees(ea_z.angle):.1f}°")

# 90° rotation about X-axis
axis_x = np.array([1.0, 0.0, 0.0])
ea_x = bh.EulerAxis(axis_x, math.radians(90.0), bh.AngleFormat.RADIANS)

print("\n90° rotation about X-axis:")
print(f"  Axis: [{ea_x.axis[0]:.3f}, {ea_x.axis[1]:.3f}, {ea_x.axis[2]:.3f}]")
print(f"  Angle: {math.degrees(ea_x.angle):.1f}°")

# Initialize from another representation (quaternion)
q = bh.Quaternion(math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8))
ea_from_q = bh.EulerAxis.from_quaternion(q)

print("\nFrom quaternion (45° about Z):")
print(
    f"  Axis: [{ea_from_q.axis[0]:.6f}, {ea_from_q.axis[1]:.6f}, {ea_from_q.axis[2]:.6f}]"
)
print(f"  Angle: {math.degrees(ea_from_q.angle):.1f}°")

# Initialize from rotation matrix
rm = bh.RotationMatrix.Rz(45, bh.AngleFormat.DEGREES)
ea_from_rm = bh.EulerAxis.from_rotation_matrix(rm)

print("\nFrom rotation matrix (45° about Z):")
print(
    f"  Axis: [{ea_from_rm.axis[0]:.6f}, {ea_from_rm.axis[1]:.6f}, {ea_from_rm.axis[2]:.6f}]"
)
print(f"  Angle: {math.degrees(ea_from_rm.angle):.1f}°")

# Initialize from EulerAngle
euler_angle = bh.EulerAngle(
    bh.EulerAngleOrder.ZYX, 45.0, 0.0, 0.0, bh.AngleFormat.DEGREES
)
ea_from_euler = bh.EulerAxis.from_euler_angle(euler_angle)

print("\nFrom EulerAngle (45° about Z):")
print(
    f"  Axis: [{ea_from_euler.axis[0]:.6f}, {ea_from_euler.axis[1]:.6f}, {ea_from_euler.axis[2]:.6f}]"
)
print(f"  Angle: {math.degrees(ea_from_euler.angle):.1f}°")

# Expected output:
# 45° rotation about Z-axis:
#   Axis: [0.000, 0.000, 1.000]
#   Angle: 45.0°

# 90° rotation about X-axis:
#   Axis: [1.000, 0.000, 0.000]
#   Angle: 90.0°

# From quaternion (45° about Z):
#   Axis: [0.000000, 0.000000, 1.000000]
#   Angle: 45.0°

# From rotation matrix (45° about Z):
#   Axis: [0.000000, 0.000000, 1.000000]
#   Angle: 45.0°

# From EulerAngle (45° about Z):
#   Axis: [0.000000, 0.000000, 1.000000]
#   Angle: 45.0°
