# /// script
# dependencies = ["brahe"]
# ///

"""
Demonstrates different ways to initialize rotation matrices.
"""

import brahe as bh
import numpy as np
import math

# Initialize from 9 individual elements (row-major order)
# Identity rotation
rm_identity = bh.RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
print("Identity rotation matrix:")
print(f"  [{rm_identity.r11:.3f}, {rm_identity.r12:.3f}, {rm_identity.r13:.3f}]")
print(f"  [{rm_identity.r21:.3f}, {rm_identity.r22:.3f}, {rm_identity.r23:.3f}]")
print(f"  [{rm_identity.r31:.3f}, {rm_identity.r32:.3f}, {rm_identity.r33:.3f}]")

# Common rotation: 90° about X-axis
angle_x = 30
rm_x = bh.RotationMatrix.Rx(angle_x, bh.AngleFormat.DEGREES)
print(f"\n{angle_x}° rotation about X-axis:")
print(f"  [{rm_x.r11:.3f}, {rm_x.r12:.3f}, {rm_x.r13:.3f}]")
print(f"  [{rm_x.r21:.3f}, {rm_x.r22:.3f}, {rm_x.r23:.3f}]")
print(f"  [{rm_x.r31:.3f}, {rm_x.r32:.3f}, {rm_x.r33:.3f}]")

# Common rotation: 90° about Y-axis
angle_y = 60
rm_y = bh.RotationMatrix.Ry(angle_y, bh.AngleFormat.DEGREES)
print(f"\n{angle_y}° rotation about Y-axis:")
print(f"  [{rm_y.r11:.3f}, {rm_y.r12:.3f}, {rm_y.r13:.3f}]")
print(f"  [{rm_y.r21:.3f}, {rm_y.r22:.3f}, {rm_y.r23:.3f}]")
print(f"  [{rm_y.r31:.3f}, {rm_y.r32:.3f}, {rm_y.r33:.3f}]")

# Common rotation: 90° about Z-axis
angle_z = 45
rm_z = bh.RotationMatrix.Rz(angle_z, bh.AngleFormat.DEGREES)
print(f"\n{angle_z}° rotation about Z-axis:")
print(f"  [{rm_z.r11:.3f}, {rm_z.r12:.3f}, {rm_z.r13:.3f}]")
print(f"  [{rm_z.r21:.3f}, {rm_z.r22:.3f}, {rm_z.r23:.3f}]")
print(f"  [{rm_z.r31:.3f}, {rm_z.r32:.3f}, {rm_z.r33:.3f}]")

# Initialize from another representation (quaternion)
q = bh.Quaternion(
    math.cos(math.radians(angle_z) / 2), 0, 0, math.sin(math.radians(angle_z) / 2)
)  # 90° about Z-axis
rm_from_q = bh.RotationMatrix.from_quaternion(q)
print("\nFrom quaternion (90° about Z-axis):")
print(f"  [{rm_from_q.r11:.6f}, {rm_from_q.r12:.6f}, {rm_from_q.r13:.6f}]")
print(f"  [{rm_from_q.r21:.6f}, {rm_from_q.r22:.6f}, {rm_from_q.r23:.6f}]")
print(f"  [{rm_from_q.r31:.6f}, {rm_from_q.r32:.6f}, {rm_from_q.r33:.6f}]")

# Initialize from Euler angles (ZYX sequence)
euler_angles = bh.EulerAngle(
    bh.EulerAngleOrder.ZYX, angle_z, 0, 0, bh.AngleFormat.DEGREES
)
rm_from_euler = bh.RotationMatrix.from_euler_angle(euler_angles)
print(f"\nFrom Euler angles ({angle_z}° about Z-axis):")
print(f"  [{rm_from_euler.r11:.6f}, {rm_from_euler.r12:.6f}, {rm_from_euler.r13:.6f}]")
print(f"  [{rm_from_euler.r21:.6f}, {rm_from_euler.r22:.6f}, {rm_from_euler.r23:.6f}]")
print(f"  [{rm_from_euler.r31:.6f}, {rm_from_euler.r32:.6f}, {rm_from_euler.r33:.6f}]")

# Initialize from Euler axis and angle
axis = np.array([0, 0, 1])  # Z-axis
euler_axis = bh.EulerAxis(axis, angle_z, bh.AngleFormat.DEGREES)
rm_from_axis_angle = bh.RotationMatrix.from_euler_axis(euler_axis)
print(f"\nFrom Euler axis ({angle_z}° about Z-axis):")
print(
    f"  [{rm_from_axis_angle.r11:.6f}, {rm_from_axis_angle.r12:.6f}, {rm_from_axis_angle.r13:.6f}]"
)
print(
    f"  [{rm_from_axis_angle.r21:.6f}, {rm_from_axis_angle.r22:.6f}, {rm_from_axis_angle.r23:.6f}]"
)
print(
    f"  [{rm_from_axis_angle.r31:.6f}, {rm_from_axis_angle.r32:.6f}, {rm_from_axis_angle.r33:.6f}]"
)

# Expected output:
# Identity rotation matrix:
#   [1.000, 0.000, 0.000]
#   [0.000, 1.000, 0.000]
#   [0.000, 0.000, 1.000]
#
# 90° rotation about X-axis:
#   [1.000, 0.000, 0.000]
#   [0.000, 0.000, -1.000]
#   [0.000, 1.000, 0.000]
#
# 90° rotation about Y-axis:
#   [0.000, 0.000, 1.000]
#   [0.000, 1.000, 0.000]
#   [-1.000, 0.000, 0.000]
#
# 90° rotation about Z-axis:
#   [0.000, -1.000, 0.000]
#   [1.000, 0.000, 0.000]
#   [0.000, 0.000, 1.000]
#
# 45° rotation about Z-axis:
#   [0.707107, -0.707107, 0.000000]
#   [0.707107, 0.707107, 0.000000]
#   [0.000000, 0.000000, 1.000000]
#
# From quaternion (45° about Z-axis):
#   [0.707107, -0.707107, 0.000000]
#   [0.707107, 0.707107, 0.000000]
#   [0.000000, 0.000000, 1.000000]
