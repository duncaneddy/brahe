# /// script
# dependencies = ["brahe"]
# ///

"""
Demonstrates different ways to initialize Euler angles.
"""

import brahe as bh
import numpy as np
import math

# Initialize from individual angles with ZYX sequence (yaw-pitch-roll)
# 45° yaw, 30° pitch, 15° roll
ea_zyx = bh.EulerAngle(
    bh.EulerAngleOrder.ZYX,
    45.0,  # Yaw (Z)
    30.0,  # Pitch (Y)
    15.0,  # Roll (X)
    bh.AngleFormat.DEGREES,
)
print("ZYX Euler angles (yaw-pitch-roll):")
print(f"  Yaw (Z):   {math.degrees(ea_zyx.phi):.1f}°")
print(f"  Pitch (Y): {math.degrees(ea_zyx.theta):.1f}°")
print(f"  Roll (X):  {math.degrees(ea_zyx.psi):.1f}°")
print(f"  Order: {ea_zyx.order}")

# Initialize from vector with XYZ sequence
angles_vec = np.array([15.0, 30.0, 45.0])
ea_xyz = bh.EulerAngle.from_vector(
    angles_vec, bh.EulerAngleOrder.XYZ, bh.AngleFormat.DEGREES
)
print("\nXYZ Euler angles (from vector):")
print(f"  Angle 1 (X): {math.degrees(ea_xyz.phi):.1f}°")
print(f"  Angle 2 (Y): {math.degrees(ea_xyz.theta):.1f}°")
print(f"  Angle 3 (Z): {math.degrees(ea_xyz.psi):.1f}°")
print(f"  Order: {ea_xyz.order}")

# Simple rotation about single axis (45° about Z using ZYX)
ea_z_only = bh.EulerAngle(
    bh.EulerAngleOrder.ZYX,
    45.0,  # Z
    0.0,  # Y
    0.0,  # X
    bh.AngleFormat.DEGREES,
)
print("\nSingle-axis rotation (45° about Z using ZYX):")
print(f"  Yaw (Z):   {math.degrees(ea_z_only.phi):.1f}°")
print(f"  Pitch (Y): {math.degrees(ea_z_only.theta):.1f}°")
print(f"  Roll (X):  {math.degrees(ea_z_only.psi):.1f}°")

# Initialize from another representation (quaternion)
q = bh.Quaternion(math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8))
ea_from_q = bh.EulerAngle.from_quaternion(q, bh.EulerAngleOrder.ZYX)
print("\nFrom quaternion (45° about Z):")
print(f"  Yaw (Z):   {math.degrees(ea_from_q.phi):.1f}°")
print(f"  Pitch (Y): {math.degrees(ea_from_q.theta):.1f}°")
print(f"  Roll (X):  {math.degrees(ea_from_q.psi):.1f}°")

# Initialize from Rotation Matrix
rm = bh.RotationMatrix.Rz(45.0, bh.AngleFormat.DEGREES)
ea_from_rm = bh.EulerAngle.from_rotation_matrix(rm, bh.EulerAngleOrder.ZYX)
print("\nFrom rotation matrix (45° about Z):")
print(f"  Yaw (Z):   {math.degrees(ea_from_rm.phi):.1f}°")
print(f"  Pitch (Y): {math.degrees(ea_from_rm.theta):.1f}°")
print(f"  Roll (X):  {math.degrees(ea_from_rm.psi):.1f}°")

# Initialize from Euler Axis
euler_axis = bh.EulerAxis(np.array([0.0, 0.0, 1.0]), 45.0, bh.AngleFormat.DEGREES)
ea_from_ea = bh.EulerAngle.from_euler_axis(euler_axis, bh.EulerAngleOrder.ZYX)

print("\nFrom Euler axis (45° about Z):")
print(f"  Yaw (Z):   {math.degrees(ea_from_ea.phi):.1f}°")
print(f"  Pitch (Y): {math.degrees(ea_from_ea.theta):.1f}°")
print(f"  Roll (X):  {math.degrees(ea_from_ea.psi):.1f}°")

# Initialize from one EulerAngle to another with different order
# Start with XZY order
ea_xzy = bh.EulerAngle.from_euler_angle(ea_zyx, bh.EulerAngleOrder.XZY)
print("\nXZY Euler angles from ZYX:")
print(f"  Angle 1 (X): {math.degrees(ea_xzy.phi):.1f}°")
print(f"  Angle 2 (Z): {math.degrees(ea_xzy.theta):.1f}°")
print(f"  Angle 3 (Y): {math.degrees(ea_xzy.psi):.1f}°")
print(f"  Order: {ea_xzy.order}")

# Convert to ZYX order (same physical rotation, different representation)
# Go through quaternion as intermediate representation
q_xzy = ea_xzy.to_quaternion()
ea_zyx_converted = bh.EulerAngle.from_quaternion(q_xzy, bh.EulerAngleOrder.ZYX)
print("\nConverted back to ZYX order (same rotation):")
print(f"  Angle 1 (Z): {math.degrees(ea_zyx_converted.phi):.1f}°")
print(f"  Angle 2 (Y): {math.degrees(ea_zyx_converted.theta):.1f}°")
print(f"  Angle 3 (X): {math.degrees(ea_zyx_converted.psi):.1f}°")
print(f"  Order: {ea_zyx_converted.order}")

# Expected output:
# ZYX Euler angles (yaw-pitch-roll):
#   Yaw (Z):   45.0°
#   Pitch (Y): 30.0°
#   Roll (X):  15.0°
#   Order: ZYX
#
# XYZ Euler angles (from vector):
#   Angle 1 (X): 15.0°
#   Angle 2 (Y): 30.0°
#   Angle 3 (Z): 45.0°
#   Order: XYZ
#
# Single-axis rotation (45° about Z using ZYX):
#   Yaw (Z):   45.0°
#   Pitch (Y): 0.0°
#   Roll (X):  0.0°
#
# From quaternion (45° about Z):
#   Yaw (Z):   45.0°
#   Pitch (Y): 0.0°
#   Roll (X):  0.0°
#
# From rotation matrix (45° about Z):
#   Yaw (Z):   45.0°
#   Pitch (Y): 0.0°
#   Roll (X):  0.0°
#
# From Euler axis (45° about Z):
#   Yaw (Z):   45.0°
#   Pitch (Y): 0.0°
#   Roll (X):  -0.0°
#
# XZY Euler angles from ZYX:
#   Angle 1 (X): 20.8°
#   Angle 2 (Z): 50.8°
#   Angle 3 (Y): 14.5°
#   Order: XZY
#
# Converted back to ZYX order (same rotation):
#   Angle 1 (Z): 45.0°
#   Angle 2 (Y): 30.0°
#   Angle 3 (X): 15.0°
#   Order: ZYX
