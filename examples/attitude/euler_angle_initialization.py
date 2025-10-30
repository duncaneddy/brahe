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
    math.radians(45.0),  # Yaw (Z)
    math.radians(30.0),  # Pitch (Y)
    math.radians(15.0),  # Roll (X)
    bh.EulerAngleOrder.ZYX,
    bh.AngleFormat.RADIANS,
)
print("ZYX Euler angles (yaw-pitch-roll):")
print(f"  Yaw (Z):   {math.degrees(ea_zyx.phi):.1f}°")
print(f"  Pitch (Y): {math.degrees(ea_zyx.theta):.1f}°")
print(f"  Roll (X):  {math.degrees(ea_zyx.psi):.1f}°")
print(f"  Order: {ea_zyx.order}")

# Initialize from vector with XYZ sequence
angles_vec = np.array([math.radians(15.0), math.radians(30.0), math.radians(45.0)])
ea_xyz = bh.EulerAngle.from_vector(
    angles_vec, bh.EulerAngleOrder.XYZ, bh.AngleFormat.RADIANS
)
print("\nXYZ Euler angles (from vector):")
print(f"  Angle 1 (X): {math.degrees(ea_xyz.phi):.1f}°")
print(f"  Angle 2 (Y): {math.degrees(ea_xyz.theta):.1f}°")
print(f"  Angle 3 (Z): {math.degrees(ea_xyz.psi):.1f}°")
print(f"  Order: {ea_xyz.order}")

# Simple rotation about single axis (45° about Z using ZYX)
ea_z_only = bh.EulerAngle(
    math.radians(45.0),  # Z
    0.0,  # Y
    0.0,  # X
    bh.EulerAngleOrder.ZYX,
    bh.AngleFormat.RADIANS,
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

# Different sequence: XZY
ea_xzy = bh.EulerAngle(
    math.radians(30.0),  # X
    math.radians(20.0),  # Z
    math.radians(10.0),  # Y
    bh.EulerAngleOrder.XZY,
    bh.AngleFormat.RADIANS,
)
print("\nXZY Euler angles:")
print(f"  Angle 1 (X): {math.degrees(ea_xzy.phi):.1f}°")
print(f"  Angle 2 (Z): {math.degrees(ea_xzy.theta):.1f}°")
print(f"  Angle 3 (Y): {math.degrees(ea_xzy.psi):.1f}°")
print(f"  Order: {ea_xzy.order}")

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
# XZY Euler angles:
#   Angle 1 (X): 30.0°
#   Angle 2 (Z): 20.0°
#   Angle 3 (Y): 10.0°
#   Order: XZY
