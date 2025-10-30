# /// script
# dependencies = ["brahe"]
# ///

"""
Demonstrates common quaternion operations.
"""

import brahe as bh
import math

# Create a quaternion from rotation matrix (90° about X, then 45° about Z)
q = bh.Quaternion.from_rotation_matrix(
    bh.RotationMatrix.Rx(90, bh.AngleFormat.DEGREES)
    * bh.RotationMatrix.Rz(45, bh.AngleFormat.DEGREES)
)

print("Original quaternion:")
print(f"  q = [{q.w:.6f}, {q.x:.6f}, {q.y:.6f}, {q.z:.6f}]")

# Compute norm
norm = q.norm()
print(f"\nNorm: {norm:.6f}")

# Normalize quaternion (in-place)
q.normalize()  # In-place normalization (This shouldn't really do anything here since q already applies normalization on creation)
print("After normalization:")
print(f"  q = [{q.w:.6f}, {q.x:.6f}, {q.y:.6f}, {q.z:.6f}]")
print(f"  Norm: {q.norm():.6f}")

# Compute conjugate
q_conj = q.conjugate()
print("\nConjugate:")
print(f"  q* = [{q_conj.w:.6f}, {q_conj.x:.6f}, {q_conj.y:.6f}, {q_conj.z:.6f}]")

# Compute inverse (same as conjugate for normalized quaternions)
q_inv = q.inverse()
print("\nInverse:")
print(f"  q^-1 = [{q_inv.w:.6f}, {q_inv.x:.6f}, {q_inv.y:.6f}, {q_inv.z:.6f}]")

# Quaternion multiplication (compose rotations)
# 90° about X, then 90° about Z
q_x = bh.Quaternion(math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
q_z = bh.Quaternion(math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
q_composed = q_z * q_x  # Apply q_x first, then q_z
print("\nComposed rotation (90° X then 90° Z):")
print(f"  q_x = [{q_x.w:.6f}, {q_x.x:.6f}, {q_x.y:.6f}, {q_x.z:.6f}]")
print(f"  q_z = [{q_z.w:.6f}, {q_z.x:.6f}, {q_z.y:.6f}, {q_z.z:.6f}]")
print(
    f"  q_composed = [{q_composed.w:.6f}, {q_composed.x:.6f}, {q_composed.y:.6f}, {q_composed.z:.6f}]"
)

# Multiply q and its inverse to verify identity
identity = q * q_inv
print("\nq * q^-1 (should be identity):")
print(
    f"  q_identity = [{identity.w:.6f}, {identity.x:.6f}, {identity.y:.6f}, {identity.z:.6f}]"
)

# SLERP (Spherical Linear Interpolation) between two quaternions
# Interpolate from q_x (90° about X) to q_z (90° about Z)
print("\nSLERP interpolation from q_x to q_z:")
q_slerp_0 = q_x.slerp(q_z, 0.0)  # t=0, should equal q_x
print(
    f"  t=0.0: [{q_slerp_0.w:.6f}, {q_slerp_0.x:.6f}, {q_slerp_0.y:.6f}, {q_slerp_0.z:.6f}]"
)
q_slerp_25 = q_x.slerp(q_z, 0.25)
print(
    f"  t=0.25: [{q_slerp_25.w:.6f}, {q_slerp_25.x:.6f}, {q_slerp_25.y:.6f}, {q_slerp_25.z:.6f}]"
)
q_slerp_5 = q_x.slerp(q_z, 0.5)  # t=0.5, halfway
print(
    f"  t=0.5: [{q_slerp_5.w:.6f}, {q_slerp_5.x:.6f}, {q_slerp_5.y:.6f}, {q_slerp_5.z:.6f}]"
)
q_slerp_75 = q_x.slerp(q_z, 0.75)
print(
    f"  t=0.75: [{q_slerp_75.w:.6f}, {q_slerp_75.x:.6f}, {q_slerp_75.y:.6f}, {q_slerp_75.z:.6f}]"
)
q_slerp_1 = q_x.slerp(q_z, 1.0)  # t=1, should equal q_z
print(
    f"  t=1.0: [{q_slerp_1.w:.6f}, {q_slerp_1.x:.6f}, {q_slerp_1.y:.6f}, {q_slerp_1.z:.6f}]"
)

# Expected output:
# Original quaternion:
#   q = [0.923880, 0.000000, 0.000000, 0.382683]

# To rotation matrix:
#   [0.707107, 0.707107, 0.000000]
#   [-0.707107, 0.707107, 0.000000]
#   [0.000000, 0.000000, 1.000000]

# To Euler angles (ZYX):
#   Yaw (Z):   45.000°
#   Pitch (Y): 0.000°
#   Roll (X):  -0.000°

# To Euler angles (XYZ):
#   Angle 1 (X): 0.000°
#   Angle 2 (Y): -0.000°
#   Angle 3 (Z): 45.000°

# To Euler axis:
#   Axis: [0.000000, 0.000000, 1.000000]
#   Angle: 45.000°

# Round-trip (Quaternion → RotationMatrix → Quaternion):
#   q_rt = [0.923880, 0.000000, 0.000000, 0.382683]
