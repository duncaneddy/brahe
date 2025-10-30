# /// script
# dependencies = ["brahe"]
# ///

"""
Demonstrates converting rotation matrices to other attitude representations.
"""

import brahe as bh
import math

# Create a rotation matrix (45° about Z-axis)
cos45 = math.cos(math.pi / 4)
sin45 = math.sin(math.pi / 4)
rm = bh.RotationMatrix(cos45, -sin45, 0.0, sin45, cos45, 0.0, 0.0, 0.0, 1.0)

print("Original rotation matrix:")
print(f"  [{rm.r11:.6f}, {rm.r12:.6f}, {rm.r13:.6f}]")
print(f"  [{rm.r21:.6f}, {rm.r22:.6f}, {rm.r23:.6f}]")
print(f"  [{rm.r31:.6f}, {rm.r32:.6f}, {rm.r33:.6f}]")

# Convert to quaternion
q = rm.to_quaternion()
print("\nTo quaternion:")
print(f"  q = [{q.w:.6f}, {q.x:.6f}, {q.y:.6f}, {q.z:.6f}]")

# Convert to Euler angles (ZYX sequence)
ea_zyx = rm.to_euler_angle(bh.EulerAngleOrder.ZYX)
print("\nTo Euler angles (ZYX):")
print(f"  Yaw (Z):   {math.degrees(ea_zyx.phi):.3f}°")
print(f"  Pitch (Y): {math.degrees(ea_zyx.theta):.3f}°")
print(f"  Roll (X):  {math.degrees(ea_zyx.psi):.3f}°")

# Convert to Euler angles (XYZ sequence)
ea_xyz = rm.to_euler_angle(bh.EulerAngleOrder.XYZ)
print("\nTo Euler angles (XYZ):")
print(f"  Angle 1 (X): {math.degrees(ea_xyz.phi):.3f}°")
print(f"  Angle 2 (Y): {math.degrees(ea_xyz.theta):.3f}°")
print(f"  Angle 3 (Z): {math.degrees(ea_xyz.psi):.3f}°")

# Convert to Euler axis (axis-angle)
ea = rm.to_euler_axis()
print("\nTo Euler axis:")
print(f"  Axis: [{ea.axis[0]:.6f}, {ea.axis[1]:.6f}, {ea.axis[2]:.6f}]")
print(f"  Angle: {math.degrees(ea.angle):.3f}°")

# Round-trip conversion test
rm_roundtrip = bh.RotationMatrix.from_quaternion(q)
print("\nRound-trip (RotationMatrix → Quaternion → RotationMatrix):")
print(f"  [{rm_roundtrip.r11:.6f}, {rm_roundtrip.r12:.6f}, {rm_roundtrip.r13:.6f}]")
print(f"  [{rm_roundtrip.r21:.6f}, {rm_roundtrip.r22:.6f}, {rm_roundtrip.r23:.6f}]")
print(f"  [{rm_roundtrip.r31:.6f}, {rm_roundtrip.r32:.6f}, {rm_roundtrip.r33:.6f}]")

# Expected output:
# Original rotation matrix:
#   [0.707107, -0.707107, 0.000000]
#   [0.707107, 0.707107, 0.000000]
#   [0.000000, 0.000000, 1.000000]
#
# To quaternion:
#   q = [0.923880, 0.000000, 0.000000, 0.382683]
#
# To Euler angles (ZYX):
#   Yaw (Z):   45.000°
#   Pitch (Y): 0.000°
#   Roll (X):  0.000°
#
# To Euler angles (XYZ):
#   Angle 1 (X): 0.000°
#   Angle 2 (Y): 0.000°
#   Angle 3 (Z): 45.000°
#
# To Euler axis:
#   Axis: [0.000000, 0.000000, 1.000000]
#   Angle: 45.000°
#
# Round-trip (RotationMatrix → Quaternion → RotationMatrix):
#   [0.707107, -0.707107, 0.000000]
#   [0.707107, 0.707107, 0.000000]
#   [0.000000, 0.000000, 1.000000]
