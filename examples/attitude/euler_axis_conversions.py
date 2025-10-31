# /// script
# dependencies = ["brahe"]
# ///

"""
Demonstrates converting Euler axis to other attitude representations.
"""

import brahe as bh
import numpy as np
import math

# Create an Euler axis (45° rotation about Z-axis)
ea = bh.EulerAxis(np.array([0.0, 0.0, 1.0]), math.radians(45.0), bh.AngleFormat.RADIANS)

print("Original Euler axis:")
print(f"  Axis: [{ea.axis[0]:.6f}, {ea.axis[1]:.6f}, {ea.axis[2]:.6f}]")
print(f"  Angle: {math.degrees(ea.angle):.1f}°")

# Convert to quaternion
q = ea.to_quaternion()
print("\nTo quaternion:")
print(f"  q = [{q.w:.6f}, {q.x:.6f}, {q.y:.6f}, {q.z:.6f}]")

# Convert to rotation matrix
rm = ea.to_rotation_matrix()
print("\nTo rotation matrix:")
print(f"  [{rm.r11:.6f}, {rm.r12:.6f}, {rm.r13:.6f}]")
print(f"  [{rm.r21:.6f}, {rm.r22:.6f}, {rm.r23:.6f}]")
print(f"  [{rm.r31:.6f}, {rm.r32:.6f}, {rm.r33:.6f}]")

# Convert to Euler angles (ZYX sequence)
ea_angles_zyx = ea.to_euler_angle(bh.EulerAngleOrder.ZYX)
print("\nTo Euler angles (ZYX):")
print(f"  Yaw (Z):   {math.degrees(ea_angles_zyx.phi):.3f}°")
print(f"  Pitch (Y): {math.degrees(ea_angles_zyx.theta):.3f}°")
print(f"  Roll (X):  {math.degrees(ea_angles_zyx.psi):.3f}°")

# Convert to Euler angles (XYZ sequence)
ea_angles_xyz = ea.to_euler_angle(bh.EulerAngleOrder.XYZ)
print("\nTo Euler angles (XYZ):")
print(f"  Angle 1 (X): {math.degrees(ea_angles_xyz.phi):.3f}°")
print(f"  Angle 2 (Y): {math.degrees(ea_angles_xyz.theta):.3f}°")
print(f"  Angle 3 (Z): {math.degrees(ea_angles_xyz.psi):.3f}°")

# Round-trip conversion test
q_roundtrip = ea.to_quaternion()
ea_roundtrip = bh.EulerAxis.from_quaternion(q_roundtrip)
print("\nRound-trip (EulerAxis → Quaternion → EulerAxis):")
print(
    f"  Axis: [{ea_roundtrip.axis[0]:.6f}, {ea_roundtrip.axis[1]:.6f}, {ea_roundtrip.axis[2]:.6f}]"
)
print(f"  Angle: {math.degrees(ea_roundtrip.angle):.1f}°")

# Expected output:
# Original Euler axis:
#   Axis: [0.000000, 0.000000, 1.000000]
#   Angle: 45.0°

# To quaternion:
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

# Round-trip (EulerAxis → Quaternion → EulerAxis):
#   Axis: [0.000000, 0.000000, 1.000000]
#   Angle: 45.0°
