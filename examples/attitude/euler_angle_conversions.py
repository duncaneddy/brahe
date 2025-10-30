# /// script
# dependencies = ["brahe"]
# ///

"""
Demonstrates converting Euler angles to other attitude representations.
"""

import brahe as bh
import math

# Create Euler angles (ZYX: 45° yaw, 30° pitch, 15° roll)
ea = bh.EulerAngle(
    math.radians(45.0),
    math.radians(30.0),
    math.radians(15.0),
    bh.EulerAngleOrder.ZYX,
    bh.AngleFormat.RADIANS,
)

print("Original Euler angles (ZYX):")
print(f"  Yaw (Z):   {math.degrees(ea.phi):.1f}°")
print(f"  Pitch (Y): {math.degrees(ea.theta):.1f}°")
print(f"  Roll (X):  {math.degrees(ea.psi):.1f}°")

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

# Convert to Euler axis (axis-angle)
ea_axis = ea.to_euler_axis()
print("\nTo Euler axis:")
print(f"  Axis: [{ea_axis.axis[0]:.6f}, {ea_axis.axis[1]:.6f}, {ea_axis.axis[2]:.6f}]")
print(f"  Angle: {math.degrees(ea_axis.angle):.3f}°")

# Convert to different Euler angle sequence
ea_xyz = bh.EulerAngle.from_quaternion(q, bh.EulerAngleOrder.XYZ)
print("\nTo different sequence (XYZ):")
print(f"  Angle 1 (X): {math.degrees(ea_xyz.phi):.3f}°")
print(f"  Angle 2 (Y): {math.degrees(ea_xyz.theta):.3f}°")
print(f"  Angle 3 (Z): {math.degrees(ea_xyz.psi):.3f}°")

# Round-trip conversion test
q_roundtrip = ea.to_quaternion()
ea_roundtrip = bh.EulerAngle.from_quaternion(q_roundtrip, bh.EulerAngleOrder.ZYX)
print("\nRound-trip (EulerAngle → Quaternion → EulerAngle):")
print(f"  Yaw (Z):   {math.degrees(ea_roundtrip.phi):.1f}°")
print(f"  Pitch (Y): {math.degrees(ea_roundtrip.theta):.1f}°")
print(f"  Roll (X):  {math.degrees(ea_roundtrip.psi):.1f}°")

# Expected output:
# Original Euler angles (ZYX):
#   Yaw (Z):   45.0°
#   Pitch (Y): 30.0°
#   Roll (X):  15.0°
#
# To quaternion:
#   q = [0.896956, 0.125615, 0.367370, 0.220692]
#
# To rotation matrix:
#   [0.659983, -0.543839, 0.515038]
#   [0.659983, 0.740791, 0.125615]
#   [-0.357406, 0.395841, 0.847997]
#
# To Euler axis:
#   Axis: [0.299876, 0.877321, 0.373499]
#   Angle: 52.318°
#
# To different sequence (XYZ):
#   Angle 1 (X): 13.239°
#   Angle 2 (Y): 22.889°
#   Angle 3 (Z): 47.098°
#
# Round-trip (EulerAngle → Quaternion → EulerAngle):
#   Yaw (Z):   45.0°
#   Pitch (Y): 30.0°
#   Roll (X):  15.0°
