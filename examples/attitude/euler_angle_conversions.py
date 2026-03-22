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
    bh.EulerAngleOrder.ZYX,
    math.radians(45.0),
    math.radians(30.0),
    math.radians(15.0),
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
