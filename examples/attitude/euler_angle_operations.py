# /// script
# dependencies = ["brahe"]
# ///

"""
Demonstrates Euler angle operations and conversions.

Note: Euler angles don't have operations like addition or composition directly.
To compose rotations, convert to quaternions or rotation matrices first.
"""

import brahe as bh
import math

# Create two Euler angle rotations (ZYX sequence)
ea1 = bh.EulerAngle(
    math.radians(45.0),  # Yaw
    0.0,  # Pitch
    0.0,  # Roll
    bh.EulerAngleOrder.ZYX,
    bh.AngleFormat.RADIANS,
)

ea2 = bh.EulerAngle(
    0.0,  # Yaw
    math.radians(30.0),  # Pitch
    0.0,  # Roll
    bh.EulerAngleOrder.ZYX,
    bh.AngleFormat.RADIANS,
)

print("First rotation (45° yaw):")
print(
    f"  Yaw: {math.degrees(ea1.phi):.1f}°, Pitch: {math.degrees(ea1.theta):.1f}°, Roll: {math.degrees(ea1.psi):.1f}°"
)

print("\nSecond rotation (30° pitch):")
print(
    f"  Yaw: {math.degrees(ea2.phi):.1f}°, Pitch: {math.degrees(ea2.theta):.1f}°, Roll: {math.degrees(ea2.psi):.1f}°"
)

# Compose rotations by converting to quaternions
q1 = ea1.to_quaternion()
q2 = ea2.to_quaternion()
q_composed = q2 * q1  # Apply ea1 first, then ea2

# Convert composed rotation back to Euler angles
ea_composed = bh.EulerAngle.from_quaternion(q_composed, bh.EulerAngleOrder.ZYX)

print("\nComposed rotation (via quaternions):")
print(f"  Yaw: {math.degrees(ea_composed.phi):.3f}°")
print(f"  Pitch: {math.degrees(ea_composed.theta):.3f}°")
print(f"  Roll: {math.degrees(ea_composed.psi):.3f}°")

# Demonstrate sequence order matters
# Same angles, different sequence
ea_zyx = bh.EulerAngle(
    math.radians(30.0),
    math.radians(20.0),
    math.radians(10.0),
    bh.EulerAngleOrder.ZYX,
    bh.AngleFormat.RADIANS,
)
ea_xyz = bh.EulerAngle(
    math.radians(30.0),
    math.radians(20.0),
    math.radians(10.0),
    bh.EulerAngleOrder.XYZ,
    bh.AngleFormat.RADIANS,
)

q_zyx = ea_zyx.to_quaternion()
q_xyz = ea_xyz.to_quaternion()

print("\nSame angles, different sequences:")
print(f"  ZYX quaternion: [{q_zyx.w:.6f}, {q_zyx.x:.6f}, {q_zyx.y:.6f}, {q_zyx.z:.6f}]")
print(f"  XYZ quaternion: [{q_xyz.w:.6f}, {q_xyz.x:.6f}, {q_xyz.y:.6f}, {q_xyz.z:.6f}]")
print("  → Different quaternions show sequence order matters!")

# Expected output:
# First rotation (45° yaw):
#   Yaw: 45.0°, Pitch: 0.0°, Roll: 0.0°
#
# Second rotation (30° pitch):
#   Yaw: 0.0°, Pitch: 30.0°, Roll: 0.0°
#
# Composed rotation (via quaternions):
#   Yaw: 40.893°
#   Pitch: 30.000°
#   Roll: -10.893°
#
# Same angles, different sequences:
#   ZYX quaternion: [0.936117, 0.086824, 0.278559, 0.189199]
#   XYZ quaternion: [0.936117, 0.189199, 0.278559, 0.086824]
#   → Different quaternions show sequence order matters!
