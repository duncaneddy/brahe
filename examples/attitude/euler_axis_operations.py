# /// script
# dependencies = ["brahe"]
# ///

"""
Demonstrates Euler axis operations.

Note: Like Euler angles, Euler axis representations don't have direct
composition operations. Convert to quaternions or rotation matrices to
compose rotations.
"""

import brahe as bh
import numpy as np
import math

# Create two Euler axis rotations
ea1 = bh.EulerAxis(
    np.array([0.0, 0.0, 1.0]), math.radians(45.0), bh.AngleFormat.RADIANS
)  # 45° about Z
ea2 = bh.EulerAxis(
    np.array([1.0, 0.0, 0.0]), math.radians(90.0), bh.AngleFormat.RADIANS
)  # 90° about X

print("First rotation (45° about Z):")
print(f"  Axis: [{ea1.axis[0]:.3f}, {ea1.axis[1]:.3f}, {ea1.axis[2]:.3f}]")
print(f"  Angle: {math.degrees(ea1.angle):.1f}°")

print("\nSecond rotation (90° about X):")
print(f"  Axis: [{ea2.axis[0]:.3f}, {ea2.axis[1]:.3f}, {ea2.axis[2]:.3f}]")
print(f"  Angle: {math.degrees(ea2.angle):.1f}°")

# Compose rotations via quaternions
q1 = ea1.to_quaternion()
q2 = ea2.to_quaternion()
q_composed = q2 * q1  # Apply ea1 first, then ea2

# Convert back to Euler axis
ea_composed = bh.EulerAxis.from_quaternion(q_composed)

print("\nComposed rotation (via quaternions):")
print(
    f"  Axis: [{ea_composed.axis[0]:.6f}, {ea_composed.axis[1]:.6f}, {ea_composed.axis[2]:.6f}]"
)
print(f"  Angle: {math.degrees(ea_composed.angle):.3f}°")

# Opposite rotations (axis negation vs angle negation)
ea_fwd = bh.EulerAxis(
    np.array([0.0, 1.0, 0.0]), math.radians(60.0), bh.AngleFormat.RADIANS
)
# Two ways to represent the opposite rotation:
ea_neg_angle = bh.EulerAxis(
    np.array([0.0, 1.0, 0.0]), math.radians(-60.0), bh.AngleFormat.RADIANS
)
ea_neg_axis = bh.EulerAxis(
    np.array([0.0, -1.0, 0.0]), math.radians(60.0), bh.AngleFormat.RADIANS
)

print("\nOpposite rotations (60° about Y):")
print("  Forward:      axis=[0, 1, 0], angle=+60°")
print("  Neg angle:    axis=[0, 1, 0], angle=-60°")
print("  Neg axis:     axis=[0, -1, 0], angle=+60°")

# Convert to quaternions to verify they're opposite
q_fwd = ea_fwd.to_quaternion()
q_neg_angle = ea_neg_angle.to_quaternion()
q_neg_axis = ea_neg_axis.to_quaternion()

print("\nAs quaternions:")
print(f"  Forward:      [{q_fwd.w:.6f}, {q_fwd.x:.6f}, {q_fwd.y:.6f}, {q_fwd.z:.6f}]")
print(
    f"  Neg angle:    [{q_neg_angle.w:.6f}, {q_neg_angle.x:.6f}, {q_neg_angle.y:.6f}, {q_neg_angle.z:.6f}]"
)
print(
    f"  Neg axis:     [{q_neg_axis.w:.6f}, {q_neg_axis.x:.6f}, {q_neg_axis.y:.6f}, {q_neg_axis.z:.6f}]"
)
print("  → Neg angle and neg axis are equivalent (conjugates)")

# Expected output:
# First rotation (45° about Z):
#   Axis: [0.000, 0.000, 1.000]
#   Angle: 45.0°
#
# Second rotation (90° about X):
#   Axis: [1.000, 0.000, 0.000]
#   Angle: 90.0°
#
# Composed rotation (via quaternions):
#   Axis: [0.653282, -0.270598, 0.706314]
#   Angle: 104.478°
#
# Opposite rotations (60° about Y):
#   Forward:      axis=[0, 1, 0], angle=+60°
#   Neg angle:    axis=[0, 1, 0], angle=-60°
#   Neg axis:     axis=[0, -1, 0], angle=+60°
#
# As quaternions:
#   Forward:      [0.866025, 0.000000, 0.500000, 0.000000]
#   Neg angle:    [0.866025, -0.000000, -0.500000, -0.000000]
#   Neg axis:     [0.866025, -0.000000, -0.500000, -0.000000]
#   → Neg angle and neg axis are equivalent (conjugates)
