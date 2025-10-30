# /// script
# dependencies = ["brahe"]
# ///

"""
Demonstrates how to access and output Euler axis components.
"""

import brahe as bh
import numpy as np
import math

# Create an Euler axis (45° rotation about Z-axis)
axis = np.array([0.0, 0.0, 1.0])
angle = math.radians(45.0)
ea = bh.EulerAxis(axis, angle, bh.AngleFormat.RADIANS)

# Access individual components
print("Individual components:")
print(f"  Axis vector: [{ea.axis[0]:.6f}, {ea.axis[1]:.6f}, {ea.axis[2]:.6f}]")
print(f"  Angle (radians): {ea.angle:.6f}")
print(f"  Angle (degrees): {math.degrees(ea.angle):.3f}°")

# Verify axis is unit vector
axis_magnitude = np.linalg.norm(ea.axis)
print(f"\nAxis magnitude: {axis_magnitude:.6f}")

# Convert to 4-element vector [x, y, z, angle]
rot_vec = ea.to_vector(
    bh.AngleFormat.RADIANS, True
)  # vector_first=True means [x,y,z,angle]
print("\nAs 4-element vector [x, y, z, angle]:")
print(f"  [{rot_vec[0]:.6f}, {rot_vec[1]:.6f}, {rot_vec[2]:.6f}, {rot_vec[3]:.6f}]")
print(f"  Angle: {rot_vec[3]:.6f} rad = {math.degrees(rot_vec[3]):.3f}°")

# String representation
print("\nString representation:")
print(f"  {ea}")

# Example with different axis
axis2 = np.array([1.0, 1.0, 1.0])  # Will be normalized
ea2 = bh.EulerAxis(axis2, math.radians(120.0), bh.AngleFormat.RADIANS)

print("\n\n120° rotation about [1, 1, 1] axis:")
print(f"  Normalized axis: [{ea2.axis[0]:.6f}, {ea2.axis[1]:.6f}, {ea2.axis[2]:.6f}]")
print(f"  Angle: {math.degrees(ea2.angle):.1f}°")

# Expected output:
# Individual components:
#   Axis vector: [0.000000, 0.000000, 1.000000]
#   Angle (radians): 0.785398
#   Angle (degrees): 45.000°
#
# Axis magnitude: 1.000000
#
# As 4-element vector [x, y, z, angle]:
#   [0.000000, 0.000000, 1.000000, 0.785398]
#   Angle: 0.785398 rad = 45.000°
#
# String representation:
#   EulerAxis { axis: [0.0, 0.0, 1.0], angle: 0.7853981633974483 }
#
#
# 120° rotation about [1, 1, 1] axis:
#   Normalized axis: [0.577350, 0.577350, 0.577350]
#   Angle: 120.0°
