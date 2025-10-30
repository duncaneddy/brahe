# /// script
# dependencies = ["brahe"]
# ///

"""
Demonstrates how to access and output Euler angle components.
"""

import brahe as bh
import numpy as np
import math

# Create Euler angles (ZYX: 45° yaw, 30° pitch, 15° roll)
ea = bh.EulerAngle(
    math.radians(45.0),
    math.radians(30.0),
    math.radians(15.0),
    bh.EulerAngleOrder.ZYX,
    bh.AngleFormat.RADIANS,
)

# Access individual angles
print("Individual angles (radians):")
print(f"  angle1 (Yaw/Z):   {ea.phi:.6f}")
print(f"  angle2 (Pitch/Y): {ea.theta:.6f}")
print(f"  angle3 (Roll/X):  {ea.psi:.6f}")

# Convert to degrees for readability
print("\nIndividual angles (degrees):")
print(f"  angle1 (Yaw/Z):   {math.degrees(ea.phi):.3f}°")
print(f"  angle2 (Pitch/Y): {math.degrees(ea.theta):.3f}°")
print(f"  angle3 (Roll/X):  {math.degrees(ea.psi):.3f}°")

# Access sequence order
print(f"\nSequence order: {ea.order}")

# Create vector from components [phi, theta, psi]
vec = np.array([ea.phi, ea.theta, ea.psi])
print("\nAs vector [phi, theta, psi] (radians):")
print(f"  {vec}")

# String representation
print("\nString representation:")
print(f"  {ea}")

# Expected output:
# Individual angles (radians):
#   angle1 (Yaw/Z):   0.785398
#   angle2 (Pitch/Y): 0.523599
#   angle3 (Roll/X):  0.261799
#
# Individual angles (degrees):
#   angle1 (Yaw/Z):   45.000°
#   angle2 (Pitch/Y): 30.000°
#   angle3 (Roll/X):  15.000°
#
# Sequence order: ZYX
#
# As vector [phi, theta, psi] (radians):
#   [0.78539816 0.52359878 0.26179939]
#
# String representation:
#   EulerAngle { angle1: 0.7853981633974483, angle2: 0.5235987755982988, angle3: 0.2617993877991494, order: ZYX }
