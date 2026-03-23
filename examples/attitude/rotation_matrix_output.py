# /// script
# dependencies = ["brahe"]
# ///

"""
Demonstrates how to access and output rotation matrix elements.
"""

import brahe as bh

# Create a rotation matrix (45° about Z-axis)
rm = bh.RotationMatrix.Rz(45, bh.AngleFormat.DEGREES)

# Access individual elements
print("Individual elements (row-by-row):")
print(f"  r11: {rm.r11:.6f}, r12: {rm.r12:.6f}, r13: {rm.r13:.6f}")
print(f"  r21: {rm.r21:.6f}, r22: {rm.r22:.6f}, r23: {rm.r23:.6f}")
print(f"  r31: {rm.r31:.6f}, r32: {rm.r32:.6f}, r33: {rm.r33:.6f}")
# String representation
print("\nString representation:")
print(f"  {rm}")
