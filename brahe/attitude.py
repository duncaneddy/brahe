"""
Attitude Module

Attitude representations and conversions.

This module provides multiple representations of spacecraft attitude/orientation:

**Quaternion:**
- Quaternion representation (4-parameter)
- Quaternion algebra operations
- Conversions to/from other representations

**Euler Angles:**
- Multiple Euler angle sequences (3-1-3, 3-2-1, etc.)
- Conversions to/from other representations

**Euler Axis:**
- Euler axis-angle representation (rotation vector)
- Conversions to/from other representations

**Rotation Matrix:**
- Direction Cosine Matrix (DCM) representation
- Matrix operations and compositions
- Conversions to/from other representations

All attitude representations can be converted between each other, providing
flexibility in how orientation is specified and computed.
"""

from brahe._brahe import (
    # Attitude representation classes
    Quaternion,
    EulerAxis,
    EulerAngle,
    EulerAngleOrder,
    RotationMatrix,
)

__all__ = [
    # Attitude representations
    "Quaternion",
    "EulerAxis",
    "EulerAngle",
    "EulerAngleOrder",
    "RotationMatrix",
]
