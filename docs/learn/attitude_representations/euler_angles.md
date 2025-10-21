# Euler Angles

Euler angles represent rotations as three sequential rotations about coordinate axes.

## Overview

Euler angles describe orientation using three angles representing sequential rotations about specified axes. Brahe supports all 12 possible Euler angle sequences (e.g., XYZ, ZYX, ZYZ).

## Mathematical Representation

An Euler angle rotation is specified by:

- Three angles: $(\alpha, \beta, \gamma)$
- A rotation sequence (e.g., XYZ, ZYX)

## Common Sequences

- **ZYX (Yaw-Pitch-Roll)**: Common in aerospace applications
- **XYZ**: Common in robotics
- **ZYZ**: Common in classical mechanics

## Advantages

- **Intuitive**: Easy to visualize and understand
- **Minimal**: Only 3 parameters
- **Human-readable**: Natural for manual input

## Disadvantages

- **Gimbal lock**: Singularities occur when middle rotation is ±90°
- **Ambiguous**: Multiple angle sets can represent same orientation
- **Interpolation**: Non-linear, difficult to interpolate smoothly

## Gimbal Lock

Gimbal lock occurs when the middle rotation approaches ±90 degrees, causing loss of one degree of freedom. Use quaternions to avoid this issue.

## See Also

- [Euler Angles API Reference](../../library_api/attitude/euler_angles.md)
- [Attitude Representations Overview](index.md)
