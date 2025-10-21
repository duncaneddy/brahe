# Attitude Representations

Brahe supports multiple mathematical representations for 3D rotations and spacecraft attitude.

## Overview

Attitude representation is fundamental to spacecraft dynamics and control. Brahe provides four different representations, each with their own advantages:

- **[Quaternions](quaternions.md)**: Singularity-free, compact representation (4 parameters)
- **[Rotation Matrices](rotation_matrices.md)**: Direct transformation matrices (9 parameters)
- **[Euler Angles](euler_angles.md)**: Intuitive angular representation (3 parameters, but with singularities)
- **[Euler Axis](euler_axis.md)**: Axis-angle representation (4 parameters)

## Choosing a Representation

**Use Quaternions when:**
- Numerical stability is critical
- Interpolating between attitudes
- Propagating attitude dynamics

**Use Rotation Matrices when:**
- Transforming vectors between frames
- Maximum computational speed is needed

**Use Euler Angles when:**
- Human readability is important
- Working with small attitude changes
- Avoiding gimbal lock singularities

**Use Euler Axis when:**
- Representing single rotations about an axis
- Geometric interpretation is important

## Conversions

All representations can be converted between each other using built-in conversion functions.

## See Also

- [API Reference - Attitude](../../library_api/attitude/index.md)
