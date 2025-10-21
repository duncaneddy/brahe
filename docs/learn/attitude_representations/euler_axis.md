# Euler Axis (Axis-Angle)

The Euler axis representation describes rotations using a rotation axis and angle.

## Overview

Also known as axis-angle representation, this describes any rotation as a single rotation about a unit vector (axis) by a specified angle.

## Mathematical Representation

An Euler axis rotation is specified by:

- Unit vector (axis): $\hat{n} = [n_x, n_y, n_z]$ where $|\hat{n}| = 1$
- Rotation angle: $\theta$ (in radians)

Together: $[\theta, n_x, n_y, n_z]$ (4 parameters)

## Rodrigues' Rotation Formula

Any vector $\vec{v}$ can be rotated about axis $\hat{n}$ by angle $\theta$ using:

$$\vec{v}_{rot} = \vec{v}\cos\theta + (\hat{n} \times \vec{v})\sin\theta + \hat{n}(\hat{n} \cdot \vec{v})(1-\cos\theta)$$

## Advantages

- **Intuitive**: Natural geometric interpretation
- **Minimal representation**: Efficient for single rotations
- **Useful for visualization**: Easy to show rotation axis

## Disadvantages

- **Composition complexity**: Combining rotations is not straightforward
- **Singularity at zero rotation**: Axis becomes undefined
- **Interpolation**: Non-linear

## Applications

Best used for:

- Visualizing rotation axes
- Specifying rotations geometrically
- Converting from/to other representations

## See Also

- [Euler Axis API Reference](../../library_api/attitude/euler_axis.md)
- [Attitude Representations Overview](index.md)
