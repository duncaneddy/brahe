# Euler Axis (Axis-Angle)

The Euler axis representation describes rotations using a rotation axis and angle.

## Overview

Also known as axis-angle representation, this describes any rotation as a single rotation about a unit vector (axis) by a specified angle.

## Mathematical Representation

An Euler axis rotation is specified by:

- Unit vector (axis): $\hat{n} = [n_x, n_y, n_z]$ where $|\hat{n}| = 1$
- Rotation angle: $\theta$ (in radians)

Together: $[\theta, n_x, n_y, n_z]$ (4 parameters)

## Initialization

### From Rotation Matrix

### From Quaternion

### From Euler Angles

### From Euler Axis

## Functions and Operations

## Outputs

### To Rotation Matrix

### To Matrix

### To Quaternion

### To Euler Angles

### To Euler Axis

## See Also

- [Euler Axis API Reference](../../library_api/attitude/euler_axis.md)
- [Attitude Representations Overview](index.md)
