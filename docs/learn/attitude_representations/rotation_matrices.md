# Rotation Matrices

Rotation matrices (Direction Cosine Matrices) represent rotations as 3×3 orthogonal matrices.

## Overview

A rotation matrix is a 3×3 matrix that transforms vectors from one coordinate frame to another. Also known as Direction Cosine Matrices (DCM).

## Mathematical Representation

A rotation matrix $R$ satisfies:

$$R^T R = I$$

$$\det(R) = 1$$

where $I$ is the identity matrix.

## Initialization

### From Rotation Matrix

### From Quaternion

### From Euler Angles

### From Euler Axis

## Functions and Operationss

## Outputs

### To Rotation Matrix

### To Matrix

### To Quaternion

### To Euler Angles

### To Euler Axis

## Common Rotations

### Rotation about X-axis

### Rotation about Y-axis

### Rotation about Z-axis

## See Also

- [Rotation Matrix API Reference](../../library_api/attitude/rotation_matrix.md)
- [Attitude Representations Overview](index.md)
