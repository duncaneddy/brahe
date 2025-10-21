# Rotation Matrices

Rotation matrices (Direction Cosine Matrices) represent rotations as 3×3 orthogonal matrices.

## Overview

A rotation matrix is a 3×3 matrix that transforms vectors from one coordinate frame to another. Also known as Direction Cosine Matrices (DCM).

## Mathematical Representation

A rotation matrix $R$ satisfies:

$$R^T R = I$$

$$\det(R) = 1$$

where $I$ is the identity matrix.

## Advantages

- **Direct transformations**: Multiply matrix by vector to transform
- **Intuitive**: Each column/row represents a coordinate axis
- **Fast computation**: Matrix multiplication is highly optimized

## Disadvantages

- **Redundant**: 9 parameters represent only 3 degrees of freedom
- **Numerical drift**: Orthogonality can degrade with repeated operations
- **Storage**: Requires more memory than quaternions

## Operations

Common rotation matrix operations:

- Matrix multiplication (composition)
- Matrix transpose (inverse rotation)
- Vector transformation
- Orthogonalization (Gram-Schmidt)

## See Also

- [Rotation Matrix API Reference](../../library_api/attitude/rotation_matrix.md)
- [Attitude Representations Overview](index.md)
