# Quaternions

Quaternions provide a singularity-free representation of 3D rotations.

## Overview

A quaternion is a four-element mathematical object that can represent any 3D rotation without singularities. In Brahe, quaternions use the scalar-first convention: `[w, x, y, z]`.

## Mathematical Representation

A unit quaternion is defined as:

$$q = [w, x, y, z]$$

where $w^2 + x^2 + y^2 + z^2 = 1$

## Advantages

- **No singularities**: Unlike Euler angles, quaternions work for all orientations
- **Compact**: Only 4 parameters (vs 9 for rotation matrices)
- **Efficient**: Quaternion multiplication is faster than matrix multiplication
- **Interpolation**: SLERP (Spherical Linear Interpolation) provides smooth attitude interpolation

## Operations

Common quaternion operations include:

- Quaternion multiplication (composition of rotations)
- Quaternion conjugate (inverse rotation)
- Quaternion normalization
- Vector rotation

## See Also

- [Quaternion API Reference](../../library_api/attitude/quaternion.md)
- [Attitude Representations Overview](index.md)
