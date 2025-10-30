# Quaternions

Quaternions provide a singularity-free representation of 3D rotations.

## Overview

A quaternion is a four-element mathematical object that can represent any 3D rotation without singularities. In Brahe, quaternions use the scalar-first convention: `[w, x, y, z]`.

## Mathematical Representation

A unit quaternion is defined as:

$$q = [w, x, y, z]$$

where $w^2 + x^2 + y^2 + z^2 = 1$ for unit quaternions. $w$ is the scalar part, and $(x, y, z)$ is the vector part.

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

- [Quaternion API Reference](../../library_api/attitude/quaternion.md)
- [Attitude Representations Overview](index.md)
