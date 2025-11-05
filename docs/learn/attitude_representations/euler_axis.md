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

Euler axis representations can be created from an axis vector and angle, or converted from other attitude representations:

=== "Python"

    ``` python
    --8<-- "examples/attitude/euler_axis_initialization.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "examples/attitude/euler_axis_initialization.rs:4"
    ```

## Conversions

Convert between Euler axis and other attitude representations:

=== "Python"

    ``` python
    --8<-- "examples/attitude/euler_axis_conversions.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "examples/attitude/euler_axis_conversions.rs:4"
    ```

---

## See Also

- [Euler Axis API Reference](../../library_api/attitude/euler_axis.md)
- [Attitude Representations Overview](index.md)
