# Euler Angles

Euler angles represent rotations as three sequential rotations about coordinate axes.

## Overview

Euler angles describe orientation using three angles representing sequential rotations about specified axes. Brahe supports all 12 possible Euler angle sequences (e.g., XYZ, ZYX, ZYZ).

!!! info "Euler Angle Application Order Conventions"

    Brahe follows standard aerospace engineering conventions for the naming of Euler angle sequences with the application of rotation matrices being applied from _left_ to _right_.

    A 3-2-1 (Z-Y-X) Euler angle sequence is equivalent to the multication by $R$ where $R = R_x(\psi)R_y(\theta)R_z(\phi)$. Note the "3" (Z) is the first rotation applied when a vector $\vec{x}$ is rotated: $R\vec{x}$. Then the "2" (Y) is applied, and finally the "1" (X) is applied.

    This is slightly at odds with the as-written Diebel convention, but is consistent with the aerospace engineering convention. Brahe internally transforms between the aerospace convention to the Diebel convention.

## Mathematical Representation

An Euler angle rotation is specified by:

- Three angles: $(\phi, \theta, \psi)$
- A rotation sequence (e.g., XYZ, ZYX)

## Sequences

Brahe supports all valid Euler angle sequences, though some are more commonly used are:

- **ZYX (3-2-1)**: Common in aerospace applications. Known as yaw-pitch-roll.
- **XYZ (1-2-3)**: Common in robotics
- **ZYZ (3-1-3)**: Common in classical mechanics

## Initialization

Euler angles can be created from individual angles with a specified rotation sequence, or converted from other attitude representations. When creating a new `EulerAngle` object, the rotation sequence of the created object must be specified.

=== "Python"

    ``` python
    --8<-- "examples/attitude/euler_angle_initialization.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "examples/attitude/euler_angle_initialization.rs:4"
    ```

## Conversions

Convert between Euler angles and other attitude representations:

=== "Python"

    ``` python
    --8<-- "examples/attitude/euler_angle_conversions.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "examples/attitude/euler_angle_conversions.rs:4"
    ```

---

## See Also

- [Euler Angles API Reference](../../library_api/attitude/euler_angles.md)
- [Attitude Representations Overview](index.md)
