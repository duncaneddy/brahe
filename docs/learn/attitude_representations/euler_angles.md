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

## Initialization

Euler angles can be created from individual angles with a specified rotation sequence, or converted from other attitude representations:

=== "Python"

    ``` python
    --8<-- "examples/attitude/euler_angle_initialization.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "examples/attitude/euler_angle_initialization.rs:4"
    ```

## Output and Access

Access individual Euler angles and the rotation sequence:

=== "Python"

    ``` python
    --8<-- "examples/attitude/euler_angle_output.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "examples/attitude/euler_angle_output.rs:4"
    ```

## Operations

Euler angles are typically converted to other representations for composition. Direct operations can suffer from gimbal lock:

=== "Python"

    ``` python
    --8<-- "examples/attitude/euler_angle_operations.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "examples/attitude/euler_angle_operations.rs:4"
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

## See Also

- [Euler Angles API Reference](../../library_api/attitude/euler_angles.md)
- [Attitude Representations Overview](index.md)
