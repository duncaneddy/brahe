# Quaternions

A quaternion is a four-element mathematical object that can represent any 3D rotation without singularities. In Brahe, quaternions use the scalar-first convention: `[w, x, y, z]`.

## Mathematical Representation

A unit quaternion is defined as:

$$q = [w, x, y, z]$$

where $w^2 + x^2 + y^2 + z^2 = 1$ for unit quaternions. $w$ is the scalar part, and $(x, y, z)$ is the vector part. Quaternions can also be formulated with the scalar part as the last element, which brahe also supports for input/output.

## Initialization

Quaternions can be initialized in several ways, including directly from all other attitude representations:

=== "Python"

    ``` python
    --8<-- "examples/attitude/quaternion_initialization.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "examples/attitude/quaternion_initialization.rs:4"
    ```

## Output and Access

You can access quaternion components directly or convert them to other data formats:


=== "Python"

    ``` python
    --8<-- "examples/attitude/quaternion_output.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "examples/attitude/quaternion_output.rs:4"
    ```

## Operations

Quaternions support multiplication, normalization, conjugation, inversion, and interpolation (through [Spherical Linear Interpolation (SLERP)](https://en.wikipedia.org/wiki/Slerp)):

=== "Python"

    ``` python
    --8<-- "examples/attitude/quaternion_operations.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "examples/attitude/quaternion_operations.rs:4"
    ```

## Conversions

You can convert quaternions to all other attitude representations and vice versa:

=== "Python"

    ``` python
    --8<-- "examples/attitude/quaternion_conversions.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "examples/attitude/quaternion_conversions.rs:4"
    ```

---

## See Also

- [Quaternion API Reference](../../library_api/attitude/quaternion.md)
- [Attitude Representations Overview](index.md)
