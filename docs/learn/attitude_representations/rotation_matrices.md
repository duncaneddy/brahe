# Rotation Matrices

A rotation matrix is a 3×3 matrix that transforms vectors from one coordinate frame to another. Also known as Direction Cosine Matrices (DCM).

## Mathematical Representation

A rotation matrix is represented as:

$$
R = \begin{bmatrix}
r_{11} & r_{12} & r_{13} \\
r_{21} & r_{22} & r_{23} \\
r_{31} & r_{32} & r_{33}
\end{bmatrix}
$$

A rotation matrix $R$ satisfies the properties:

$$R^T R = I$$

$$\det(R) = 1$$

where $I$ is the identity matrix.

## Initialization

Rotation matrices can be created directly from elements, elementary rotations, or converted from other attitude representations:

=== "Python"

    ``` python
    --8<-- "examples/attitude/rotation_matrix_initialization.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "examples/attitude/rotation_matrix_initialization.rs:4"
    ```

!!! tip
    Brahe provides convenient methods to create rotation matrices for elementary rotations about the X, Y, and Z axes:

    - `RotationMatrix.Rx(angle, format)`
    - `RotationMatrix.Ry(angle, format)`
    - `RotationMatrix.Rz(angle, format)`

## Output and Access

Access rotation matrix elements and convert to other formats:

=== "Python"

    ``` python
    --8<-- "examples/attitude/rotation_matrix_output.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "examples/attitude/rotation_matrix_output.rs:4"
    ```

## Operations

Rotation matrices support composition through matrix multiplication and vector rotation:

=== "Python"

    ``` python
    --8<-- "examples/attitude/rotation_matrix_operations.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "examples/attitude/rotation_matrix_operations.rs:5"
    ```

## Conversions

Convert between rotation matrices and other attitude representations:

=== "Python"

    ``` python
    --8<-- "examples/attitude/rotation_matrix_conversions.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "examples/attitude/rotation_matrix_conversions.rs:4"
    ```

---

## See Also

- [Rotation Matrix API Reference](../../library_api/attitude/rotation_matrix.md)
- [Attitude Representations Overview](index.md)
