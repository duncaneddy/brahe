# Frame Transformations

Brahe provides functions for transforming between different reference frames.

!!! tip "Cartesian Inputs"
    All frame transformation functions in Brahe assume the input state are in Cartesian coordinates (position and velocity).

## Earth-Centered Inertial (ECI) and Earth-Centered Earth-Fixed (ECEF) Transformations

We can transform from ECI to ECEF using `state_eci_to_ecef`, which properly accounts for Earth's precession, nutation, sidereal rotation, and polar motion.

=== "Python"

    ``` python
    --8<-- "./examples/getting_started/frames_eci_ecef.py:4"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/getting_started/frames_eci_ecef.rs:1"
    ```

???+ example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/getting_started/frames_eci_ecef.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/getting_started/frames_eci_ecef.rs.txt"
        ```
