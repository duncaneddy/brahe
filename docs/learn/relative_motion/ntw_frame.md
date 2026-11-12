# NTW Frame

NTW places Y along the velocity, Z along the orbit normal, and $X = Y \times Z$ in the orbit plane, pointing outward. It coincides with RTN on a circular orbit and differs from it by the flight-path angle otherwise. Its rate is the two-body turn rate of the velocity direction, $\mu |h| / (r^3 v^2)$ about Z.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/ntw_frame.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/ntw_frame.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/relative_motion/ntw_frame.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/relative_motion/ntw_frame.rs.txt"
        ```

## See Also

- [Relative Motion](index.md) - Frame variants, rates, and covariance transformation
- [NTW Frame API Reference](../../library_api/relative_motion/ntw_frame.md)
