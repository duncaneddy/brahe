# EQW Transformations

The equinoctial frame EQW places E along the ascending node, W along the orbit normal, and $Q = W \times E$, so a satellite's own position is $r[\cos u, \sin u, 0]$ with $u$ the argument of latitude. On an equatorial orbit E is taken along the inertial x axis projected into the orbit plane.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/eqw_frame.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/eqw_frame.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/relative_motion/eqw_frame.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/relative_motion/eqw_frame.rs.txt"
        ```

## See Also

- [Relative Motion](index.md) - Frame variants, rates, and covariance transformation
- [EQW Transformations API Reference](../../library_api/relative_motion/eqw_transformations.md)
