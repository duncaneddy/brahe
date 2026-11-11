# PQW Frame

The perifocal frame PQW places P toward periapsis, W along the orbit normal, and $Q = W \times P$, so a satellite's own position is $r[\cos f, \sin f, 0]$ with $f$ the true anomaly. The periapsis direction comes from the eccentricity vector, which needs the central body's gravitational parameter, so the rotation has an Earth form and a `_for_body` form. On a circular orbit P is taken along the ascending node, and on a circular equatorial orbit along the inertial x axis projected into the orbit plane.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/pqw_frame.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/pqw_frame.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/relative_motion/pqw_frame.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/relative_motion/pqw_frame.rs.txt"
        ```

## See Also

- [Relative Motion](index.md) - Frame variants, rates, and covariance transformation
- [PQW Transformations API Reference](../../library_api/relative_motion/pqw_transformations.md)
