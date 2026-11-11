# SEZ Frame

SEZ is the topocentric horizon frame of a site on the WGS84 ellipsoid: S points south, E east, and Z along the geodetic vertical. Its functions take the ECEF states of the site and of a target, so the frame is Earth-only. `omega_sez` is the rate relative to ECEF, $[-\dot\lambda \cos\varphi, -\dot\varphi, \dot\lambda \sin\varphi]$ from the site's longitude and geodetic latitude rates, which is zero for a fixed site. The frame graph adds Earth's rotation.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/sez_frame.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/sez_frame.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/relative_motion/sez_frame.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/relative_motion/sez_frame.rs.txt"
        ```

## See Also

- [Relative Motion](index.md) - Frame variants, rates, and covariance transformation
- [SEZ Transformations API Reference](../../library_api/relative_motion/sez_transformations.md)
