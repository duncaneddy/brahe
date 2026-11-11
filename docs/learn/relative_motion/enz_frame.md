# ENZ Frame

ENZ is the topocentric horizon frame with E pointing east, N north, and Z along the geodetic vertical. It is SEZ with its horizontal axes reordered, $E_\mathrm{ENZ} = E_\mathrm{SEZ}$ and $N_\mathrm{ENZ} = -S_\mathrm{SEZ}$, and its functions take the same ECEF site and target states.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/enz_frame.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/enz_frame.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/relative_motion/enz_frame.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/relative_motion/enz_frame.rs.txt"
        ```

## See Also

- [Relative Motion](index.md) - Frame variants, rates, and covariance transformation
- [ENZ Frame API Reference](../../library_api/relative_motion/enz_frame.md)
