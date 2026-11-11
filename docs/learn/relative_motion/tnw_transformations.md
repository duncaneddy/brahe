# TNW Transformations

TNW is NTW with its in-plane axes reordered, $X_\mathrm{TNW} = Y_\mathrm{NTW}$ and $Y_\mathrm{TNW} = -X_\mathrm{NTW}$, so X lies along the velocity and Y points inward. Its rate is the NTW rate about Z.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/tnw_frame.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/tnw_frame.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/relative_motion/tnw_frame.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/relative_motion/tnw_frame.rs.txt"
        ```

## See Also

- [Relative Motion](index.md) - Frame variants, rates, and covariance transformation
- [TNW Transformations API Reference](../../library_api/relative_motion/tnw_transformations.md)
