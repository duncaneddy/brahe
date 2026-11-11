# VNC Transformations

VNC (Velocity, Normal, Co-normal) places X along the velocity, Y along the orbit normal, and the co-normal $Z = X \times Y$ in the orbit plane, pointing outward, so $Z_\mathrm{VNC} = X_\mathrm{NTW}$. STK calls this frame VNB. Its rate is the NTW rate about Y.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/vnc_frame.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/vnc_frame.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/relative_motion/vnc_frame.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/relative_motion/vnc_frame.rs.txt"
        ```

## See Also

- [Relative Motion](index.md) - Frame variants, rates, and covariance transformation
- [VNC Transformations API Reference](../../library_api/relative_motion/vnc_transformations.md)
