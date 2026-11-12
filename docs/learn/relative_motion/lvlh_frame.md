# LVLH Frame

LVLH (Local Vertical, Local Horizontal) places Z along nadir, Y opposite the orbit normal, and X completing the right-handed set. Its rate is the true-anomaly rate about $-Y$. Some references, including Vallado and STK, use the name LVLH for the RTN axes. In brahe the two are distinct frames, related by $X_\mathrm{LVLH} = T$, $Y_\mathrm{LVLH} = -N$, and $Z_\mathrm{LVLH} = -R$.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/lvlh_frame.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/lvlh_frame.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/relative_motion/lvlh_frame.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/relative_motion/lvlh_frame.rs.txt"
        ```

## See Also

- [Relative Motion](index.md) - Frame variants, rates, and covariance transformation
- [LVLH Frame API Reference](../../library_api/relative_motion/lvlh_frame.md)
