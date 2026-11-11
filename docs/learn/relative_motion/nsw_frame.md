# NSW Frame

NSW places X toward nadir, Y toward the Sun as nearly as possible while normal to X, and $Z = X \times Y$. Each function takes the Sun's state, relative to the same center as the spacecraft state, as the argument after the spacecraft states. The rate follows from the time derivatives of the axes and includes the Sun's velocity. When the Sun lies on the nadir line, Y falls back to the along-track direction $\hat{h} \times \hat{r}$. The frame graph takes the Sun's state from the SPICE registry.

=== "Python"

    ``` python
    --8<-- "./examples/relative_motion/nsw_frame.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/relative_motion/nsw_frame.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/relative_motion/nsw_frame.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/relative_motion/nsw_frame.rs.txt"
        ```

## See Also

- [Relative Motion](index.md) - Frame variants, rates, and covariance transformation
- [NSW Transformations API Reference](../../library_api/relative_motion/nsw_transformations.md)
