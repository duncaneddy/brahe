# Time Range

The `TimeRange` class provides an easy way to iterate over a range of time instances. You can specify a start and end `Epoch`, along with a time step in seconds, and the `TimeRange` will generate all the `Epoch` instances within that range at the specified intervals.

=== "Python"

    ``` python
    --8<-- "./examples/time/time_range.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/time/time_range.rs:4"
    ```