# Multithreading

Brahe uses a global thread pool to parallelize computationally intensive operations, such as computing access windows between satellites and ground locations. The threading utilities allow you to configure the number of threads used by the thread pool.

For complete API details, see the [Threading API Reference](../../library_api/utils/threading.md).

## Default Behavior

By default, Brahe automatically configures the thread pool to use **90% of available CPU cores** on first use. This greatly accelerates computations while leaving some resources for other processes to avoid resource-starving other processes on the machine.

For example, on a system with 8 CPU cores, Brahe will use 7 threads by default.

!!! tip "Lazy Initialization"

    The thread pool is initialized on first use, not when you import Brahe. This means the default thread count is determined when you first call a function that uses the thread pool.

    You can configure the thread pool before first use to override the default behavior by calling `set_num_threads()` or `set_max_threads()` as shown below.

!!! note "Thread Safety"

    All Brahe functions are thread-safe. You can safely call Brahe functions from multiple threads simultaneously.

## Setting Thread Count

### Set Specific Number

You can set the thread pool to use a specific number of threads:

=== "Python"

    ``` python
    --8<-- "./examples/utilities/parallelization.py:20:23"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/utilities/parallelization.rs:17:20"
    ```

### Set Maximum Threads

To use all available CPU cores (100%), use `set_max_threads()`:

=== "Python"

    ``` python
    --8<-- "./examples/utilities/parallelization.py:25:28"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/utilities/parallelization.rs:22:25"
    ```

!!! tip "When to Use Maximum Threads"

    Use `set_max_threads()` when Brahe is the sole computational task running on a server and you want to maximize throughput.

### Ludicrous Speed!

For a bit of fun, there's an alias for `set_max_threads()`:

=== "Python"

    ``` python
    --8<-- "./examples/utilities/parallelization.py:30:33"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/utilities/parallelization.rs:27:30"
    ```

## Querying Thread Count

You can check the current thread pool configuration at any time:

=== "Python"

    ``` python
    --8<-- "./examples/utilities/parallelization.py:15:18"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/utilities/parallelization.rs:12:15"
    ```

## Reconfiguring the Thread Pool

The thread pool can be reconfigured at any time during program execution. Simply call `set_num_threads()` or `set_max_threads()` again with the new desired configuration:

=== "Python"

    ``` python
    --8<-- "./examples/utilities/parallelization.py:35:38"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/utilities/parallelization.rs:32:35"
    ```

## Complete Example

Here's a complete example demonstrating all threading configuration functions:

=== "Python"

    ``` python
    --8<-- "./examples/utilities/parallelization.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/utilities/parallelization.rs:7"
    ```

---

## See Also

- [Utilities Overview](index.md) - Overview of all utilities
- [Threading API Reference](../../library_api/utils/threading.md) - Complete threading function documentation
