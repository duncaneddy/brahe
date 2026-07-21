# Time

> In the beginning [time] was created. This has made a lot of people very angry and has been widely regarded as a bad move.
>
> -- Douglas Adams, The Restaurant at the End of the Universe

Since astrodynamics is the study of the motion of objects in space, time is a fundamental quantity to the package. Unfortunately, time can quickly become a complex topic with many different time systems, formats, and conventions.

Brahe attempts to solve this challenge by providing the [`Epoch`](../library_api/time/epoch.md) class to represent a single instant in time. An `Epoch` can be initialized in a variety of ways, it can be manipulated with simple arithmetic operations, and and it can be compared to other `Epoch` instances. Brahe ensures that all of the complexities of time systems, formats, and conventions are handled internally by the `Epoch` class so that users can work with time in a simple and intuitive way.

There are more ways to work with Epochs and time than are covered here. Checkout the [Time User Guide](../learn/time/index.md) or the [API Reference](../library_api/time/index.md) for more details and examples of working with time in Brahe.

## Epoch Creation

The `Epoch` class can be initialized in a variety of ways:

=== "Python"

    ``` python
    --8<-- "./examples/getting_started/epoch_initialization.py:4"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/getting_started/epoch_initialization.rs:1"
    ```

## Epoch Operations

Brahe supports intuitive operations on time through arithemtic operations on `Epoch` instances.

!!! tip "SI Units"
    All floating point addition/subtraction values are assumed to be in seconds. All differences between `Epoch` instances are returned in seconds.

=== "Python"

    ``` python
    --8<-- "./examples/getting_started/epoch_operations.py:4"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/getting_started/epoch_operations.rs:1"
    ```

???+ example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/getting_started/epoch_operations.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/getting_started/epoch_operations.rs.txt"
        ```

## Epoch Output

There are also a variety of ways to output `Epoch` instances as other represenstations of instances in time.

=== "Python"

    ``` python
    --8<-- "./examples/getting_started/epoch_output.py:4"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/getting_started/epoch_output.rs:1"
    ```

???+ example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/getting_started/epoch_output.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/getting_started/epoch_output.rs.txt"
        ```

## Time Ranges 

Since it's common to work with time ranges, Brahe provides the [`TimeRange`](../library_api/time/time_range.md) iterator to quickly generate a range of epochs with a specified time step. 

=== "Python"

    ``` python
    --8<-- "./examples/getting_started/epoch_range.py:4"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/getting_started/epoch_range.rs:1"
    ```

???+ example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/getting_started/epoch_range.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/getting_started/epoch_range.rs.txt"
        ```

## Time Systems

In astrodynamics, we often deal with models and measurements in other time systems (e.g. GPS, TAI, TT, etc.). Brahe provides support for working with different time systems through the [`TimeSystem`](../library_api/time/time_system.md) enum. When creating an `Epoch`, you can specify the time system of the input time, and Brahe will handle the conversion to the internal time system (TAI) for you. Similarly when outputting an `Epoch`, you can specify the desired time system for the output, and Brahe will handle the conversion for you. Comparisons between `Epoch` instances are time-system aware, so you can compare `Epoch` instances in different time systems and Brahe will handle the conversion for you.

=== "Python"

    ``` python
    --8<-- "./examples/getting_started/epoch_time_systems.py:4"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/getting_started/epoch_time_systems.rs:1"
    ```

???+ example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/getting_started/epoch_time_systems.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/getting_started/epoch_time_systems.rs.txt"
        ```

## See Also

- [Epoch API Reference](../library_api/time/epoch.md)
- [TimeRange API Reference](../library_api/time/time_range.md)
- [TimeSystem API Reference](../library_api/time/time_system.md)
- [Epoch (Learn)](../learn/time/epoch.md)
- [Time Ranges (Learn)](../learn/time/time_range.md)
- [Time User Guide](../learn/time/index.md)