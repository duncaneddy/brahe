# Package Conventions

There are a few common conventions that are useful to know when working with Brahe.

## Vector Types

In Python, Brahe uses array-like types (lists, tuples, numpy arrays, etc.) as inputs for vector quantities (e.g. position, velocity, etc.) and returns numpy arrays as outputs for vector quantities. In Rust, Brahe uses nalgebra arrays for both inputs and outputs of vector quantities.

=== "Python"

    ``` python
    --8<-- "./examples/getting_started/vector_types.py:4"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/getting_started/vector_types.rs:1"
    ```

???+ example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/getting_started/vector_types.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/getting_started/vector_types.rs.txt"
        ```

## SI Units

Brahe uses SI _**base units**_ (meters, seconds, kilograms, etc.) for all inputs and outputs of physical quantities unless otherwise specified. This is to remove any ambiguity about the units of inputs and outputs, and to make it easier to work with the library and chain together different functions without needing to worry about unit conversions.

It is the _responsibility of the user_ to ensure that the inputs to Brahe functions are in the correct units, and to convert the outputs of Brahe functions to the desired units if needed.

## Angle Format

For functions that deal with angular quantities for either inputs or outputs, Brahe provides the [`AngleFormat`](../library_api/coordinates/enums.md) enum to specify the format of the angles. Given the frequency of working with different angle formats this makes it easy to work with different formats without needing to manually convert before or after calling Brahe functions.

## See Also

- [Coordinate Enums API Reference](../library_api/coordinates/enums.md) — including `AngleFormat`
- [Units and Constants API Reference](../library_api/constants/units.md)
- [Coordinates User Guide](../learn/coordinates/index.md)