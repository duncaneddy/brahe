# Epoch

The Epoch class is the fundamental time representation in Brahe. It encapsulates a specific instant in time, defined by both a time representation and a time scale. The Epoch class provides methods for converting between different time representations and time scales, as well as for performing arithmetic operations on time instances.

There are even more capabilities and features of the Epoch class beyond what is covered in this guide. For a complete reference of all available methods and properties, please refer to the [Epoch API Reference](../../library_api/time/epoch.md).

## Initialization

THere are all sorts of ways you can initialize an Epoch instance. The most common methods are described below.

### Date Time

### MJD

### JD

### String

### GPS Week and Seconds

### GPS Week and Seconds

## Operations

Once you have an epoch class instance you can add and subtract time as you would expect. 

!!! warning

    When performing arithmetic the other operand is always interpreted as a time duration in seconds.

### Addition

### Subtraction

## Output and Formatting

Finally, you can take any Epoch instance and then output it in different representations.

### Date Time

### String Representation

### ISO 8601 String

## Time Ranges

The `TimeRange` class provides an easy way to iterate over a range of time instances. You can specify a start and end `Epoch`, along with a time step in seconds, and the `TimeRange` will generate all the `Epoch` instances within that range at the specified intervals.

=== "Python"

    ``` python
    --8<-- "./examples/time/time_range.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/time/time_range.rs:4"
    ```

## See Also

- [Epoch API Reference](../../library_api/time/epoch.md)
- [Time Systems](time_systems.md)
- [Time Conversions](time_conversions.md)
