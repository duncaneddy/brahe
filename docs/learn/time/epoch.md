# Epoch

The Epoch class is the fundamental time representation in Brahe. It encapsulates a specific instant in time, defined by both a time representation and a time scale. The Epoch class provides methods for converting between different time representations and time scales, as well as for performing arithmetic operations on time instances.

There are even more capabilities and features of the Epoch class beyond what is covered in this guide. For a complete reference of all available methods and properties, please refer to the [Epoch API Reference](../../library_api/time/epoch.md).

## Initialization

There are all sorts of ways you can initialize an Epoch instance. The most common methods are described below.

### Date Time

The most common way to create an Epoch is from date and time components. You can specify just a date (which defaults to midnight), or provide the full date and time including fractional seconds.

=== "Python"

    ``` python
    --8<-- "./examples/time/epoch_datetime.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/time/epoch_datetime.rs:4"
    ```

### MJD

Modified Julian Date (MJD) is a commonly used time representation in astronomy and astrodynamics. MJD is defined as JD - 2400000.5, which makes it more convenient for modern dates.

=== "Python"

    ``` python
    --8<-- "./examples/time/epoch_mjd.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/time/epoch_mjd.rs:4"
    ```

### JD

Julian Date (JD) is a continuous count of days since the beginning of the Julian Period. It's widely used in astronomy for precise time calculations.

=== "Python"

    ``` python
    --8<-- "./examples/time/epoch_jd.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/time/epoch_jd.rs:4"
    ```

### String

Epoch instances can be created from ISO 8601 formatted strings or simple date-time strings. The time system can be specified in the string.

=== "Python"

    ``` python
    --8<-- "./examples/time/epoch_string.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/time/epoch_string.rs:4"
    ```

### GPS Week and Seconds

For GPS applications, you can create epochs from GPS week number and seconds into the week, or from GPS seconds since the GPS epoch (January 6, 1980).

=== "Python"

    ``` python
    --8<-- "./examples/time/epoch_gps.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/time/epoch_gps.rs:4"
    ```

## Operations

Once you have an epoch class instance you can add and subtract time as you would expect.

!!! info

    When performing arithmetic the other operand is always interpreted as a time duration in **seconds**.

### Addition

You can add a time duration (in seconds) to an Epoch to get a new Epoch at a later time.

=== "Python"

    ``` python
    --8<-- "./examples/time/epoch_addition.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/time/epoch_addition.rs:4"
    ```

### Subtraction

Subtracting two Epoch instances returns the time difference between them in seconds.

=== "Python"

    ``` python
    --8<-- "./examples/time/epoch_subtraction.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/time/epoch_subtraction.rs:4"
    ```

### Other Operations

The Epoch class also supports comparison operations (e.g., equality, less than, greater than) to compare different time instances. It also supports methods for getting string representations using language-specific formatting options.

=== "Python"

    ``` python
    --8<-- "./examples/time/epoch_other_operations.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/time/epoch_other_operations.rs:4"
    ```

## Output and Formatting

Finally, you can take any Epoch instance and then output it in different representations.

### Date Time

You can extract the date and time components from an Epoch, optionally converting to a different time system.

=== "Python"

    ``` python
    --8<-- "./examples/time/epoch_output.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/time/epoch_output.rs:4"
    ```

### String Representation

Epochs can be converted to human-readable strings in various formats and time systems.

=== "Python"

    ``` python
    --8<-- "./examples/time/epoch_string_output.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/time/epoch_string_output.rs:4"
    ```

## See Also

- [Epoch API Reference](../../library_api/time/epoch.md)