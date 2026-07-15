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

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/time/epoch_datetime.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/time/epoch_datetime.rs.txt"
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

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/time/epoch_mjd.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/time/epoch_mjd.rs.txt"
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

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/time/epoch_jd.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/time/epoch_jd.rs.txt"
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

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/time/epoch_string.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/time/epoch_string.rs.txt"
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

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/time/epoch_gps.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/time/epoch_gps.rs.txt"
        ```

## Time System

Every `Epoch` carries a `TimeSystem`, set at construction. It records the time
scale the epoch is expressed in. For what each scale means and when to use it,
see [Time Systems and Representations](index.md).

In Python the time system is optional and defaults to UTC when omitted; it can
be given either as an enumeration member (`bh.TimeSystem.GPS`) or as the
equivalent module-level constant (`bh.GPS`). Rust requires it explicitly at
construction, using `bh::TimeSystem::GPS`.

An `Epoch` stores an absolute instant, and the time system only determines the
scale it reports in. `to_time_system` returns a new `Epoch` at the same instant
expressed in a different scale — it changes how the epoch prints, not when it
is. The original is left untouched, and the two compare equal because they
denote the same instant.

To read a single value out in another scale without creating a new `Epoch`, use
the `*_as_time_system` family: `to_datetime_as_time_system`,
`to_string_as_time_system`, `jd_as_time_system`, `mjd_as_time_system`,
`day_of_year_as_time_system`, and `seconds_past_j2000_as_time_system`.

=== "Python"

    ``` python
    --8<-- "./examples/time/epoch_time_system.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/time/epoch_time_system.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/time/epoch_time_system.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/time/epoch_time_system.rs.txt"
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

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/time/epoch_addition.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/time/epoch_addition.rs.txt"
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

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/time/epoch_subtraction.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/time/epoch_subtraction.rs.txt"
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

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/time/epoch_other_operations.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/time/epoch_other_operations.rs.txt"
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

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/time/epoch_output.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/time/epoch_output.rs.txt"
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

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/time/epoch_string_output.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/time/epoch_string_output.rs.txt"
        ```

### Seconds Past J2000 in a Specific Time System

`seconds_past_j2000_as_time_system` returns the number of seconds elapsed
since the J2000 epoch (2000-01-01 12:00:00 TT), expressed in a chosen time
system. Requesting `TimeSystem.TDB` gives SPICE ephemeris time (ET), the
time argument used by SPK/PCK kernel queries (`spk_position`,
`sun_position_spice`, ...); see [SPICE Kernels](../spice/index.md).
`spice_et()` is a convenience alias for
`seconds_past_j2000_as_time_system(TimeSystem.TDB)`.

=== "Python"

    ```python
    --8<-- "./examples/time/epoch_spice_et.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/time/epoch_spice_et.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/time/epoch_spice_et.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/time/epoch_spice_et.rs.txt"
        ```

---

## See Also

- [Epoch API Reference](../../library_api/time/epoch.md)
- [TimeSystem API Reference](../../library_api/time/time_system.md)
- [SPICE Kernels](../spice/index.md) - Using ET for kernel queries