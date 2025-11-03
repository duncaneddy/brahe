# String Formatting

The string formatting utilities provide functions for converting numerical values into human-readable strings. This makes it easier to display results to users in an intuitive format.

For complete API details, see the [String Formatting API Reference](../../library_api/utils/formatting.md).

## Time Duration Formatting

The primary formatting utility is `format_time_string()`, which converts a time duration in seconds into a human-readable string.

### Long Format (Default)

The long format provides a full description with proper units and grammar:

=== "Python"

    ``` python
    --8<-- "./examples/utilities/string_formatting.py:15:21"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/utilities/string_formatting.rs:12:18"
    ```

The long format uses proper grammar and includes fractional seconds:

- `30.00 seconds`
- `1 minute and 30.00 seconds`
- `6 minutes and 2.00 seconds`
- `1 hour, 1 minute and 5.00 seconds`
- `1 day, 1 hour and 0.00 seconds`

### Short Format

The short format provides a more compact representation suitable for tables or limited space:

=== "Python"

    ``` python
    --8<-- "./examples/utilities/string_formatting.py:23:29"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/utilities/string_formatting.rs:20:26"
    ```

The short format uses abbreviations without fractional seconds:

- `30s`
- `1m 30s`
- `6m 2s`
- `1h 1m 5s`
- `1d 1h 0m`

## Practical Example

Here's a practical example formatting an orbital period:

=== "Python"

    ``` python
    --8<-- "./examples/utilities/string_formatting.py:31:34"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/utilities/string_formatting.rs:28:31"
    ```

This produces:

```
LEO orbital period: 1 hour, 34 minutes and 38.34 seconds
LEO orbital period (short): 1h 34m 38s
```

## Supported Time Units

The function automatically selects the appropriate units based on the duration:

| Duration Range | Units Used |
|---------------|------------|
| < 60 seconds | seconds only |
| 60s - 1 hour | minutes and seconds |
| 1 hour - 1 day | hours, minutes, and seconds |
| > 1 day | days, hours, minutes (short format) or days, hours, minutes, and seconds (long format) |

!!! note "Precision"

    - **Long format**: Displays seconds with 2 decimal places
    - **Short format**: Displays only whole seconds (fractional part truncated)

## Complete Example

Here's a complete example demonstrating both formats with various durations:

=== "Python"

    ``` python
    --8<-- "./examples/utilities/string_formatting.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/utilities/string_formatting.rs:7"
    ```

## Common Use Cases

Time formatting is useful for:

- **Access window durations**: Display how long a satellite is visible from a ground station
- **Orbital periods**: Show the time for one complete orbit in readable form
- **Propagation times**: Display simulation duration or time steps
- **Reports and output**: Present timing information to users

!!! tip "Choosing a Format"

    - Use **long format** for reports, documentation, and user-facing output where clarity is important
    - Use **short format** for tables, logs, and situations where space is limited

## See Also

- [Utilities Overview](index.md) - Overview of all utilities
- [String Formatting API Reference](../../library_api/utils/formatting.md) - Complete formatting function documentation
