# Query Builder

`SpaceTrackQuery` provides a fluent builder API for constructing Space-Track.org API queries. Each builder method returns a new query instance, allowing method chaining. Call `build()` to produce the URL path string that the client appends to the base URL.

For the complete API reference, see the [SpaceTrackQuery Reference](../../../library_api/ephemeris/spacetrack/query.md).

## Basic Queries

Create a query by specifying the request class. The default controller is selected automatically based on the class -- `GP` and `SATCAT` use `BasicSpaceData`, while `CDMPublic` uses `ExpandedSpaceData`. Add filters with the `filter()` method using Space-Track field names.

=== "Python"

    ``` python
    --8<-- "./examples/spacetrack/query_basic.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/spacetrack/query_basic.rs:4"
    ```

## Filters and Operators

The `operators` module provides functions that generate operator-prefixed strings for filter values. These compose naturally -- `greater_than(now_offset(-7))` nests the time offset inside the comparison operator.

=== "Python"

    ``` python
    --8<-- "./examples/spacetrack/query_filters.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/spacetrack/query_filters.rs:4"
    ```

!!! tip "Operator Composition"
    Operators are string-generating functions. You can compose them by nesting:

    - `greater_than(now_offset(-7))` produces `">now-7"` (epoch after 7 days ago)
    - `inclusive_range(now_offset(-30), now())` produces `"now-30--now"` (within last 30 days)

## Ordering, Limits, and Options

Control result ordering, pagination, and field selection. Multiple `order_by` calls are cumulative -- results are sorted by the first field, then by subsequent fields for ties.

=== "Python"

    ``` python
    --8<-- "./examples/spacetrack/query_advanced.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/spacetrack/query_advanced.rs:4"
    ```

## Output Formats

The default output format is JSON, which works with `query_json()`, `query_gp()`, and `query_satcat()`. Other formats like TLE, CSV, and KVN are useful with `query_raw()` for direct text output.

=== "Python"

    ``` python
    --8<-- "./examples/spacetrack/query_formats.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/spacetrack/query_formats.rs:4"
    ```

!!! note "Format and Query Method Compatibility"
    The typed query methods (`query_gp()`, `query_satcat()`, `query_json()`) require JSON format. If you set a non-JSON format, use `query_raw()` to get the raw response string.

---

## See Also

- [Space-Track API Overview](index.md) -- Module architecture and type catalog
- [Client Usage](client.md) -- Authentication and query execution
- [SpaceTrackQuery Reference](../../../library_api/ephemeris/spacetrack/query.md) -- Complete method documentation
- [Operators Reference](../../../library_api/ephemeris/spacetrack/operators.md) -- All operator functions
- [Enumerations Reference](../../../library_api/ephemeris/spacetrack/enums.md) -- RequestClass, OutputFormat, etc.
