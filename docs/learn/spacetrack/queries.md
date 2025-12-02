# Query Filters

SpaceTrack queries support various operators for filtering results. These operators are passed as string values to the query methods.

## Basic Filtering

Filter by exact value match:

=== "Python"

    ``` python
    --8<-- "./examples/spacetrack/filter_basic.py:8"
    ```

## Comparison Operators

Use comparison operators for numeric and date fields:

=== "Python"

    ``` python
    --8<-- "./examples/spacetrack/filter_comparison.py:8"
    ```

## Range Queries

Query within a range using the `--` operator:

=== "Python"

    ``` python
    --8<-- "./examples/spacetrack/filter_range.py:8"
    ```

## Pattern Matching

Use wildcards for pattern matching:

=== "Python"

    ``` python
    --8<-- "./examples/spacetrack/filter_pattern.py:8"
    ```

## Ordering Results

Control result ordering with the `orderby` parameter:

=== "Python"

    ``` python
    --8<-- "./examples/spacetrack/filter_orderby.py:8"
    ```

## Operator Reference

| Operator | Syntax | Example | Description |
|----------|--------|---------|-------------|
| Equals | `value` | `"25544"` | Exact match |
| Not Equal | `<>value` | `<>"DEBRIS"` | Not equal to |
| Greater Than | `>value` | `">2024-01-01"` | Greater than |
| Less Than | `<value` | `"<2024-01-01"` | Less than |
| Range | `start--end` | `"2024-01-01--2024-01-31"` | Inclusive range |
| Like | `~~pattern` | `"~~STARLINK%"` | Pattern match (`%` = wildcard) |
| Starts With | `^prefix` | `"^2024-001"` | Starts with prefix |
| Null | `null-val` | `"null-val"` | Is null |

## See Also

- [Client Usage](client.md) - Basic client operations
- [Propagator Integration](propagators.md) - Converting to propagators
- [SpaceTrackClient API Reference](../../library_api/spacetrack/index.md)
