# Common Queries

This page shows practical query patterns for everyday Space-Track tasks: fetching current ephemeris data, filtering the active catalog, and monitoring upcoming decays.

For the query builder API and operator reference, see [Query Builder](query_builder.md). For executing queries against the API, see [Client](client.md).

## Latest Ephemeris for a Single Object

The most common query retrieves the latest GP record for a specific satellite. Filter by `NORAD_CAT_ID`, order by `EPOCH` descending, and limit to 1 to get the most recent element set.

=== "Python"

    ``` python
    --8<-- "./examples/spacetrack/latest_ephemeris.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/spacetrack/latest_ephemeris.rs:4"
    ```

## Latest Ephemeris for Non-Decayed Objects

To query the full active catalog, filter where `DECAY_DATE` equals `null_val()`. This excludes objects that have already reentered. Combine with additional filters like `OBJECT_TYPE` or `PERIOD` to narrow the results.

=== "Python"

    ``` python
    --8<-- "./examples/spacetrack/non_decayed_objects.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/spacetrack/non_decayed_objects.rs:4"
    ```

!!! tip "Catalog Size"
    The full non-decayed catalog contains tens of thousands of records. Consider adding `OBJECT_TYPE`, orbit regime filters, or `limit()` to keep response sizes manageable.

## Objects Decaying Soon

The `Decay` request class provides reentry predictions and historical decay records. Use `inclusive_range` with `now()` and `now_offset()` to query a time window.

=== "Python"

    ``` python
    --8<-- "./examples/spacetrack/decaying_objects.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/spacetrack/decaying_objects.rs:4"
    ```

---

## See Also

- [Query Builder](query_builder.md) -- Filters, ordering, limits, and output formats
- [Conjunction Data Messages](cdm.md) -- Querying CDM collision risk data
- [Client](client.md) -- Authentication and query execution
- [Operators Reference](../../../library_api/ephemeris/spacetrack/operators.md) -- All operator functions
