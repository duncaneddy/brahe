# Client

`SpaceTrackClient` handles authentication and HTTP communication with Space-Track.org. It manages session cookies, provides both raw and typed query execution, and automatically authenticates on the first request if needed.

For the complete API reference, see the [SpaceTrackClient Reference](../../../library_api/ephemeris/spacetrack/client.md).

## Client Creation

Create a client with your Space-Track.org credentials:

=== "Python"

    ``` python
    import brahe as bh
    client = bh.SpaceTrackClient("user@example.com", "password")
    ```

=== "Rust"

    ``` rust
    use brahe::spacetrack::SpaceTrackClient;
    let client = SpaceTrackClient::new("user@example.com", "password");
    ```

## Rate Limiting

The client includes a built-in rate limiter that prevents exceeding Space-Track.org's request limits. By default, conservative limits of 25 requests/minute and 250 requests/hour are applied automatically. Pass a `RateLimitConfig` to customize or disable rate limiting. For a full explanation of how rate limiting works, see [Rate Limiting](rate_limiting.md).

## Authentication

The client authenticates lazily -- credentials are sent on the first query. If you want to verify credentials before executing queries, call `authenticate()` explicitly. This is useful for failing fast on bad credentials rather than discovering the problem mid-workflow.

## Query Execution

The client provides four query methods, each returning data in a different form:

<div class="center-table" markdown="1">

| Method | Return Type | Format Required |
|--------|-------------|-----------------|
| `query_raw(query)` | Raw string | Any |
| `query_json(query)` | JSON array | JSON |
| `query_gp(query)` | `GPRecord` list | JSON |
| `query_satcat(query)` | `SATCATRecord` list | JSON |

</div>

### Typed Queries

Use `query_gp()` and `query_satcat()` for strongly-typed access to GP and SATCAT data:

=== "Python"

    ``` python
    import brahe as bh

    client = bh.SpaceTrackClient("user@example.com", "password")

    # Query GP data for the ISS
    query = (
        bh.SpaceTrackQuery(bh.RequestClass.GP)
        .filter("NORAD_CAT_ID", "25544")
        .order_by("EPOCH", bh.SortOrder.DESC)
        .limit(1)
    )
    records = client.query_gp(query)

    iss = records[0]
    print(f"Object: {iss.object_name}")
    print(f"Epoch: {iss.epoch}")
    print(f"Inclination: {iss.inclination}")

    # Query SATCAT metadata
    query = (
        bh.SpaceTrackQuery(bh.RequestClass.SATCAT)
        .filter("NORAD_CAT_ID", "25544")
        .limit(1)
    )
    records = client.query_satcat(query)

    iss_meta = records[0]
    print(f"Name: {iss_meta.satname}")
    print(f"Country: {iss_meta.country}")
    print(f"Launch: {iss_meta.launch}")
    ```

=== "Rust"

    ``` rust
    use brahe::spacetrack::{SpaceTrackClient, SpaceTrackQuery, RequestClass, SortOrder};

    let client = SpaceTrackClient::new("user@example.com", "password");

    // Query GP data for the ISS
    let query = SpaceTrackQuery::new(RequestClass::GP)
        .filter("NORAD_CAT_ID", "25544")
        .order_by("EPOCH", SortOrder::Desc)
        .limit(1);
    let records = client.query_gp(&query).unwrap();

    let iss = &records[0];
    println!("Object: {:?}", iss.object_name);
    println!("Epoch: {:?}", iss.epoch);
    println!("Inclination: {:?}", iss.inclination);

    // Query SATCAT metadata
    let query = SpaceTrackQuery::new(RequestClass::SATCAT)
        .filter("NORAD_CAT_ID", "25544")
        .limit(1);
    let records = client.query_satcat(&query).unwrap();

    let iss_meta = &records[0];
    println!("Name: {:?}", iss_meta.satname);
    println!("Country: {:?}", iss_meta.country);
    println!("Launch: {:?}", iss_meta.launch);
    ```

For non-JSON formats (TLE, CSV, KVN), use `query_raw()` to get the response as a plain string. For generic JSON access without deserialization into a specific record type, use `query_json()`.

## Error Handling

The client returns errors for authentication failures, network issues, and format mismatches. In Python, these raise exceptions; in Rust, they return `Result<_, BraheError>`.

Common error scenarios:

- **Invalid credentials** -- `authenticate()` or the first query fails
- **Format mismatch** -- Using `query_gp()` with a non-JSON format set on the query
- **Network errors** -- Connection failures, timeouts
- **API errors** -- Invalid field names, unsupported filter combinations

---

## See Also

- [Space-Track API Overview](index.md) -- Module architecture and type catalog
- [Query Builder](query_builder.md) -- Building queries with filters and operators
- [File Operations](file_operations.md) -- FileShare, SP Ephemeris, and Public Files
- [SpaceTrackClient Reference](../../../library_api/ephemeris/spacetrack/client.md) -- Complete method documentation
- [Response Types Reference](../../../library_api/ephemeris/spacetrack/responses.md) -- All response type field definitions
