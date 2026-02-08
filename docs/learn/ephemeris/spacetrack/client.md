# Client

`SpaceTrackClient` handles authentication and HTTP communication with Space-Track.org. It manages session cookies, provides both raw and typed query execution, and automatically authenticates on the first request if needed.

For the complete API reference, see the [SpaceTrackClient Reference](../../../library_api/ephemeris/spacetrack/client.md).

## Client Creation

Create a client with your Space-Track.org credentials. An optional `base_url` parameter allows pointing to a different endpoint for testing.

=== "Python"

    ``` python
    import brahe as bh

    # Standard client
    client = bh.SpaceTrackClient("user@example.com", "password")

    # Client with custom base URL (e.g., for testing)
    client = bh.SpaceTrackClient("user@example.com", "password",
                                  base_url="https://test.space-track.org")
    ```

=== "Rust"

    ``` rust
    use brahe::spacetrack::SpaceTrackClient;

    // Standard client
    let client = SpaceTrackClient::new("user@example.com", "password");

    // Client with custom base URL (e.g., for testing)
    let client = SpaceTrackClient::with_base_url(
        "user@example.com", "password",
        "https://test.space-track.org"
    );
    ```

## Rate Limiting

The client includes a built-in rate limiter that prevents exceeding Space-Track.org's request limits. By default, conservative limits of 25 requests/minute and 250 requests/hour are applied automatically. Pass a `RateLimitConfig` to customize or disable rate limiting.

=== "Python"

    ``` python
    import brahe as bh

    # Custom rate limits
    config = bh.RateLimitConfig(max_per_minute=10, max_per_hour=100)
    client = bh.SpaceTrackClient("user@example.com", "password", rate_limit=config)

    # Disable rate limiting
    client = bh.SpaceTrackClient("user@example.com", "password",
                                  rate_limit=bh.RateLimitConfig.disabled())
    ```

=== "Rust"

    ``` rust
    use brahe::spacetrack::{SpaceTrackClient, RateLimitConfig};

    // Custom rate limits
    let config = RateLimitConfig { max_per_minute: 10, max_per_hour: 100 };
    let client = SpaceTrackClient::with_rate_limit(
        "user@example.com", "password", config
    );

    // Disable rate limiting
    let client = SpaceTrackClient::with_rate_limit(
        "user@example.com", "password", RateLimitConfig::disabled()
    );
    ```

For a full explanation of how rate limiting works, see [Rate Limiting](rate_limiting.md).

## Authentication

The client authenticates lazily -- credentials are sent on the first query. Call `authenticate()` explicitly to verify credentials before executing queries.

=== "Python"

    ``` python
    import brahe as bh

    client = bh.SpaceTrackClient("user@example.com", "password")

    # Explicitly authenticate to verify credentials
    client.authenticate()
    ```

=== "Rust"

    ``` rust
    use brahe::spacetrack::SpaceTrackClient;

    let client = SpaceTrackClient::new("user@example.com", "password");

    // Explicitly authenticate to verify credentials
    client.authenticate().unwrap();
    ```

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

### Raw and JSON Queries

Use `query_raw()` for non-JSON formats (TLE, CSV, KVN) and `query_json()` for generic JSON access:

=== "Python"

    ``` python
    import brahe as bh

    client = bh.SpaceTrackClient("user@example.com", "password")

    # Get TLE text directly
    query = (
        bh.SpaceTrackQuery(bh.RequestClass.GP)
        .filter("NORAD_CAT_ID", "25544")
        .format(bh.OutputFormat.TLE)
        .limit(1)
    )
    tle_text = client.query_raw(query)
    print(tle_text)

    # Get raw JSON for flexible processing
    query = (
        bh.SpaceTrackQuery(bh.RequestClass.GP)
        .filter("NORAD_CAT_ID", "25544")
        .limit(1)
    )
    json_data = client.query_json(query)
    print(json_data[0]["OBJECT_NAME"])
    ```

=== "Rust"

    ``` rust
    use brahe::spacetrack::{
        SpaceTrackClient, SpaceTrackQuery, RequestClass, OutputFormat
    };

    let client = SpaceTrackClient::new("user@example.com", "password");

    // Get TLE text directly
    let query = SpaceTrackQuery::new(RequestClass::GP)
        .filter("NORAD_CAT_ID", "25544")
        .format(OutputFormat::TLE)
        .limit(1);
    let tle_text = client.query_raw(&query).unwrap();
    println!("{}", tle_text);

    // Get raw JSON for flexible processing
    let query = SpaceTrackQuery::new(RequestClass::GP)
        .filter("NORAD_CAT_ID", "25544")
        .limit(1);
    let json_data = client.query_json(&query).unwrap();
    println!("{}", json_data[0]["OBJECT_NAME"]);
    ```

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
