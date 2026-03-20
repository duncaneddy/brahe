# Space-Track API

!!! info "Account Required"
    Space-Track.org requires a free account. Register at [https://www.space-track.org/auth/createAccount](https://www.space-track.org/auth/createAccount) to obtain credentials.

[Space-Track.org](https://www.space-track.org) is the public interface to the US Space Command satellite catalog. It provides authoritative orbital data (general perturbations records), satellite catalog metadata, conjunction data messages, and decay/re-entry predictions. Unlike other public sources, Space-Track offers full server-side filtering, data types such as CDMs and decay predictions, and file-share access -- all through a REST API that Brahe wraps with a typed client and fluent query builder.

## How It Works

`SpaceTrackClient` manages authentication and HTTP communication with Space-Track.org. You create a client with your credentials, build a query describing the data you want, and execute it through the client. Authentication is lazy: credentials are sent automatically on the first query unless you call `authenticate()` explicitly.

`SpaceTrackQuery` constructs API requests using a fluent builder pattern. Each builder method -- `filter`, `order_by`, `limit`, `format`, and others -- returns a new query instance, so queries are immutable and can be reused or extended without side effects. Once built, the query produces the URL path that the client sends to Space-Track.

The client includes a built-in rate limiter that prevents exceeding Space-Track's request policies. Queries return typed results when using JSON format: `GPRecord` for general perturbations data, `SATCATRecord` for satellite catalog metadata, or raw strings for non-JSON formats and other request classes.

## Key Concepts

**Request classes.** Space-Track organizes its data into request classes, each representing a distinct data category. GP contains general perturbations orbital elements, SATCAT holds satellite catalog metadata, Decay and TIP provide re-entry predictions, and CDMPublic offers conjunction data messages. Each request class belongs to a specific API controller (endpoint namespace), but the query builder selects the correct controller automatically based on the request class you choose.

**Query builder pattern.** All query construction happens through the builder. Filters narrow results by field values, ordering controls sort direction, and limit/offset enable pagination. The builder also supports selecting specific fields, requesting metadata, and toggling distinct or empty-result behavior. Because each method returns a new instance, you can fork a base query into multiple specialized queries without mutation.

**Output formats.** Space-Track supports several output formats: JSON (the default), TLE, 3LE, CSV, XML, and KVN. Typed query methods (`query_gp`, `query_satcat`) require JSON format to deserialize into structured records. For other formats, use `query_raw()` which returns the response body as a plain string.

**Rate limiting.** Space-Track enforces request rate limits to protect service availability. The client applies conservative defaults (25 requests per minute, 250 per hour) that stay well within these policies. You can adjust or disable the rate limiter through `RateLimitConfig` if your use case requires different thresholds.

## Subpages

- [Query Builder](query_builder.md) -- Building queries with filters, ordering, and output formats
- [Common Queries](common_queries.md) -- Practical query patterns for GP, SATCAT, and Decay data
- [Conjunction Data Messages](cdm.md) -- Querying CDM collision risk assessments
- [Client](client.md) -- Authentication, query execution, and response handling
- [File Operations](file_operations.md) -- FileShare, SP Ephemeris, and Public Files
- [Rate Limiting](rate_limiting.md) -- Configuring request rate limits

---

## See Also

- [Space-Track API Reference](../../../library_api/ephemeris/spacetrack/index.md) -- Complete function documentation
- [CelesTrak Data Source](../celestrak.md) -- Alternative TLE data source (no account required)
