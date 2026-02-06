# Rate Limiting

Space-Track.org enforces rate limits of **30 requests per minute** and **300 requests per hour**. Exceeding these limits results in temporary account suspension. Brahe's `RateLimitConfig` controls a sliding-window rate limiter built into `SpaceTrackClient` that automatically delays requests to stay within the configured thresholds.

By default, the client uses conservative limits of **25 requests per minute** and **250 requests per hour** (~83% of the actual limits), providing safety margin for clock drift and shared accounts. Most users do not need to configure rate limiting at all -- the defaults are applied automatically.

For the complete API reference, see the [RateLimitConfig Reference](../../library_api/spacetrack/rate_limiting.md).

## Configuration

`RateLimitConfig` supports three modes: default conservative limits, custom limits, and disabled (no limiting).

=== "Python"

    ``` python
    --8<-- "./examples/spacetrack/rate_limiting.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/spacetrack/rate_limiting.rs:4"
    ```

!!! note "Defaults Are Automatic"
    Creating a `SpaceTrackClient` without specifying a `RateLimitConfig` applies the default conservative limits (25/min, 250/hour). You only need `RateLimitConfig` if you want to change or disable the limits.

## How It Works

The rate limiter tracks request timestamps in two sliding windows (1-minute and 1-hour). Before each HTTP request, the client checks whether the configured limit has been reached in either window. If a limit would be exceeded, the calling thread sleeps until enough time has passed for the oldest request in the window to expire. This is transparent to the caller -- queries simply take longer when the limit is approached.

The limiter applies to all client operations: authentication, queries, file operations, and public file downloads.

---

## See Also

- [RateLimitConfig Reference](../../library_api/spacetrack/rate_limiting.md) -- Complete API documentation
- [Client](client.md) -- Client creation and query execution
- [Space-Track API Overview](index.md) -- Module architecture and type catalog
