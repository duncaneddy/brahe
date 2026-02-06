# Space-Track Module

The spacetrack module provides a typed client and fluent query builder for accessing satellite catalog data from [Space-Track.org](https://www.space-track.org).

## Submodules

- [Client](client.md) -- `SpaceTrackClient` for authentication and query execution
- [Query Builder](query.md) -- `SpaceTrackQuery` fluent builder
- [Responses](responses.md) -- `GPRecord`, `SATCATRecord`, `FileShareFileRecord`, `FolderRecord`, and `SPEphemerisFileRecord` typed response structs
- [Enumerations](enums.md) -- `RequestController`, `RequestClass`, `SortOrder`, `OutputFormat`
- [Operators](operators.md) -- Filter operator functions
- [Rate Limiting](rate_limiting.md) -- `RateLimitConfig` rate limit configuration

---

## See Also

- [Space-Track API Overview](../../learn/spacetrack/index.md) -- Conceptual introduction and module architecture
- [Query Builder Guide](../../learn/spacetrack/query_builder.md) -- Building queries with examples
- [Client Guide](../../learn/spacetrack/client.md) -- Authentication and query execution patterns
- [File Operations Guide](../../learn/spacetrack/file_operations.md) -- FileShare, SP Ephemeris, and Public Files
