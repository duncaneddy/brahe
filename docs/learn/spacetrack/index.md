# SpaceTrack API

Space-Track.org is the U.S. Space Command's public catalog of tracked space objects. It provides orbital elements (TLEs/GPs), satellite catalog data, conjunction assessments, and decay predictions for over 50,000 objects in Earth orbit.

Brahe's SpaceTrack module provides a type-safe client for querying this data, with built-in rate limiting and direct integration with SGP propagators.

## Key Concepts

### Authentication

Space-Track requires authentication for all API access. You need a free account from [space-track.org](https://www.space-track.org/auth/createAccount).

```python
import brahe as bh

# Client authenticates on creation
client = bh.SpaceTrackClient("username", "password")
```

### Rate Limiting

Space-Track enforces rate limits: 30 requests/minute and 300 requests/hour. The client handles this automatically, queuing requests when necessary.

### GP vs TLE

**General Perturbations (GP)** data is the modern format, providing orbital elements in a structured JSON format with additional metadata. **Two-Line Elements (TLE)** is the legacy text format. The GP endpoint is recommended for new applications.

### NORAD Catalog ID

Every tracked object has a unique NORAD Catalog ID (also called SATCAT ID or NORAD ID). This 5-digit identifier is the primary key for querying specific objects. For example, the ISS is `25544`.

## Module Contents

- **[Client Usage](client.md)** - Creating clients and making queries
- **[Query Filters](queries.md)** - Building filtered queries
- **[Propagator Integration](propagators.md)** - Converting GP data to SGP propagators

## Quick Example

```python
import brahe as bh

# Create authenticated client
client = bh.SpaceTrackClient("username", "password")

# Get latest ISS orbital elements
records = client.gp(norad_cat_id=25544, limit=1)
print(f"ISS epoch: {records[0]['EPOCH']}")

# Get as SGP propagator for orbit propagation
propagators = client.gp_as_propagators(
    step_size=60.0,
    norad_cat_id=25544,
    limit=1
)
```

## See Also

- [SpaceTrack API Reference](../../library_api/spacetrack/index.md) - Complete API documentation
- [Space-Track.org](https://www.space-track.org) - Official website and documentation
