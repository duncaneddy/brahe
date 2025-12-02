# SpaceTrack

The SpaceTrack module provides a Python client for accessing the [Space-Track.org](https://www.space-track.org) API, enabling retrieval of orbital element data, satellite catalog information, and other space surveillance data.

## Overview

- **[SpaceTrackClient](client.md)** - Authenticated API client for querying data
- **[Record Classes](records.md)** - Typed record classes returned by client queries

## Key Features

- Automatic authentication and session management
- Built-in rate limiting (30 requests/minute, 300 requests/hour)
- Typed record classes with property access and dictionary conversion
- Direct conversion to SGP propagators for orbit propagation

## Quick Example

```python
import brahe as bh

# Create client with Space-Track credentials
client = bh.SpaceTrackClient("username", "password")

# Query GP data for ISS
records = client.gp(norad_cat_id=25544, limit=1)
print(records[0].object_name)      # Property access
print(records[0].inclination)      # Orbital elements
print(records[0].as_dict())        # Dictionary conversion

# Convert to SGP propagator
propagators = client.gp_as_propagators(60.0, norad_cat_id=25544, limit=1)
```

## See Also

- [SpaceTrack User Guide](../../learn/spacetrack/index.md) - Conceptual introduction and usage examples
- [Query Filters](../../learn/spacetrack/queries.md) - Query operators and filtering
- [Propagator Integration](../../learn/spacetrack/propagators.md) - Converting records to propagators
