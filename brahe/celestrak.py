"""
Celestrak Module

Provides a client for querying satellite catalog data from Celestrak.

This module provides:
- CelestrakClient: HTTP client with caching (no authentication required)
- CelestrakQuery: Fluent query builder for GP, SupGP, and SATCAT endpoints
- CelestrakOutputFormat: Output format enum
- CelestrakQueryType: Query endpoint type enum
- SupGPSource: Supplemental GP data source enum
- CelestrakSATCATRecord: Typed SATCAT response record

GP queries return the same GPRecord type as the SpaceTrack module,
enabling interoperability between both data sources.

Example:
    ```python
    import brahe as bh

    client = bh.celestrak.CelestrakClient()

    # Compact convenience methods (most common use cases)
    records = client.get_gp(group="stations")
    records = client.get_gp(catnr=25544)
    propagator = client.get_sgp_propagator(catnr=25544, step_size=60.0)

    # Query builder for complex queries with filtering/sorting/limiting
    from brahe.spacetrack import operators as op

    query = (
        bh.celestrak.CelestrakQuery.gp
        .group("active")
        .filter("OBJECT_TYPE", op.not_equal("DEBRIS"))
        .filter("INCLINATION", op.greater_than("50"))
    )
    records = client.query(query)
    ```
"""

from brahe._brahe import (
    # Client
    CelestrakClient,
    CelestrakOutputFormat,
    # Query builder
    CelestrakQuery,
    # Enums
    CelestrakQueryType,
    # Response types
    CelestrakSATCATRecord,
    SupGPSource,
)

__all__ = [
    # Client
    "CelestrakClient",
    "CelestrakOutputFormat",
    # Query builder
    "CelestrakQuery",
    # Enums
    "CelestrakQueryType",
    # Response types
    "CelestrakSATCATRecord",
    "SupGPSource",
]
