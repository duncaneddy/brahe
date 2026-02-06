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

    # Query GP data for the stations group
    query = bh.celestrak.CelestrakQuery.gp().group("stations")
    records = client.query_gp(query)
    for r in records:
        print(f"{r.object_name}: inc={r.inclination}")

    # Query with client-side filtering
    from brahe.spacetrack import operators as op

    query = (
        bh.celestrak.CelestrakQuery.gp()
        .group("active")
        .filter("OBJECT_TYPE", op.not_equal("DEBRIS"))
        .filter("INCLINATION", op.greater_than("50"))
    )
    records = client.query_gp(query)
    ```
"""

from brahe._brahe import (
    # Enums
    CelestrakQueryType,
    CelestrakOutputFormat,
    SupGPSource,
    # Query builder
    CelestrakQuery,
    # Client
    CelestrakClient,
    # Response types
    CelestrakSATCATRecord,
)

__all__ = [
    # Enums
    "CelestrakQueryType",
    "CelestrakOutputFormat",
    "SupGPSource",
    # Query builder
    "CelestrakQuery",
    # Client
    "CelestrakClient",
    # Response types
    "CelestrakSATCATRecord",
]
