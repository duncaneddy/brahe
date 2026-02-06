"""
SpaceTrack Module

Provides a client for querying satellite catalog data from Space-Track.org.

This module provides:
- SpaceTrackClient: HTTP client with authentication
- SpaceTrackQuery: Fluent query builder
- Request class/controller/format enums
- Operator functions for query filters

Example:
    ```python
    import brahe as bh
    from brahe.spacetrack import operators as op

    client = bh.SpaceTrackClient("user@example.com", "password")

    query = (
        bh.SpaceTrackQuery(bh.RequestClass.GP)
        .filter("NORAD_CAT_ID", "25544")
        .order_by("EPOCH", bh.SortOrder.DESC)
        .limit(1)
    )
    records = client.query_gp(query)
    print(records[0].object_name)
    ```
"""

from brahe._brahe import (
    # Enums
    RequestController,
    RequestClass,
    SortOrder,
    OutputFormat,
    # Query builder
    SpaceTrackQuery,
    # Client
    SpaceTrackClient,
    # Response types
    GpRecord,
    SatcatRecord,
    # Operator functions
    spacetrack_greater_than,
    spacetrack_less_than,
    spacetrack_not_equal,
    spacetrack_inclusive_range,
    spacetrack_like,
    spacetrack_startswith,
    spacetrack_now,
    spacetrack_now_offset,
    spacetrack_null_val,
    spacetrack_or_list,
)


class _OperatorsNamespace:
    """SpaceTrack query operator functions.

    Provides operator functions for constructing SpaceTrack query filters.
    These functions generate operator-prefixed strings for use in filter values.

    Example:
        ```python
        from brahe.spacetrack import operators as op

        op.greater_than("25544")         # ">25544"
        op.less_than("0.01")             # "<0.01"
        op.inclusive_range("1", "100")    # "1--100"
        op.now_offset(-7)                # "now-7"
        ```
    """

    greater_than = staticmethod(spacetrack_greater_than)
    less_than = staticmethod(spacetrack_less_than)
    not_equal = staticmethod(spacetrack_not_equal)
    inclusive_range = staticmethod(spacetrack_inclusive_range)
    like = staticmethod(spacetrack_like)
    startswith = staticmethod(spacetrack_startswith)
    now = staticmethod(spacetrack_now)
    now_offset = staticmethod(spacetrack_now_offset)
    null_val = staticmethod(spacetrack_null_val)
    or_list = staticmethod(spacetrack_or_list)


# Create operators namespace instance
operators = _OperatorsNamespace()

__all__ = [
    # Enums
    "RequestController",
    "RequestClass",
    "SortOrder",
    "OutputFormat",
    # Query builder
    "SpaceTrackQuery",
    # Client
    "SpaceTrackClient",
    # Response types
    "GpRecord",
    "SatcatRecord",
    # Operators namespace
    "operators",
]
