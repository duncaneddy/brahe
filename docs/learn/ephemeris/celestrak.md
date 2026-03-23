# CelesTrak Data Source

!!! tip "Respectful Usage"
    CelesTrak is freely available for public use, but users should be respectful of the service. Avoid excessive automated requests, and take advantage of caching to minimize repeated queries. For large-scale or commercial applications, consider downloading once and distributing data internally.

CelesTrak is the simplest entry point for satellite ephemeris data: it is free, requires no account, and
provides frequently updated orbital element sets for thousands of satellites. Maintained by T.S. Kelso
since 1985, it is a widely used resource for satellite tracking and space situational awareness.

## How It Works

Brahe provides a `CelestrakClient` that talks to celestrak.org and returns structured records. For GP
(General Perturbations) queries the client returns `GPRecord` objects -- the same type used by the
SpaceTrack client, so downstream code works interchangeably with either data source. The client also
supports Supplemental GP data from constellation operators (`CelestrakQuery.sup_gp`) and satellite
catalog metadata (`CelestrakQuery.satcat`).

To minimize load on CelesTrak's servers and improve performance, the client caches downloaded data for
6 hours. Cache files are stored in the system cache directory (`~/.cache/brahe/celestrak/`) and are
keyed by query URL.

!!! tip "Customizing Cache"
    Pass `cache_max_age=0.0` to disable caching, or a custom value in seconds to change the TTL.

## Querying Satellite Data

The client offers two levels of API. Compact convenience methods handle the most common lookups --
pass a catalog number, group name, object name, or international designator directly to `get_gp` or
`get_satcat`. For more control, the `CelestrakQuery` builder lets you compose queries with filtering,
sorting, output format selection, and result limits.

The following example retrieves GP data for a single satellite by NORAD catalog number:

=== "Python"
    ``` python
    --8<-- "./examples/datasets/celestrak_get_by_id.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/datasets/celestrak_get_by_id.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/datasets/celestrak_get_by_id.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/datasets/celestrak_get_by_id.rs.txt"
        ```

Other common lookup patterns include querying by group name (`group="stations"`), object name
(`name="ISS"`), or international designator (`intdes="1998-067A"`). All of these are available as
keyword arguments to `get_gp` or as builder methods on `CelestrakQuery.gp`.

## Client-Side Filtering

CelesTrak's API only supports a few server-side filters (group, catalog number, name, international
designator). For more complex filtering, brahe provides client-side operators that use the same syntax
as the SpaceTrack query interface. These filters are applied after downloading the full dataset, so
they work on any field in the response:

```python
import brahe as bh
from brahe.spacetrack import operators as op

client = bh.celestrak.CelestrakClient()
query = (
    bh.celestrak.CelestrakQuery.gp
    .group("stations")
    .filter("INCLINATION", op.greater_than("50"))
    .filter("OBJECT_TYPE", op.not_equal("DEBRIS"))
    .order_by("INCLINATION", False)
    .limit(10)
)
records = client.query(query)
```

## Satellite Groups

CelesTrak organizes satellites into logical groups accessible via simple names such as `stations`,
`starlink`, `gnss`, `active`, and `weather`. Groups span several categories including temporal
(`active`, `last-30-days`), communications (`starlink`, `oneweb`, `iridium-NEXT`), navigation
(`gnss`, `gps-ops`, `galileo`), earth observation (`weather`, `planet`, `earth-resources`), and
scientific or special purpose (`science`, `analyst`, `visual`). Group names and contents evolve as
missions launch, deorbit, or change status.

For a complete listing of available groups, see the
[CelesTrak API Reference](../../library_api/ephemeris/celestrak.md#satellite-groups) or visit
[CelesTrak GP Element Sets](https://celestrak.org/NORAD/elements).

---

## See Also

- [Ephemeris Data Sources](index.md) -- Shared types, operators, and source comparison
- [Two-Line Elements](../orbits/two_line_elements.md) -- TLE and 3LE format details
- [CelesTrak API Reference](../../library_api/ephemeris/celestrak.md) -- Class and method documentation
