# CelesTrak Data Source

[CelesTrak](https://celestrak.org) is a public source for satellite Two-Line Element (TLE) data, maintained by T.S. Kelso since 1985. It provides free, frequently updated orbital element sets for thousands of satellites, making it a useful resource for satellite tracking, orbit determination, and space situational awareness.

!!! tip "Respectful Usage"
    CelesTrak is freely available for public use, but users should be respectful of the service. Avoid excessive automated requests, and design your calls to take advantage of caching to minimize repeated queries. For large-scale or commercial applications, consider setting up a single download and local caching strategy to distribute ephemeris data internally.

## Overview

### Architecture

Brahe provides a `CelestrakClient` with a query builder pattern that mirrors the `SpaceTrackClient` interface. Both clients return `GPRecord` for GP queries, enabling code that works interchangeably with either data source.

The client supports three CelesTrak endpoints:

<div class="center-table" markdown="1">

| Endpoint | Query Constructor | Description |
|----------|-------------------|-------------|
| GP | `CelestrakQuery.gp` | General Perturbations (OMM) data |
| SupGP | `CelestrakQuery.sup_gp` | Supplemental GP data from constellation operators |
| SATCAT | `CelestrakQuery.satcat` | Satellite catalog metadata |

</div>

### Caching

To minimize load on CelesTrak's servers and improve performance, brahe implements a 6-hour cache for downloaded data:

- **Cache key**: Query URL (group, CATNR, etc.)
- **Cache duration**: 6 hours (default, configurable)
- **Cache location**: System cache directory (`~/.cache/brahe/celestrak/`)

!!! tip "Customizing Cache"
    Pass `cache_max_age=0.0` to disable caching, or a custom value in seconds to change the TTL.

### Client-Side Filtering

CelesTrak's API only supports a few server-side filters (GROUP, CATNR, NAME, INTDES). For more complex filtering, brahe provides client-side filtering using the same SpaceTrack operator syntax:

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

Client-side filters are applied after downloading the full dataset, so they work on any field in the response.

## Usage

### Querying GP Data

The most common use case is querying GP (General Perturbations) data, which returns `GPRecord` objects -- the same type used by SpaceTrack. Compact convenience methods handle the most common lookups:

```python
import brahe as bh

client = bh.celestrak.CelestrakClient()

# By satellite group
records = client.get_gp(group="stations")

# By NORAD catalog number
records = client.get_gp(catnr=25544)

# By name search
records = client.get_gp(name="ISS")

# By international designator
records = client.get_gp(intdes="1998-067A")

for rec in records:
    print(f"{rec.object_name}: inc={rec.inclination}°")
```

For complex queries with filtering, sorting, or limiting, use the query builder:

```python
query = bh.celestrak.CelestrakQuery.gp.group("stations")
records = client.query(query)
```

### Getting Raw TLE Data

For direct TLE text (e.g., for file output or custom parsing):

```python
import brahe as bh

client = bh.celestrak.CelestrakClient()
query = (
    bh.celestrak.CelestrakQuery.gp
    .catnr(25544)
    .format(bh.celestrak.CelestrakOutputFormat.THREE_LE)
)
tle_text = client.query_raw(query)
print(tle_text)
```

### Querying SATCAT Data

The SATCAT endpoint provides satellite catalog metadata:

```python
import brahe as bh

client = bh.celestrak.CelestrakClient()

# Compact method for simple lookups
records = client.get_satcat(active=True, payloads=True, on_orbit=True)

# Or use the query builder for equivalent results
query = (
    bh.celestrak.CelestrakQuery.satcat
    .active(True)
    .payloads(True)
    .on_orbit(True)
)
records = client.query(query)

for rec in records:
    print(f"{rec.object_name}: {rec.owner}, launched {rec.launch_date}")
```

### Downloading to File

Save query results directly to a file:

```python
import brahe as bh

client = bh.celestrak.CelestrakClient()
query = (
    bh.celestrak.CelestrakQuery.gp
    .group("stations")
    .format(bh.celestrak.CelestrakOutputFormat.THREE_LE)
)
client.download(query, "stations.txt")
```

### Interoperability with SpaceTrack

Both CelesTrak and SpaceTrack return `list[GPRecord]` for GP queries, so downstream processing code works with either source:

```python
import brahe as bh

def process_records(records):
    """Works with records from either CelesTrak or SpaceTrack."""
    for rec in records:
        print(f"{rec.object_name}: {rec.norad_cat_id}")

# From CelesTrak (no authentication required)
ct_client = bh.celestrak.CelestrakClient()
ct_records = ct_client.get_gp(group="stations")
process_records(ct_records)

# From SpaceTrack (requires authentication)
# st_client = bh.spacetrack.SpaceTrackClient(username, password)
# st_query = bh.spacetrack.SpaceTrackQuery(bh.spacetrack.RequestClass.GP)
# st_records = st_client.query_gp(st_query)
# process_records(st_records)
```

## Satellite Groups

CelesTrak organizes satellites into logical groups accessible via simple names. These groups are updated as active constellations evolve. It is best to download TLEs by group name rather than ID to minimize the number of distinct requests.

### Temporal Groups

<div class="center-table" markdown="1">

| Group | Description |
|-------|-------------|
| `active` | All active satellites |
| `last-30-days` | Recently launched satellites |
| `tle-new` | Newly added TLEs (last 15 days) |

</div>

### Communications

<div class="center-table" markdown="1">

| Group | Description |
|-------|-------------|
| `starlink` | SpaceX Starlink constellation |
| `oneweb` | OneWeb constellation |
| `kuiper` | Amazon Kuiper constellation |
| `intelsat` | Intelsat satellites |
| `eutelsat` | Eutelsat constellation |
| `orbcomm` | ORBCOMM constellation |
| `telesat` | Telesat constellation |
| `globalstar` | Globalstar constellation |
| `iridium-NEXT` | Iridium constellation |
| `qianfan` | Qianfan constellation |
| `hulianwang` | Hulianwang Digui constellation |

</div>

### Earth Observation

<div class="center-table" markdown="1">

| Group | Description |
|-------|-------------|
| `weather` | Weather satellites (NOAA, GOES, Metop, etc.) |
| `earth-resources` | Earth observation (Landsat, Sentinel, etc.) |
| `planet` | Planet Labs imaging satellites |
| `spire` | Spire Global satellites |

</div>

### Navigation

<div class="center-table" markdown="1">

| Group | Description |
|-------|-------------|
| `gnss` | All navigation satellites (GPS, GLONASS, Galileo, BeiDou, QZSS, IRNSS) |
| `gps-ops` | Operational GPS satellites only |
| `glonass-ops` | Operational GLONASS satellites only |
| `galileo` | European Galileo constellation |
| `beidou` | Chinese BeiDou/COMPASS constellation |
| `sbas` | Satellite-Based Augmentation System (WAAS/EGNOS/MSAS) |

</div>

### Scientific and Special Purpose

<div class="center-table" markdown="1">

| Group | Description |
|-------|-------------|
| `science` | Scientific research satellites |
| `noaa` | NOAA satellites |
| `stations` | Space stations (ISS, Tiangong) |
| `analyst` | Analyst satellites (tracking placeholder IDs) |
| `visual` | 100 (or so) brightest objects |
| `gpz` | Geostationary Protected Zone |
| `gpz-plus` | Geostationary Protected Zone Plus |

</div>

**Note**: Group names and contents evolve as missions launch, deorbit, or change status. Visit [CelesTrak GP Element Sets](https://celestrak.org/NORAD/elements) for the current complete list.

---

## See Also

- [Ephemeris Data Sources](index.md) - Shared types, operators, and source comparison
- [Two-Line Elements](../orbits/two_line_elements.md) - TLE and 3LE format details
- [CelesTrak API Reference](../../library_api/ephemeris/celestrak.md) - Class and method documentation
