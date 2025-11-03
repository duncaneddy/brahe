# CelesTrak Data Source

[CelesTrak](https://celestrak.org) is a public source for satellite Two-Line Element (TLE) data, maintained by T.S. Kelso since 1985. It provides free, frequently updated orbital element sets for thousands of satellites, making it a useful resource for satellite tracking, orbit determination, and space situational awareness.

!!! tip "Respectful Usage"
    CelesTrak is freely available for public use, but users should be respectful of the service. Avoid excessive automated requests, and design your calls to take advantage of caching to minimize repeated queries. For large-scale or commercial applications, consider setting up a single download and local caching strategy to disribute ephemeris data internally.

## Overview

### What is CelesTrak?

CelesTrak is a public data source for satellite orbital elements, maintained by Dr. T.S. Kelso since 1985. It provides free, frequently updated Two-Line Element (TLE) data for thousands of satellites, making it an essential resource for satellite tracking, orbit determination, and space situational awareness.

### TLE Format

Two-Line Elements (TLEs) are a compact text format for encoding satellite orbital parameters compatible with the SGP4/SDP4 propagation models. For more information on TLEs, see the [Two-Line Elements](../orbits/two_line_elements.md) documentation.

### Caching

To minimize load on CelesTrak's servers and improve performance, brahe implements a 6-hour cache for downloaded data:

- **Cache key**: Satellite group name (e.g., "starlink", "stations")
- **Cache duration**: 6 hours (default, configurable)
- **Cache location**: System temp directory

When you request a satellite by ID or name with a group hint, brahe checks if that group was recently downloaded and uses cached data if available. This is much faster and more respectful than making individual requests.

!!! tip "Customizing Cache"
    See the [Caching](../utilities/caching.md) documentation for details on customizing cache behavior.

## Usage

### Getting Ephemeris by Group

The most efficient way to get TLE data is by downloading entire groups. This minimizes API requests and leverages caching:

=== "Python"

    ``` python
    --8<-- "./examples/datasets/celestrak_get_group.py:12"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/datasets/celestrak_get_group.rs:9"
    ```

### Getting a Satellite by ID

To get a specific satellite, provide its NORAD ID. **Always include a group hint** to enable cache-efficient lookups:

=== "Python"

    ``` python
    --8<-- "./examples/datasets/celestrak_get_by_id.py:12"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/datasets/celestrak_get_by_id.rs:9"
    ```

!!! tip "Cache-Efficient Pattern"
    The most efficient workflow is:

    1. Download the group once: `get_ephemeris("stations")`
    2. Query specific satellites with the group hint: `get_tle_by_id(25544, "stations")`

    This pattern uses cached data and avoids redundant downloads.

### Converting to Propagators

For most applications, you'll want to convert TLEs directly to SGP propagators. Brahe provides convenience functions that do this in one step:

=== "Python"

    ``` python
    --8<-- "./examples/datasets/celestrak_as_propagator.py:12"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/datasets/celestrak_as_propagator.rs:9"
    ```

### Getting by Name

You can also search for satellites by name. This performs a cascading search across groups:

=== "Python"

    ``` python
    --8<-- "./examples/datasets/celestrak_get_by_name.py:12"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/datasets/celestrak_get_by_name.rs:9"
    ```

!!! note "Name Matching"
    Name searches are case-insensitive and support partial matches. If multiple satellites match, the function returns the first match. 

## Satellite Groups

CelesTrak organizes satellites into logical groups accessible via simple names. These groups are updated as active constellations evolve. It is best to download TLEs by group name rather than ID to minimize the number of distinct requests.

### Temporal Groups

| Group | Description |
|-------|-------------|
| `active` | All active satellites |
| `last-30-days` | Recently launched satellites |
| `tle-new` | Newly added TLEs (last 15 days) |

### Communications

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

### Earth Observation

| Group | Description |
|-------|-------------|
| `weather` | Weather satellites (NOAA, GOES, Metop, etc.) |
| `earth-resources` | Earth observation (Landsat, Sentinel, etc.) |
| `planet` | Planet Labs imaging satellites |
| `spire` | Spire Global satellites |

### Navigation

| Group | Description |
|-------|-------------|
| `gnss` | All navigation satellites (GPS, GLONASS, Galileo, BeiDou, QZSS, IRNSS) |
| `gps-ops` | Operational GPS satellites only |
| `glonass-ops` | Operational GLONASS satellites only |
| `galileo` | European Galileo constellation |
| `beidou` | Chinese BeiDou/COMPASS constellation |
| `sbas` | Satellite-Based Augmentation System (WAAS/EGNOS/MSAS) |

### Scientific and Special Purpose

| Group | Description |
|-------|-------------|
| `science` | Scientific research satellites |
| `noaa` | NOAA satellites |
| `stations` | Space stations (ISS, Tiangong) |
| `analyst` | Analyst satellites (tracking placeholder IDs) |
| `visual` | 100 (or so) brightest objects |
| `gpz` | Geostationary Protected Zone |
| `gpz-plus` | Geostationary Protected Zone Plus |

**Note**: Group names and contents evolve as missions launch, deorbit, or change status. Visit [CelesTrak GP Element Sets](https://celestrak.org/NORAD/elements) for the current complete list.

## See Also

- [Datasets Overview](index.md) - Understanding satellite ephemeris datasets
- [Two-Line Elements](../orbits/two_line_elements.md) - TLE and 3LE format details
- [Downloading TLE Data](../../examples/downloading_tle_data.md) - Practical examples
- [CelesTrak API Reference](../../library_api/datasets/celestrak.md) - Function documentation
