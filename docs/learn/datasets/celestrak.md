# CelesTrak Data Source

[CelesTrak]((https://celestrak.org)) is a public source for satellite Two-Line Element (TLE) data, maintained by T.S. Kelso since 1985. It provides free, frequently updated orbital element sets for thousands of satellites, making it a useful resource for satellite tracking, orbit determination, and space situational awareness.

!!! tip "Respectful Usage"
    CelesTrak is freely available for public use, but users should be respectful of the service. Avoid excessive automated requests, and design your calls to take advantage of caching to minimize repeated queries. For large-scale or commercial applications, consider setting up a single download and local caching strategy to disribute ephemeris data internally.

## Overview

## Usage 

## Satellite Groups

CelesTrak organizes satellites into logical groups accessible via simple names. These groups are updated as active constellations evolve. It is best to download TLEs by group name rather than ID to minimize the number of distinct requests.

### Temporal Groups

| Group | Description | Typical Count |
|-------|-------------|---------------|
| `active` | All active satellites | ~5,000+ |
| `last-30-days` | Recently launched satellites | 20-100 (varies) |
| `tle-new` | Newly added TLEs (last 15 days) | Variable |

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
