# Datasets

The datasets module in Brahe provides easy access to common data sources used in astrodynamics, space mission analysis, and research. This includes ephemeris data for Earth-orbiting satellites and locations of ground stations.

## Overview

Working with satellite data typically requires gathering information from multiple sources:

- **Orbital elements** (TLEs) for satellite tracking and propagation
- **Groundstation locations** for computing contact opportunities
- **Satellite metadata** for mission planning and analysis

Brahe's datasets module centralizes access to these data sources, handling the details of fetching, parsing, and caching so you can focus on analysis rather than data wrangling.

## Available Data Sources

### [CelesTrak](celestrak.md)

[CelesTrak](https://celestrak.org) provides Two-Line Element (TLE) data for thousands of Earth-orbiting satellites. The brahe interface supports:

- **Group downloads**: Fetch entire satellite constellations (Starlink, OneWeb, GPS, etc.)
- **Individual lookups**: Get specific satellites by NORAD ID or name
- **Direct propagation**: Convert TLEs to SGP4 propagators in one step

**Best for**: Satellite analysis, orbit propagation, space situational awareness

### [Groundstations](groundstations.md)

Embedded GeoJSON data for commercial groundstation networks. Includes 6 major providers:

- **Atlas Space Operations**
- **Amazon Web Services Ground Station**
- **Kongsberg Satellite Services (KSAT)**
- **Leaf Space**
- **Swedish Space Corporation (SSC)**
- **Viasat**

**Best for**: Contact opportunity analysis, network planning, coverage studies

## Data Philosophy

Brahe's datasets module aims to:

- **Reduce friction**: Provide easy access to commonly needed data
- **No surprises**: Data sources are clearly documented with known limitations
- **Offline capable**: Prefer embedded data when feasible
- **Respect providers**: Follow best practices and rate limiting
- **Stay current**: Update data sources as the ecosystem evolves

---

## See Also

- [CelesTrak Data Source](celestrak.md) - TLE ephemeris data
- [Groundstation Datasets](groundstations.md) - Ground facility locations
- [Datasets API Reference](../../library_api/datasets/index.md) - Complete function documentation