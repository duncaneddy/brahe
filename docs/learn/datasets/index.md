# Datasets

The datasets module in Brahe provides easy access to common static data sources used in astrodynamics, space mission analysis, and research. This includes planetary ephemeris kernels and locations of ground stations.

!!! tip "Satellite Ephemeris Data"
    For satellite ephemeris API clients (CelesTrak, Space-Track), see [Ephemeris Data Sources](../ephemeris/index.md).

## Overview

Working with satellite and planetary data typically requires gathering information from multiple sources:

- **Planetary ephemeris** (DE kernels) for high-precision solar system body positions
- **Groundstation locations** for computing contact opportunities
- **Spherical harmonic gravity models** for high-fidelity central-body force modeling

Brahe's datasets module centralizes access to these data sources, handling the details of fetching, parsing, and caching so you can focus on analysis rather than data wrangling.

## Available Data Sources

### [NAIF](naif.md)

[NASA JPL's NAIF archive](https://naif.jpl.nasa.gov/) provides high-precision planetary ephemeris kernels. The brahe interface supports:

- **DE kernel downloads**: Fetch Development Ephemeris binary files (SPK format)
- **Automatic caching**: Downloaded kernels are cached permanently
- **Multiple versions**: Support for DE430, DE440, DE442 and variants

**Best for**: High-precision planetary ephemeris, interplanetary mission analysis

### [GCAT Satellite Catalogs](gcat.md)

[Jonathan McDowell's GCAT](https://planet4589.org/space/gcat/) provides comprehensive metadata for all cataloged artificial space objects. The brahe interface supports:

- **SATCAT**: Physical, orbital, and administrative properties for all cataloged objects
- **PSATCAT**: Payload-specific mission metadata and UN registry information
- **Automatic caching**: File-based caching with configurable 24-hour TTL
- **Search and filtering**: Name search, type/status/owner filters, orbital range filters

**Best for**: Satellite catalog research, constellation analysis, historical studies

### [Star Catalogs](star_catalogs.md)

Fixed-epoch star catalogs for reference-frame realization and star-based attitude determination. The brahe interface supports:

- **FK5**: 1,535 fundamental stars, J2000.0
- **Hipparcos**: ~118,000 stars, ICRS at epoch J1991.25
- **Tycho-2**: ~2.54 million stars, ICRS
- **Never-stale caching**: Cached copies never expire by default, since published catalogs do not change
- **Proper motion**: Propagate catalog positions to any epoch via `radec_at_epoch`

**Best for**: Star-based attitude determination, reference-frame realization, astrometric cross-matching

### [ICGEM Gravity Models](icgem.md)

The [International Centre for Global Earth Models (ICGEM)](https://icgem.gfz.de) hosts spherical harmonic gravity models for Earth and other solar system bodies (Moon, Mars, Venus, Ceres, asteroids, …). The brahe interface supports:

- **Catalog listing**: Discover available models per body, with degree and publication year
- **On-demand download**: Fetch `.gfc` files into a local cache on first use
- **Index TTL + stale fallback**: 30-day index refresh with graceful degradation when offline
- **Propagator integration**: Reference a downloaded model directly from `GravityModelType.icgem(body, name)`

**Best for**: High-fidelity gravity modeling beyond the three packaged Earth models, gravity fields for other bodies, and pinning a specific published model version

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

- [Ephemeris Data Sources](../ephemeris/index.md) - CelesTrak and Space-Track API clients
- [NAIF Ephemeris Kernels](naif.md) - Planetary ephemeris data
- [GCAT Satellite Catalogs](gcat.md) - GCAT SATCAT and PSATCAT catalogs
- [Star Catalogs](star_catalogs.md) - FK5, Hipparcos, and Tycho-2 fixed-epoch star catalogs
- [ICGEM Gravity Models](icgem.md) - Spherical harmonic gravity model catalog
- [Groundstation Datasets](groundstations.md) - Ground facility locations
- [Datasets API Reference](../../library_api/datasets/index.md) - Complete function documentation
