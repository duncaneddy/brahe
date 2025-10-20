# Datasets

The datasets module in Brahe provides easy access to common datasources
used in astrodynamics, space mission analysis, and research. This includes
ephemeris data for Earth-orbiting satellites and locations of ground stations.

## Available Datasets

### Satellite Ephemeris

- **[CelesTrak](celestrak.md)**: Two-Line Element (TLE) sets for thousands of Earth-orbiting satellites
  - Free access, no registration required
  - Organized by satellite groups (GNSS, communications, Earth observation, etc.)
  - Updated multiple times daily
  - See [CelesTrak documentation](celestrak.md) for details

### Groundstation Networks

- **[Groundstation Datasets](groundstations.md)**: Commercial ground station locations and metadata
  - Six major provider networks (Atlas, AWS, KSAT, Leaf, SSC, Viasat)
  - Embedded data (no external files required)
  - Geographic coordinates and frequency band information
  - See [Groundstation documentation](groundstations.md) for details

## Use Cases

**Satellite Tracking**: Download current TLE data for satellites of interest and propagate their orbits

**Coverage Analysis**: Evaluate ground network coverage for satellite missions

**Mission Planning**: Assess communication opportunities using real groundstation networks

**Research**: Access historical or current ephemeris for analysis and validation

## Data Philosophy

Brahe's datasets module aims to:

- **Reduce friction**: Provide easy access to commonly needed data
- **No surprises**: Data sources are clearly documented with known limitations
- **Offline capable**: Prefer embedded data when feasible (groundstations)
- **Respect providers**: Follow best practices and rate limiting (CelesTrak)
- **Stay current**: Update data sources as the ecosystem evolves