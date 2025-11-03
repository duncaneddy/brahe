# Datasets

The datasets module in Brahe provides easy access to common datasources
used in astrodynamics, space mission analysis, and research. This includes
ephemeris data for Earth-orbiting satellites and locations of ground stations.

## Data Philosophy

Brahe's datasets module aims to:

- **Reduce friction**: Provide easy access to commonly needed data
- **No surprises**: Data sources are clearly documented with known limitations
- **Offline capable**: Prefer embedded data when feasible (groundstations)
- **Respect providers**: Follow best practices and rate limiting (CelesTrak)
- **Stay current**: Update data sources as the ecosystem evolves