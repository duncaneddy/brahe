# Datasets Module

The datasets module provides convenient access to satellite ephemeris data and groundstation locations from multiple sources. It handles downloading, parsing, and format conversion automatically.

## Module Overview

The module is organized by data source, with each source providing a consistent API:

- **`brahe.celestrak`**: CelesTrak satellite ephemeris data (client, query builder, and response types)
- **`brahe.datasets.groundstations`**: Curated groundstation location datasets
- **`brahe.datasets.naif`**: NASA JPL NAIF planetary ephemeris kernels

## Submodules

- [CelesTrak](celestrak.md) - CelesTrak client, query builder, and response types
- [Groundstation Functions](groundstations.md) - Groundstation location datasets
- [NAIF Functions](naif.md) - Planetary ephemeris kernels from NASA JPL

---

## See Also

- [Datasets Overview](../../learn/datasets/index.md) - Understanding datasets module
- [CelesTrak Details](../../learn/datasets/celestrak.md) - CelesTrak data source specifics
