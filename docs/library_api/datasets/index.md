# Datasets Module

The datasets module provides convenient access to groundstation locations and planetary ephemeris kernels.

!!! tip "Satellite Ephemeris Data"
    For satellite ephemeris API clients (CelesTrak, Space-Track), see [Ephemeris Data Sources](../ephemeris/index.md).

## Module Overview

The module is organized by data source:

- **`brahe.datasets.groundstations`**: Curated groundstation location datasets
- **`brahe.datasets.naif`**: NASA JPL NAIF planetary ephemeris kernels
- **`brahe.datasets.gcat`**: GCAT satellite catalogs (SATCAT, PSATCAT)

## Submodules

- [Groundstation Functions](groundstations.md) - Groundstation location datasets
- [NAIF Functions](naif.md) - Planetary ephemeris kernels from NASA JPL
- [GCAT Functions](gcat.md) - GCAT satellite catalog access

---

## See Also

- [Datasets Overview](../../learn/datasets/index.md) - Understanding datasets module
- [Ephemeris Data Sources](../ephemeris/index.md) - CelesTrak and Space-Track API clients
