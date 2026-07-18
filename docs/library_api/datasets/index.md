# Datasets Module

The datasets module provides convenient access to groundstation locations and planetary ephemeris kernels.

!!! tip "Satellite Ephemeris Data"
    For satellite ephemeris API clients (CelesTrak, Space-Track), see [Ephemeris Data Sources](../ephemeris/index.md).

## Module Overview

The module is organized by data source:

- **`brahe.datasets.groundstations`**: Curated groundstation location datasets
- **`brahe.datasets.ssn_sensors`**: Vallado SSN sensor site dataset
- **`brahe.datasets.naif`**: NASA JPL NAIF planetary ephemeris kernels
- **`brahe.datasets.gcat`**: GCAT satellite catalogs (SATCAT, PSATCAT)
- **`brahe.datasets.star_catalogs`**: Fixed-epoch star catalogs (FK5, Hipparcos, Tycho-2)
- **`brahe.datasets.icgem`**: ICGEM spherical harmonic gravity models (Earth + celestial bodies)

## Submodules

- [Groundstation Functions](groundstations.md) - Groundstation location datasets
- [SSN Sensor Functions](ssn_sensors.md) - Vallado SSN sensor site dataset
- [NAIF Functions](naif.md) - Planetary ephemeris kernels from NASA JPL
- [GCAT Functions](gcat.md) - GCAT satellite catalog access
- [Star Catalog Functions](star_catalogs.md) - FK5, Hipparcos, and Tycho-2 star catalog access
- [ICGEM Functions](icgem.md) - ICGEM spherical harmonic gravity models

---

## See Also

- [Datasets Overview](../../learn/datasets/index.md) - Understanding datasets module
- [Ephemeris Data Sources](../ephemeris/index.md) - CelesTrak and Space-Track API clients
