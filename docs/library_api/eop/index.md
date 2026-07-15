# Earth Orientation Parameters (EOP)

**Module**: `brahe.eop`

Earth Orientation Parameters provide corrections for the irregular rotation and orientation of the Earth, essential for accurate coordinate frame transformations between ECI and ECEF systems.

## Overview

EOP data includes:
- **UT1-UTC**: Difference between UT1 (Earth rotation time) and UTC
- **Polar Motion** (x, y): Movement of Earth's rotation axis relative to the crust
- **dX, dY**: Celestial pole offsets
- **LOD**: Length of day variations

## EOP Providers

Brahe supports three types of EOP providers:

### [CachingEOPProvider](caching_provider.md)
Automatically manage EOP file freshness with cache management and automatic updates.

### [FileEOPProvider](file_provider.md)
Load EOP data from files (Standard or C04 format) for production applications with current data.

### [StaticEOPProvider](static_provider.md)
Use user-defined fixed data, ideal for testing, offline use, or applications not requiring the most precise transformations.

## Global EOP Management

EOP data is managed globally to avoid passing providers through every function call.

### [Functions](functions.md)
- Setting global EOP providers
- Querying global EOP data
- Downloading latest EOP files

## See Also

- [Frames](../frames/index.md) - Coordinate frame transformations that use EOP
- [Epoch](../time/epoch.md) - Time representation
