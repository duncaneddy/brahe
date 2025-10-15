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

Brahe supports two types of EOP providers:

### [FileEOPProvider](file_provider.md)
Load EOP data from files (Standard or C04 format) for production applications with current data.

### [StaticEOPProvider](static_provider.md)
Use built-in historical EOP data for testing or when internet access is unavailable.

## Global EOP Management

EOP data is managed globally to avoid passing providers through every function call.

### [Functions](functions.md)
- Setting global EOP providers
- Querying global EOP data
- Downloading latest EOP files

## Quick Start

```python
import brahe as bh

# Option 1: Use file-based EOP (recommended for production)
bh.set_global_eop_provider_from_file_provider(
    bh.FileEOPProvider.from_default_standard()
)

# Option 2: Use static EOP (for testing/offline use)
bh.set_global_eop_provider_from_static_provider(
    bh.StaticEOPProvider.from_zero()
)

# Now frame transformations will use the global EOP data
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
pos_eci = [7000000.0, 0.0, 0.0]  # meters
pos_ecef = bh.position_eci_to_ecef(epoch, pos_eci)
```

## See Also

- [Frames](../frames.md) - Coordinate frame transformations that use EOP
- [Epoch](../time/epoch.md) - Time representation
