# FileEOPProvider

Load Earth Orientation Parameters from IERS data files.

::: brahe.FileEOPProvider
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 2

## Overview

`FileEOPProvider` loads EOP data from files in either Standard or C04 format provided by the International Earth Rotation and Reference Systems Service (IERS).

**Module**: `brahe.eop`

**Data Sources**:
- **Standard Format**: finals2000A.all - Combined rapid + predicted data
- **C04 Format**: eopc04_IAU2000.XX - Long-term historical data

## Creating a Provider

### From Default Files

```python
import brahe as bh

# Use default standard format file
provider = bh.FileEOPProvider.from_default_standard()

# Use default C04 format file
provider = bh.FileEOPProvider.from_default_c04()
```

### From Custom Files

```python
import brahe as bh

# Load from custom standard file
provider = bh.FileEOPProvider.from_standard_file(
    "/path/to/finals2000A.all",
    interpolate=True,
    extrapolate="Hold"
)

# Load from custom C04 file
provider = bh.FileEOPProvider.from_c04_file(
    "/path/to/eopc04.XX",
    interpolate=True,
    extrapolate="Hold"
)
```

## Configuration Options

### Interpolation

**`interpolate: bool`** - Enable/disable interpolation between data points

- `True`: Linear interpolation for dates between data points (recommended)
- `False`: Use nearest data point (step function)

### Extrapolation

**`extrapolate: str`** - Behavior when querying dates outside data range

- `"Hold"`: Use first/last values for dates before/after data range
- `"Zero"`: Return zero for all EOP values outside range
- `"Error"`: Raise an error if date is outside range

## Usage with Global EOP

```python
import brahe as bh

# Create provider from file
provider = bh.FileEOPProvider.from_default_standard(
    interpolate=True,
    extrapolate="Hold"
)

# Set as global provider
bh.set_global_eop_provider(provider)

# Now all frame transformations use this EOP data
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
pos_eci = [7000000.0, 0.0, 0.0]
pos_ecef = bh.position_eci_to_ecef(epoch, pos_eci)
```

## Downloading EOP Files

```python
import brahe as bh

# Download latest standard EOP file
filepath = bh.download_standard_eop_file("./data")

# Download latest C04 EOP file
filepath = bh.download_c04_eop_file("./data")

# Use downloaded file
provider = bh.FileEOPProvider.from_standard_file(filepath)
```

## See Also

- [StaticEOPProvider](static_provider.md) - Built-in historical EOP data
- [EOP Functions](functions.md) - Global EOP management
- [Frames](../frames.md) - Coordinate transformations using EOP
