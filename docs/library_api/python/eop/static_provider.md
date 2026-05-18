# StaticEOPProvider

Built-in Earth Orientation Parameters for testing and offline use.

::: brahe.StaticEOPProvider
    options:
      show_root_heading: true
      show_root_full_path: false

## Overview

`StaticEOPProvider` provides built-in historical EOP data that doesn't require external files. Useful for testing, examples, or when internet access is unavailable.

**Module**: `brahe.eop`

**Use Cases**:
- Unit testing
- Examples and tutorials
- Offline applications
- Quick prototyping

**Limitations**:
- Fixed historical data (not updated)
- Less accurate than file-based providers
- Not suitable for production applications requiring current data

## Creating a Provider

### Zero Values

```python
import brahe as bh

# All EOP values set to zero
provider = bh.StaticEOPProvider.from_zero()

# Set as global provider
bh.set_global_eop_provider(provider)
```

### Custom Values

```python
import brahe as bh

# Specify custom EOP values
provider = bh.StaticEOPProvider.from_values(
    ut1_utc=0.1,      # UT1-UTC offset (seconds)
    pm_x=0.0001,      # Polar motion X (radians)
    pm_y=0.0001,      # Polar motion Y (radians)
    dx=0.00001,       # Celestial pole offset dX (radians)
    dy=0.00001,       # Celestial pole offset dY (radians)
    lod=0.001         # Length of day offset (seconds)
)
```

### Default Values

```python
import brahe as bh

# Use built-in default values
provider = bh.StaticEOPProvider()
```

## Usage Example

```python
import brahe as bh

# Set up static EOP for testing
bh.set_global_eop_provider(
    bh.StaticEOPProvider.from_zero()
)

# Perform frame transformations
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# ECI to ECEF transformation
pos_eci = [7000000.0, 0.0, 0.0]  # meters in ECI
pos_ecef = bh.position_eci_to_ecef(epoch, pos_eci)

# ECEF to ECI transformation
vel_ecef = [0.0, 7500.0, 0.0]  # m/s in ECEF
vel_eci = bh.position_ecef_to_eci(epoch, vel_ecef)
```

## When to Use

✅ **Use StaticEOPProvider for**:
- Unit tests
- Documentation examples
- Learning and prototyping
- Applications where high accuracy isn't critical

❌ **Don't use StaticEOPProvider for**:
- Production orbit determination
- Precise tracking applications
- Applications requiring current EOP data
- High-accuracy simulations

---

## See Also

- [FileEOPProvider](file_provider.md) - File-based EOP for production use
- [EOP Functions](functions.md) - Global EOP management
- [Frames](../frames/index.md) - Coordinate transformations
