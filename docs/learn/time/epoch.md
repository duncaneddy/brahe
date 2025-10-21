# Epoch

The Epoch class is the fundamental time representation in Brahe.

## Overview

An `Epoch` represents a specific instant in time with:

- A time system (UTC, TAI, TT, GPS, UT1)
- Nanosecond precision
- Efficient arithmetic operations

## Creating Epochs

### From Calendar Date

```python
import brahe as bh

# Create from year, month, day, hour, minute, second, nanosecond, time system
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.UTC)
```

### From Julian Date

```python
import brahe as bh

epoch = bh.Epoch.from_jd(2460310.5, bh.UTC)
```

### From Modified Julian Date

```python
import brahe as bh

epoch = bh.Epoch.from_mjd(60310.0, bh.UTC)
```

## Epoch Arithmetic

```python
import brahe as bh

epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.UTC)

# Add seconds
later_epoch = epoch + 3600.0  # 1 hour later

# Subtract seconds
earlier_epoch = epoch - 1800.0  # 30 minutes earlier

# Difference between epochs (returns seconds)
delta_seconds = later_epoch - epoch
```

## Time System Conversions

```python
import brahe as bh

utc_epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.UTC)

# Convert to different time systems
tai_epoch = utc_epoch.to_time_system(bh.TAI)
gps_epoch = utc_epoch.to_time_system(bh.GPS)
tt_epoch = utc_epoch.to_time_system(bh.TT)
```

## Getting Time Values

```python
import brahe as bh

epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.UTC)

# Get Julian Date
jd = epoch.jd()

# Get Modified Julian Date
mjd = epoch.mjd()

# Get calendar date components
year, month, day, hour, minute, second, nanosecond = epoch.to_datetime()
```

## Best Practices

- Store all epochs in a consistent time system (UTC or TAI recommended)
- Use TAI for propagation (no leap seconds)
- Convert to UTC only for display
- Always specify time system explicitly

## See Also

- [Epoch API Reference](../../library_api/time/epoch.md)
- [Time Systems](time_systems.md)
- [Time Conversions](time_conversions.md)
