# Managing EOP Data

Learn how to manage Earth Orientation Parameter (EOP) data in Brahe.

## Overview

Earth Orientation Parameters are essential for accurate transformations between Earth-Centered Inertial (ECI) and Earth-Centered Earth-Fixed (ECEF) coordinate frames. Brahe provides multiple options for managing EOP data.

## EOP Provider Types

### StaticEOPProvider

Use constant EOP values (no time variation):

```python
import brahe as bh

# Use zero values (no Earth orientation corrections)
eop = bh.StaticEOPProvider.from_zero()
bh.set_global_eop_provider(eop)

# Use custom static values
eop = bh.StaticEOPProvider.from_values(
    pm_x=0.1,      # radians
    pm_y=0.0,      # radians
    ut1_utc=0.0,   # seconds
    dx=0.0,        # radians
    dy=0.0,        # radians
    lod=0.0        # seconds
)
bh.set_global_eop_provider(eop)
```

**When to use**: Testing, low-accuracy applications, or when EOP data is unavailable.

### FileEOPProvider

Load EOP data from files:

```python
import brahe as bh

# Use default file location
eop = bh.FileEOPProvider.from_default_standard(
    interpolate=True,
    extrapolate="Hold"
)
bh.set_global_eop_provider(eop)

# Or specify a custom file
eop = bh.FileEOPProvider.from_standard_file(
    "./data/finals.all.iau2000.txt",
    interpolate=True,
    extrapolate="Hold"
)
bh.set_global_eop_provider(eop)
```

**When to use**: Production applications requiring accurate transformations.

### CachingEOPProvider

Automatically download and cache EOP data:

```python
import brahe as bh

eop = bh.CachingEOPProvider(
    filepath="./data/eop.txt",
    eop_type="StandardBulletinA",
    max_age_seconds=7 * 86400,  # Refresh if older than 7 days
    auto_refresh=False,
    interpolate=True,
    extrapolate="Hold"
)
bh.set_global_eop_provider(eop)
```

**When to use**: Long-running applications that need up-to-date EOP data.

## Downloading EOP Files

```python
import brahe as bh

# Download standard IERS finals file
bh.download_standard_eop_file("./data/finals.all.iau2000.txt")

# Download C04 file
bh.download_c04_eop_file("./data/finals2000A.all.csv")
```

## Interpolation and Extrapolation

**Interpolation** (enabled/disabled):
- `True`: Linearly interpolate between tabulated values
- `False`: Use nearest tabulated value

**Extrapolation** modes:
- `"Hold"`: Use last known value when beyond data range
- `"Zero"`: Return zero when beyond data range
- `"Error"`: Raise exception when beyond data range

## Best Practices

1. **Use FileEOPProvider for production**: Most accurate results
2. **Enable interpolation**: Smoother EOP values
3. **Use "Hold" extrapolation**: Better than zero for near-term extrapolation
4. **Update EOP data regularly**: IERS data updates weekly
5. **Use StaticEOPProvider only for testing**: Or when accuracy isn't critical

## See Also

- [What is EOP Data?](what_is_eop_data.md)
- [FileEOPProvider API Reference](../../library_api/eop/file_provider.md)
- [StaticEOPProvider API Reference](../../library_api/eop/static_provider.md)
- [CachingEOPProvider API Reference](../../library_api/eop/caching_provider.md)
