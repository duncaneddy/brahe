# Earth Orientation Parameters (EOP)

Earth Orientation Parameters are essential data for high-precision transformations between Earth-Centered Inertial (ECI) and Earth-Centered Earth-Fixed (ECEF) reference frames. These parameters account for irregularities in Earth's rotation that cannot be modeled purely mathematically.

Since many transformations depend on accurate EOP data, Brahe abstracts thsi complexity through a flexible global provider system. This allows
users to initialize EOP data once and have it automatically applied in all relevant calculations without needing to pass EOP data explicitly with
each function call.

## Overview

EOP data provides corrections for:

- **UT1-UTC**: Difference between Universal Time (based on Earth rotation) and Coordinated Universal Time
- **Polar Motion (x, y)**: Variation in Earth's rotation axis position relative to its crust
- **Celestial Pole Offsets (dX, dY)**: Corrections to precession-nutation models
- **Length of Day (LOD)**: Variations in Earth's rotation rate

## EOP Providers

Brahe supports multiple ways to access EOP data, each suited for different use cases.

### StaticEOPProvider

Uses fixed EOP data, best for applications that don't require the most current data or when internet access is unavailable.

```python
import brahe as bh

# Use built-in static data
provider = bh.StaticEOPProvider.from_zero() # Sets all EOP values to zero
bh.set_global_eop_provider_from_static_provider(provider)
```

**When to use**:

- Testing and development
- Offline environments
- Applications where high precision is not critical

### FileEOPProvider

Loads EOP data from IERS (International Earth Rotation and Reference Systems Service) data files. Provides the most flexibility and control over EOP data sources.

```python
import brahe as bh

# Download latest EOP file
bh.download_standard_eop_file("./eop_data/finals.all.iau2000.txt")

# Load from file
provider = bh.FileEOPProvider.from_file(
    "./eop_data/finals.all.iau2000.txt",
    interpolate=True,
    extrapolate="Hold"
)
bh.set_global_eop_provider_from_file_provider(provider)
```

**When to use**:

- You manage EOP file updates externally
- You need specific historical EOP data versions
- You want full control over data sources
- Minimal runtime overhead is critical

### CachingEOPProvider

Automatically manages EOP file freshness by monitoring file age and downloading updates when data becomes stale. Combines the precision of file-based data with automatic updates.

```python
import brahe as bh

# Create provider that refreshes files older than 7 days
provider = bh.CachingEOPProvider(
    filepath="./eop_data/finals.all.iau2000.txt",
    eop_type="StandardBulletinA",
    max_age_seconds=7 * 86400,  # 7 days
    auto_refresh=False,          # Manual refresh only
    interpolate=True,
    extrapolate="Hold"
)
bh.set_global_eop_provider_from_caching_provider(provider)
```

**When to use**:

- Long-running services that need current EOP data for accuracy
- Applications where automatic updates are preferred over manual management
- Production systems requiring data freshness guarantees

## Automatic Cache Management

The `CachingEOPProvider` offers two refresh strategies:

### Manual Refresh (Recommended)

Check and update EOP data at controlled intervals:

```python
import brahe as bh
import time

# Create provider with manual refresh
provider = bh.CachingEOPProvider(
    filepath="./eop_data/finals.all.iau2000.txt",
    eop_type="StandardBulletinA",
    max_age_seconds=7 * 86400,  # 7 days
    auto_refresh=False,
    interpolate=True,
    extrapolate="Hold"
)

# Use in application
while True:
    # Refresh at start of processing cycle
    provider.refresh()

    # Process data with current EOP
    perform_calculations()

    # Wait before next cycle
    time.sleep(3600)  # 1 hour
```

**Advantages**:

- No performance overhead during data access
- Predictable refresh timing
- Better for batch processing and scheduled tasks

### Auto-Refresh

Automatically check and update on every data access:

```python
import brahe as bh

# Provider checks file age on every access
provider = bh.CachingEOPProvider(
    filepath="./eop_data/finals.all.iau2000.txt",
    eop_type="StandardBulletinA",
    max_age_seconds=24 * 3600,  # 24 hours
    auto_refresh=True,           # Check on every access
    interpolate=True,
    extrapolate="Hold"
)

# EOP data automatically stays current
ut1_utc = provider.get_ut1_utc(mjd)
pm_x, pm_y = provider.get_pm(mjd)
```

**Advantages**:

- Guaranteed data freshness
- Simpler application code
- Suitable for long-running services

**Considerations**:

- Small performance overhead on each access (microseconds)
- May trigger downloads during time-critical operations, potentially causing delays
- Better suited for applications where data access is not in tight loops

## Monitoring File Freshness

Track when EOP data was loaded and how old it is:

```python
import brahe as bh

provider = bh.CachingEOPProvider(
    filepath="./eop_data/finals.all.iau2000.txt",
    eop_type="StandardBulletinA",
    max_age_seconds=7 * 86400,
    auto_refresh=False,
    interpolate=True,
    extrapolate="Hold"
)

# Check when file was loaded
file_epoch = provider.file_epoch()
print(f"EOP file loaded at: {file_epoch}")

# Check file age in seconds
age_seconds = provider.file_age()
age_hours = age_seconds / 3600
age_days = age_seconds / 86400

print(f"File age: {age_hours:.1f} hours ({age_days:.1f} days)")

# Refresh if needed
if age_days > 7:
    print("EOP data is stale, refreshing...")
    provider.refresh()
```

## EOP File Types

### Standard Format (finals2000A.all)

Combined rapid + predicted data updated daily by IERS. Contains:
- Historical data (final values)
- Recent rapid service data
- Predicted values for near future

**Use case**: Most applications requiring current EOP data

```python
provider = bh.CachingEOPProvider(
    filepath="./eop_data/finals.all.iau2000.txt",
    eop_type="StandardBulletinA",
    max_age_seconds=7 * 86400,
    auto_refresh=False,
    interpolate=True,
    extrapolate="Hold"
)
```

### C04 Format

Long-term consistent EOP series from IERS C04 product. Updated less frequently but provides consistent historical record.

**Use case**: Historical analysis, research, long-term consistency

```python
provider = bh.CachingEOPProvider(
    filepath="./eop_data/eopc04.txt",
    eop_type="C04",
    max_age_seconds=30 * 86400,  # 30 days (less frequent updates)
    auto_refresh=False,
    interpolate=True,
    extrapolate="Hold"
)
```

## Configuration Options

### Interpolation

Enable interpolation for smoother data between tabulated points:

```python
# With interpolation (recommended for most applications)
provider = bh.CachingEOPProvider(
    filepath="./eop_data/finals.all.iau2000.txt",
    eop_type="StandardBulletinA",
    max_age_seconds=7 * 86400,
    auto_refresh=False,
    interpolate=True,  # Smooth interpolation
    extrapolate="Hold"
)

# Without interpolation (step function between points)
provider = bh.CachingEOPProvider(
    filepath="./eop_data/finals.all.iau2000.txt",
    eop_type="StandardBulletinA",
    max_age_seconds=7 * 86400,
    auto_refresh=False,
    interpolate=False,  # No interpolation
    extrapolate="Hold"
)
```

### Extrapolation

Control behavior for dates outside the EOP data range:

```python
# Hold last value (recommended for most applications)
provider = bh.CachingEOPProvider(
    filepath="./eop_data/finals.all.iau2000.txt",
    eop_type="StandardBulletinA",
    max_age_seconds=7 * 86400,
    auto_refresh=False,
    interpolate=True,
    extrapolate="Hold"  # Use last known value
)

# Return zero for out-of-range dates
provider = bh.CachingEOPProvider(
    filepath="./eop_data/finals.all.iau2000.txt",
    eop_type="StandardBulletinA",
    max_age_seconds=7 * 86400,
    auto_refresh=False,
    interpolate=True,
    extrapolate="Zero"  # Return 0.0
)

# Raise error for out-of-range dates
provider = bh.CachingEOPProvider(
    filepath="./eop_data/finals.all.iau2000.txt",
    eop_type="StandardBulletinA",
    max_age_seconds=7 * 86400,
    auto_refresh=False,
    interpolate=True,
    extrapolate="Error"  # Raise exception
)
```

## Recommended Refresh Intervals

Choose refresh intervals based on your accuracy requirements:

| Application Type | Recommended Interval | Rationale |
|-----------------|---------------------|-----------|
| Real-time operations | 1-3 days | Balance freshness with download overhead |
| Batch processing | 7 days | Weekly updates sufficient for most accuracy needs |
| Historical analysis | 30+ days | Data rarely changes for historical periods |
| Testing/development | No auto-refresh | Use manual refresh as needed |

```python
import brahe as bh

# Real-time operations
realtime_provider = bh.CachingEOPProvider(
    filepath="./eop_data/realtime.txt",
    eop_type="StandardBulletinA",
    max_age_seconds=2 * 86400,  # 2 days
    auto_refresh=False,
    interpolate=True,
    extrapolate="Hold"
)

# Batch processing
batch_provider = bh.CachingEOPProvider(
    filepath="./eop_data/batch.txt",
    eop_type="StandardBulletinA",
    max_age_seconds=7 * 86400,  # 7 days
    auto_refresh=False,
    interpolate=True,
    extrapolate="Hold"
)

# Historical analysis
historical_provider = bh.CachingEOPProvider(
    filepath="./eop_data/historical.txt",
    eop_type="C04",
    max_age_seconds=30 * 86400,  # 30 days
    auto_refresh=False,
    interpolate=True,
    extrapolate="Hold"
)
```

## Complete Example: Long-Running Service

```python
import brahe as bh
import time
from datetime import datetime

# Initialize caching provider for service
provider = bh.CachingEOPProvider(
    filepath="/var/lib/myapp/eop_data/finals.txt",
    eop_type="StandardBulletinA",
    max_age_seconds=3 * 86400,  # 3 days
    auto_refresh=False,
    interpolate=True,
    extrapolate="Hold"
)

# Set as global provider
bh.set_global_eop_provider_from_caching_provider(provider)

print("Service started with EOP caching")
print(f"Initial EOP age: {provider.file_age() / 86400:.1f} days")

# Service loop
while True:
    # Refresh EOP data at start of cycle
    try:
        provider.refresh()
        print(f"EOP refreshed at {datetime.now()}")
    except Exception as e:
        print(f"EOP refresh failed: {e}")
        # Continue with existing data

    # Perform calculations with current EOP data
    for mjd in range(59000, 59100):
        try:
            # Frame transformations automatically use global EOP provider
            ut1_utc = bh.get_global_ut1_utc(mjd)
            pm_x, pm_y = bh.get_global_pm(mjd)

            # Use for ECEF/ECI transformations
            # ...

        except Exception as e:
            print(f"Error processing MJD {mjd}: {e}")

    # Log current EOP file age
    age_days = provider.file_age() / 86400
    print(f"EOP file age: {age_days:.1f} days")

    # Wait before next cycle (e.g., hourly processing)
    time.sleep(3600)
```

## Thread Safety

`CachingEOPProvider` is thread-safe and can be safely shared across multiple threads:

```python
import brahe as bh
from concurrent.futures import ThreadPoolExecutor

# Create shared provider
provider = bh.CachingEOPProvider(
    filepath="./eop_data/finals.txt",
    eop_type="StandardBulletinA",
    max_age_seconds=7 * 86400,
    auto_refresh=False,
    interpolate=True,
    extrapolate="Hold"
)

def process_epoch(mjd):
    """Process epoch using shared EOP provider"""
    ut1_utc = provider.get_ut1_utc(mjd)
    pm_x, pm_y = provider.get_pm(mjd)
    # Process...
    return result

# Process epochs concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    mjds = range(59000, 60000)
    results = list(executor.map(process_epoch, mjds))
```

## See Also

- [Frame Transformations](frame_transformations.md) - Using EOP data for coordinate frame conversions
- [Time Systems](time.md) - Understanding UT1, UTC, and other time systems
- [API Reference: EOP Module](../library_api/eop/index.md) - Complete API documentation
