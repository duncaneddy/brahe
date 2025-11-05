# Earth Orientation Parameters (EOP)

Earth Orientation Parameters (EOP) are essential corrections that account for irregularities in Earth's rotation. They are required for high-precision transformations between inertial (ECI) and Earth-fixed (ECEF) reference frames.

## Overview

Earth's rotation is not perfectly uniform or predictable due to:

- **Polar motion**: Wobble of Earth's rotation axis relative to its crust
- **UT1-UTC offset**: Variations in Earth's rotation rate
- **Nutation**: Short-period oscillations in Earth's axis orientation

EOP data provides the corrections needed to accurately transform between coordinate systems, which is critical for:

- Satellite orbit determination
- Ground station tracking
- Precision timing
- Navigation and geodesy

## Why EOP Matters

### Without EOP

Coordinate transformations using only theoretical models (precession, nutation) can have errors of:

- **Position errors**: 10-30 meters
- **Velocity errors**: mm/s level
- **Timing errors**: Milliseconds to seconds

### With EOP

Including measured EOP values reduces errors to:

- **Position errors**: < 1 meter
- **Velocity errors**: Sub-mm/s
- **Timing accuracy**: Sub-millisecond

For most satellite applications, EOP is **required** for accurate results.

## EOP Parameters

Brahe uses five primary EOP parameters:

| Parameter | Description | Typical Range | Units |
|-----------|-------------|---------------|-------|
| `x_p` | Polar motion X component | ±0.7 arcsec | arcseconds |
| `y_p` | Polar motion Y component | ±0.7 arcsec | arcseconds |
| `UT1_UTC` | UT1 minus UTC time offset | ±0.9 seconds | seconds |
| `dX` | Celestial pole X offset | ±0.0003 arcsec | arcseconds |
| `dY` | Celestial pole Y offset | ±0.0003 arcsec | arcseconds |

Additionally, derivatives are used for interpolation:
- `LOD`: Length of day variations
- Rates of change for `x_p` and `y_p`

## EOP Data Sources

EOP values are measured by the International Earth Rotation and Reference Systems Service (IERS) using:

- Very Long Baseline Interferometry (VLBI)
- Satellite Laser Ranging (SLR)
- Global Navigation Satellite Systems (GNSS)
- Doppler Orbitography and Radiopositioning Integrated by Satellite (DORIS)

IERS publishes several data products:

- **Finals2000A.all**: Combined observed + predicted values (recommended)
- **EOP C04**: Long-term series with consistent processing
- **Rapid**: Near real-time values updated daily
- **Predictions**: Future values (lower accuracy)

## Managing EOP in Brahe

Brahe provides three EOP provider types:

### FileEOPProvider

Loads EOP data from IERS files for production use:

```python
import brahe as bh

# Download latest EOP data
bh.download_iers_eop_data()

# Load from standard location
eop = bh.FileEOPProvider.from_default_file(bh.EOPType.StandardBulletinA)

# Set as global provider
bh.set_global_eop_provider(eop)
```

**When to use**: Production applications requiring maximum accuracy.

### StaticEOPProvider

Uses built-in historical EOP data or constant values:

```python
# Use built-in data (covers ~1990-2024)
eop = bh.StaticEOPProvider.from_static_data()

# Or use constant values (for testing)
eop = bh.StaticEOPProvider.from_values(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

bh.set_global_eop_provider(eop)
```

**When to use**:
- Testing and development
- Historical analysis within built-in data range
- Applications where 10-30m accuracy is acceptable

### CachingEOPProvider

Wraps another provider with caching for performance:

```python
# Wrap file provider with cache
file_provider = bh.FileEOPProvider.from_default_file(bh.EOPType.StandardBulletinA)
cached_provider = bh.CachingEOPProvider(file_provider, cache_size=1000)

bh.set_global_eop_provider(cached_provider)
```

**When to use**: High-frequency EOP queries at similar epochs (e.g., batch processing).

See: [Managing EOP Data](managing_eop_data.md)

## Global EOP Provider

Brahe uses a global EOP provider that is accessed automatically during frame transformations:

```python
# Set global provider (do this once at program start)
eop = bh.FileEOPProvider.from_default_file(bh.EOPType.StandardBulletinA)
bh.set_global_eop_provider(eop)

# Frame transformations automatically use global EOP
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
state_ecef = bh.state_eci_to_ecef(state_eci, epoch)  # Uses global EOP
```

**Important**: The global provider must be set before any frame transformations, or an error will occur.

## Workflow

### Production Application

```python
import brahe as bh

# 1. Download latest EOP data (run periodically, e.g., weekly)
bh.download_iers_eop_data()

# 2. Load EOP provider at program startup
eop = bh.FileEOPProvider.from_default_file(bh.EOPType.StandardBulletinA)
bh.set_global_eop_provider(eop)

# 3. Perform frame transformations (EOP used automatically)
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
state_ecef = bh.state_eci_to_ecef(state_eci, epoch)
```

### Testing/Development

```python
import brahe as bh

# Use static EOP for reproducible tests
eop = bh.StaticEOPProvider.from_values(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
bh.set_global_eop_provider(eop)

# All transformations use zero EOP values
# Results are deterministic but less accurate
```

## EOP Data Management

### Downloading Updates

EOP data should be updated periodically:

```python
# Download latest data
bh.download_iers_eop_data()

# Reload provider with new data
eop = bh.FileEOPProvider.from_default_file(bh.EOPType.StandardBulletinA)
bh.set_global_eop_provider(eop)
```

**Update frequency**:
- **Real-time applications**: Daily
- **Planning applications**: Weekly
- **Historical analysis**: Not needed (use archived data)

### Data File Location

By default, EOP files are stored in:
- **Unix/Linux/macOS**: `~/.brahe/`
- **Windows**: `C:\Users\<username>\.brahe\`

Files are cached locally and reused until updated.

### EOP Data Coverage

IERS publishes:
- **Historical**: Observed values from 1973 to ~7 days ago
- **Recent**: Rapid service values (updated daily)
- **Future**: Predictions up to 1 year ahead (less accurate)

For dates beyond prediction range, extrapolation is used (accuracy degrades).

## Performance Considerations

### EOP Query Cost

Querying EOP data requires:
1. Date conversion (epoch → MJD)
2. Table lookup or interpolation
3. Parameter extraction

**Typical cost**: 1-10 microseconds per query

### Caching Strategy

For repeated transformations at similar epochs:

```python
# Without caching: ~10 μs per EOP query
eop = bh.FileEOPProvider.from_default_file(bh.EOPType.StandardBulletinA)

# With caching: ~0.1 μs per cached query
cached_eop = bh.CachingEOPProvider(eop, cache_size=1000)
```

Caching provides 100× speedup for repeated queries.

### Batch Operations

When transforming many states at the same epoch:

```python
# Query EOP once, reuse for all transformations
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# EOP queried once internally
for state in states:
    state_ecef = bh.state_eci_to_ecef(state, epoch)  # Reuses cached EOP
```

## Common Patterns

### Startup Initialization

```python
def initialize_eop():
    """Initialize EOP provider at application startup"""
    try:
        # Try to use file-based EOP
        eop = bh.FileEOPProvider.from_default_file(bh.EOPType.StandardBulletinA)
    except Exception:
        # Fall back to static EOP if file not found
        print("Warning: Using static EOP data (reduced accuracy)")
        eop = bh.StaticEOPProvider.from_static_data()

    bh.set_global_eop_provider(eop)

# Call at program start
initialize_eop()
```

### Periodic Updates

```python
import schedule
import brahe as bh

def update_eop_data():
    """Download and reload EOP data"""
    bh.download_iers_eop_data()
    eop = bh.FileEOPProvider.from_default_file(bh.EOPType.StandardBulletinA)
    bh.set_global_eop_provider(eop)
    print("EOP data updated")

# Schedule weekly updates
schedule.every().monday.at("02:00").do(update_eop_data)
```

### Testing with Controlled EOP

```python
import pytest
import brahe as bh

@pytest.fixture(autouse=True)
def setup_eop():
    """Setup zero EOP for deterministic tests"""
    eop = bh.StaticEOPProvider.from_values(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    bh.set_global_eop_provider(eop)
    yield
    # Cleanup if needed

def test_frame_transformation():
    # Test uses zero EOP values (deterministic)
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    state_ecef = bh.state_eci_to_ecef(state_eci, epoch)
    # Assert expected values...
```

---

## See Also

- [What is EOP Data?](what_is_eop_data.md) - Detailed explanation of EOP parameters and why they matter
- [Managing EOP Data](managing_eop_data.md) - Practical guide to EOP providers and data management
- [Frame Transformations](../frame_transformations.md) - How EOP is used in coordinate transformations
- [EOP API Reference](../../library_api/eop/index.md) - Complete EOP provider documentation
- [IERS Website](https://www.iers.org/) - Official source for EOP data
