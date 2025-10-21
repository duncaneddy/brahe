# Time Systems and Representations

Time is fundamental to astrodynamics, yet surprisingly complex. Different applications require different time systems, each with specific properties and use cases. Brahe provides comprehensive time representation and conversion capabilities.

## Overview

Brahe supports multiple time systems and provides:

- **`Epoch`**: Primary time representation with nanosecond precision
- **Time system conversions**: Transform between UTC, TAI, GPS, TT, and UT1
- **Calendar conversions**: Julian Day (JD) and Modified Julian Day (MJD)
- **DateTime integration**: Natural Python datetime interoperability

## Time Systems

### UTC (Coordinated Universal Time)

Civil time standard synchronized with Earth's rotation via leap seconds.

**Properties**:
- Based on atomic clocks
- Adjusted with leap seconds to stay within 0.9s of UT1
- Not continuous (jumps during leap seconds)

**Use cases**:
- User interfaces and displays
- Ground station scheduling
- Mission planning

**Limitations**: Discontinuous due to leap seconds, not suitable for numerical integration.

### TAI (International Atomic Time)

Continuous atomic time scale, the foundation of all modern time systems.

**Properties**:
- Based on ensemble of atomic clocks worldwide
- Perfectly continuous (no leap seconds)
- TAI = UTC + (leap seconds)

**Use cases**:
- Time system conversions
- Precise time intervals
- Internal timing references

**Offset**: TAI - UTC = 37 seconds (as of 2024)

### GPS Time

Continuous time used by Global Positioning System.

**Properties**:
- Based on GPS satellite clocks
- Continuous like TAI
- GPS = TAI - 19 seconds (constant offset)

**Use cases**:
- GPS navigation
- Satellite timing
- Time synchronization

**Offset**: GPS = UTC + 18 seconds (as of 2024, will change with future leap seconds)

### TT (Terrestrial Time)

Theoretical ideal time at Earth's geoid, used for ephemerides and celestial mechanics.

**Properties**:
- Continuous
- TT = TAI + 32.184 seconds (constant offset)
- Proper time on Earth's geoid

**Use cases**:
- Orbital mechanics
- Ephemeris calculations
- Celestial mechanics

### UT1 (Universal Time)

Time based on Earth's actual rotation angle.

**Properties**:
- Follows Earth's rotation
- Non-uniform (Earth's rotation varies)
- UT1 - UTC tracked by EOP data

**Use cases**:
- Earth rotation angle calculations
- Sidereal time
- ECI/ECEF transformations

**Note**: Requires Earth Orientation Parameters (EOP) for accurate UTC ↔ UT1 conversion.

See: [Time Systems](time_systems.md)

## The Epoch Type

`Epoch` is Brahe's primary time representation:

```python
import brahe as bh

# Create from datetime
epoch = bh.Epoch.from_datetime(
    2024, 1, 1,  # Year, month, day
    12, 30, 45.123456789,  # Hour, minute, seconds (with nanosecond precision)
    0.0,  # Fractional day offset
    bh.TimeSystem.UTC
)

# Create from Julian Day
epoch = bh.Epoch.from_jd(2451545.0, bh.TimeSystem.TT)

# Create from Modified Julian Day
epoch = bh.Epoch.from_mjd(51544.5, bh.TimeSystem.UTC)

# Query in different representations
jd = epoch.jd()  # Julian Day
mjd = epoch.mjd()  # Modified Julian Day
year, month, day, hour, minute, second, frac_day = epoch.to_datetime()

# Convert between time systems
epoch_tai = epoch.to_time_system(bh.TimeSystem.TAI)
epoch_gps = epoch.to_time_system(bh.TimeSystem.GPS)
```

**Key features**:
- Nanosecond precision (sufficient for all satellite applications)
- Immutable (thread-safe)
- Efficient arithmetic (add/subtract seconds)
- Automatic time system conversion

See: [Epoch](epoch.md)

## Time System Conversions

Brahe provides both high-level (`Epoch`) and low-level conversion functions:

### Using Epoch (Recommended)

```python
# Create in UTC
epoch_utc = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Convert to other systems
epoch_tai = epoch_utc.to_time_system(bh.TimeSystem.TAI)
epoch_gps = epoch_utc.to_time_system(bh.TimeSystem.GPS)
epoch_tt = epoch_utc.to_time_system(bh.TimeSystem.TT)
```

### Using Conversion Functions

```python
# JD/MJD conversions
jd = bh.datetime_to_jd(2024, 1, 1, 12, 0, 0.0)
year, month, day, hour, minute, second = bh.jd_to_datetime(jd)

mjd = bh.jd_to_mjd(jd)
jd = bh.mjd_to_jd(mjd)

# Time system offsets
tai_utc = bh.tai_utc(epoch)  # Get TAI-UTC offset (leap seconds)
gps_utc = bh.gps_utc(epoch)  # Get GPS-UTC offset
```

See: [Time Conversions](time_conversions.md)

## Common Patterns

### Orbital Propagation Timesteps

```python
# Use GPS time for continuous propagation
epoch_start = bh.Epoch.from_datetime(
    2024, 1, 1, 0, 0, 0.0, 0.0,
    bh.TimeSystem.GPS  # Continuous time system
)

# Propagate for 24 hours
dt = 60.0  # seconds
states = []
for i in range(1440):  # 24 hours * 60 min/hr
    epoch = epoch_start + (i * dt)
    state = propagator.state(epoch)
    states.append(state)
```

### Mission Timeline with UTC

```python
# User-facing times in UTC
events = [
    bh.Epoch.from_datetime(2024, 6, 15, 10, 30, 0.0, 0.0, bh.TimeSystem.UTC),  # Launch
    bh.Epoch.from_datetime(2024, 6, 15, 10, 45, 0.0, 0.0, bh.TimeSystem.UTC),  # Stage separation
    bh.Epoch.from_datetime(2024, 6, 15, 11, 15, 0.0, 0.0, bh.TimeSystem.UTC),  # Orbit insertion
]

# Convert to GPS for calculations
events_gps = [e.to_time_system(bh.TimeSystem.GPS) for e in events]

# Compute time intervals (continuous time)
intervals = [
    (events_gps[i+1] - events_gps[i])
    for i in range(len(events_gps) - 1)
]
```

### Absolute vs Relative Time

```python
# Absolute time (Epoch)
t0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Relative time (float seconds)
dt = 3600.0  # 1 hour offset

# Compute future epoch
t1 = t0 + dt

# Time span between epochs
duration = t1 - t0  # Returns seconds (float)
```

## Julian Day Systems

### Julian Day (JD)

Days since January 1, 4713 BCE at 12:00 TT:

- **J2000 epoch**: JD 2451545.0 (January 1, 2000, 12:00 TT)
- **Range**: All historical dates
- **Precision**: ~1 microsecond at 2000 CE (double precision)

### Modified Julian Day (MJD)

Offset version of JD for convenience:

- **MJD = JD - 2400000.5**
- **MJD 0**: November 17, 1858, 00:00 UTC
- **Advantages**: Smaller numbers, starts at midnight (not noon)

**When to use**:
- **JD**: Astronomical calculations, ephemerides
- **MJD**: Satellite operations, engineering applications

## Time Precision

### Epoch Precision

`Epoch` stores time as:
- Days since epoch (integer)
- Nanoseconds within day (integer)

**Effective precision**: ~1 nanosecond

**Practical limits**:
- Satellite position: mm-level precision
- Timing events: Sub-microsecond accuracy
- Propagation: Limited by physics models, not time representation

### Floating-Point Considerations

When using JD/MJD as `float64`:

- **Precision at year 2000**: ~1 microsecond
- **Precision at year 2100**: ~10 microseconds
- **Loss of precision**: ~1 ns per century

For sub-microsecond timing, use `Epoch` instead of raw JD.

## Leap Seconds

Leap seconds are inserted to keep UTC within 0.9 seconds of UT1 (Earth's rotation).

**Historical leap seconds**: 27 since 1972 (as of 2024)

**Handling in Brahe**:
- UTC times during leap second (23:59:60) are represented as 23:59:59.999...
- TAI-UTC offset updated after each leap second
- Leap second table built into library (updated with new releases)

**Future leap seconds**: Unpredictable (depend on Earth rotation variations)

## Performance Considerations

### Epoch vs Raw Numbers

```python
# Efficient: Use Epoch for time bookkeeping
epochs = [epoch_start + (i * dt) for i in range(1000)]

# Less efficient: Manual JD arithmetic
jds = [jd_start + (i * dt / 86400.0) for i in range(1000)]
```

`Epoch` is optimized for time arithmetic and conversions.

### Caching Conversions

For repeated conversions at similar times:

```python
# Convert once, reuse
epoch_gps = epoch_utc.to_time_system(bh.TimeSystem.GPS)

# Use converted epoch multiple times
for state in states:
    propagate(state, epoch_gps)
```

## Common Pitfalls

### Discontinuous Time Systems

Don't use UTC for numerical integration:

```python
# WRONG: UTC has leap seconds (discontinuous)
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
for i in range(1000):
    epoch = epoch + 60.0  # Incorrect during leap seconds

# CORRECT: Use GPS or TAI (continuous)
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.GPS)
for i in range(1000):
    epoch = epoch + 60.0  # Always correct
```

### Time System Confusion

Always specify time system explicitly:

```python
# Ambiguous
epoch = bh.Epoch.from_jd(2451545.0, bh.TimeSystem.UTC)  # Uncommon

# Clear
epoch = bh.Epoch.from_jd(2451545.0, bh.TimeSystem.TT)  # Standard
```

### Precision Loss

Avoid unnecessary conversions:

```python
# Multiple conversions lose precision
epoch_utc = epoch_tt.to_time_system(bh.TimeSystem.UTC)
epoch_tai = epoch_utc.to_time_system(bh.TimeSystem.TAI)
epoch_gps = epoch_tai.to_time_system(bh.TimeSystem.GPS)

# Better: Direct conversion
epoch_gps = epoch_tt.to_time_system(bh.TimeSystem.GPS)
```

## See Also

- [Time Systems](time_systems.md) - Detailed explanation of each time system
- [Time Conversions](time_conversions.md) - Conversion functions and algorithms
- [Epoch](epoch.md) - Complete guide to the Epoch type
- [Time API Reference](../../library_api/time/index.md) - Complete time function documentation
- [Time Constants](../../library_api/constants/time.md) - Important time-related constants
