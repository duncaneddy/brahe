# Units

Enumerations for specifying angle formats and time systems.

## Angle Format

The `AngleFormat` enumeration specifies whether angles are in radians or degrees.

::: brahe.AngleFormat
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - RADIANS
        - DEGREES
      show_bases: false
      heading_level: 3

### Usage Example

```python
import brahe as bh
import numpy as np

# Create rotation with angle in degrees
q = bh.Quaternion.from_euler_axis(
    axis=np.array([0.0, 0.0, 1.0]),
    angle=90.0,
    angle_format=bh.AngleFormat.DEGREES
)

# Create rotation with angle in radians
q2 = bh.Quaternion.from_euler_axis(
    axis=np.array([0.0, 0.0, 1.0]),
    angle=np.pi/2,
    angle_format=bh.AngleFormat.RADIANS
)
```

## Time System

The `TimeSystem` enumeration specifies the time reference system for epochs.

::: brahe.TimeSystem
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - UTC
        - TAI
        - TT
        - GPS
        - UT1
      show_bases: false
      heading_level: 3

### Time System Descriptions

- **UTC** (Coordinated Universal Time): Civil time standard used worldwide. Includes leap seconds to keep within 0.9 seconds of UT1.
- **TAI** (International Atomic Time): Continuous time scale based on atomic clocks. Currently 37 seconds ahead of UTC (as of 2024).
- **TT** (Terrestrial Time): Theoretical time scale for solar system calculations. TT = TAI + 32.184 seconds.
- **GPS** (Global Positioning System): Continuous time starting from GPS epoch (January 6, 1980). Does not include leap seconds.
- **UT1** (Universal Time 1): Based on Earth's rotation. Computed from UTC using Earth Orientation Parameters (EOP).

### Usage Example

```python
import brahe as bh

# Create epoch in different time systems
utc_epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
tai_epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.TAI)
gps_epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.GPS)

# Time system is preserved in the epoch
print(utc_epoch.time_system)  # Output: UTC
print(tai_epoch.time_system)  # Output: TAI
```
