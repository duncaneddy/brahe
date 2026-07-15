# TimeSystem

**Module**: `brahe.time`

The `TimeSystem` enumeration specifies the time scale in which an `Epoch` is
expressed.

::: brahe.TimeSystem
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - GPS
        - TAI
        - TT
        - UTC
        - UT1
        - TDB
        - TCG
        - TCB
        - BDT
        - GST
      show_bases: false
      heading_level: 3

### Time System Descriptions

- **GPS** (Global Positioning System time): Atomic time scale aligned with UTC at inception (January 6, 1980), but does not include leap seconds. Used by GPS navigation systems.
- **TAI** (International Atomic Time): Continuous atomic time scale representing the passage of time on Earth's geoid. Does not include leap seconds.
- **TT** (Terrestrial Time): Theoretical atomic time standard used primarily for astronomy. Offset from TAI by exactly 32.184 seconds (TT = TAI + 32.184s).
- **UTC** (Coordinated Universal Time): Atomic time scale steered to remain within +/- 0.9 seconds of UT1 by incorporating leap seconds. The standard for civil timekeeping worldwide.
- **UT1** (Universal Time 1): Solar time representing Earth's rotation relative to the ICRF inertial frame. Mean solar time at 0 degrees longitude; varies irregularly with Earth's rotation. Computed from UTC using Earth Orientation Parameters.
- **TDB** (Barycentric Dynamical Time): Time scale for solar system barycentric ephemerides. Differs from TT by small periodic terms (< 1.7 ms) from the relativistic effects of Earth's orbital motion. TDB seconds past J2000 is SPICE ephemeris time (ET).
- **TCG** (Geocentric Coordinate Time): Coordinate time for geocentric reference systems. Differs from TT by a secular drift (~0.7 s/year) from Earth's gravitational time dilation.
- **TCB** (Barycentric Coordinate Time): Coordinate time for the solar system barycenter. Differs from TDB by a secular drift from relativistic effects.
- **BDT** (BeiDou Navigation Satellite System Time): Atomic time scale aligned with UTC at inception (January 1, 2006). Fixed offset from TAI of 33 seconds (BDT = TAI - 33s).
- **GST** (Galileo System Time): Atomic time scale for the Galileo navigation system. Steered to GPS time, sharing the same fixed offset from TAI of 19 seconds (GST = TAI - 19s).

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

## See Also

- [Epoch](epoch.md) - The time representation that carries a `TimeSystem`
- [Time Conversions](conversions.md) - Functions for converting between time systems
- [Time Systems and Representations](../../learn/time/index.md) - Conceptual guide to the time scales
