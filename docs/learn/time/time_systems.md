# Time Systems

Brahe supports multiple time systems used in astrodynamics and spacecraft operations.

## Overview

Different time systems are used for different purposes in orbital mechanics. Brahe provides conversions between all major time systems.

## Supported Time Systems

### UTC (Coordinated Universal Time)

- Civil time standard
- Includes leap seconds
- Most common for operational scheduling

### TAI (International Atomic Time)

- Atomic time standard
- No leap seconds
- Monotonically increasing
- UTC = TAI - (leap seconds)

### TT (Terrestrial Time)

- Theoretical time standard
- Used for solar system dynamics
- TT = TAI + 32.184 seconds

### GPS Time

- Time system used by GPS satellites
- No leap seconds since 1980
- GPS = TAI - 19 seconds

### UT1 (Universal Time 1)

- Based on Earth's rotation
- Accounts for polar motion
- Required for ECI/ECEF transformations
- UT1 ≈ UTC (within 0.9 seconds)

## Time Scale Relationships

```
TT = TAI + 32.184 s
TAI = GPS + 19 s
UTC = TAI - (leap seconds)
UT1 ≈ UTC + (UT1-UTC offset from EOP data)
```

## When to Use Each Time System

**UTC**:
- Human-readable timestamps
- Operations scheduling
- Ground station contacts

**TAI/TT**:
- Orbit propagation
- Spacecraft dynamics
- When continuous time is needed

**GPS**:
- GPS satellite operations
- GPS receiver timestamps

**UT1**:
- Earth rotation calculations
- ECI to ECEF transformations
- Ground tracking

## See Also

- [Epoch](epoch.md)
- [Time Conversions](time_conversions.md)
- [Time Constants](../../library_api/constants/time.md)
