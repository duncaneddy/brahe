# Time Conversions

Convert between different time representations and systems in Brahe.

## Overview

Brahe provides conversion functions for various time representations:

- Calendar dates (year, month, day, hour, minute, second)
- Julian Date (JD)
- Modified Julian Date (MJD)
- Day of Year (DOY)

## Time Representations

### Calendar Date

Standard Gregorian calendar representation:
- Year, Month, Day, Hour, Minute, Second, Nanosecond

### Julian Date (JD)

Continuous count of days since January 1, 4713 BC (proleptic Julian calendar):
- JD 0.0 = January 1, 4713 BC, 12:00 noon
- Commonly used in astronomy
- Can represent fractional days

### Modified Julian Date (MJD)

Offset version of Julian Date for convenience:
- MJD = JD - 2400000.5
- MJD 0.0 = November 17, 1858, 00:00:00
- More compact representation
- Avoids large numbers

### Day of Year (DOY)

Day count within a calendar year:
- January 1 = day 1
- December 31 = day 365 (or 366 in leap years)

## Conversion Functions

### Calendar to Julian Date

```python
import brahe as bh

jd = bh.datetime_to_jd(2024, 1, 1, 0, 0, 0.0, 0.0)
```

### Julian Date to Calendar

```python
import brahe as bh

year, month, day, hour, minute, second, nanosecond = bh.jd_to_datetime(2460310.5)
```

### Julian Date to Modified Julian Date

```python
mjd = jd - 2400000.5
```

### Modified Julian Date to Julian Date

```python
jd = mjd + 2400000.5
```

## Best Practices

- Use MJD for storage and computation (more compact)
- Use calendar dates for human interaction
- Use JD when interfacing with astronomy software
- Always specify the time system (UTC, TAI, etc.)

## See Also

- [Epoch](epoch.md)
- [Time Systems](time_systems.md)
- [Time Conversions API Reference](../../library_api/time/conversions.md)
