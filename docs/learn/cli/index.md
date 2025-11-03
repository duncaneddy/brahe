# Brahe CLI

The Brahe command-line interface provides tools for quick-access time, coordinate, and orbital mechanics calculations directly from the terminal. It also provides functions to download Earth Orientation Parameters (EOP) and satellite datasets.

## Installation

The CLI is included with the Brahe Python package:

```bash
pip install brahe
# or
uv pip install brahe
```

## Quick Start

```bash
# Get help
brahe --help

# Convert between time formats
brahe time convert "2024-01-01T00:00:00Z" string mjd
# 60310.00042824074

# Calculate orbital period
brahe orbits orbital-period "R_EARTH+500e3" --units minutes
# 94.616286

# Transform coordinates between representations
brahe transform coordinates keplerian cartesian "" 6878137 0.001 97.8 0 0 0 --as-degrees
# [6871258.863000, 0.000000, 0.000000, 0.000000, -1034.183142, 7549.721055

# Compute satellite access windows
brahe access compute 25544 --lon -122.4194 --lat 37.7749
# Access Windows for ISS (ZARYA) (NORAD ID: 25544)
# Location: 37.7749° lat, -122.4194° lon, 0 m alt
# Minimum elevation: 10.0°
# Found 17 access window(s)
# ...
```

## Command Groups

### [eop](eop.md)

Earth Orientation Parameter operations:
- `download` - Download EOP data from IERS
- `get-utc-ut1` - Get UTC-UT1 offset
- `get-polar-motion` - Get polar motion parameters
- `get-cip-offset` - Get CIP offset
- `get-lod` - Get length of day

### [datasets](datasets.md)

Download and query satellite data:
- `celestrak` - CelesTrak TLE data operations
- `groundstations` - Ground station database operations

### [time](time.md)

Time system operations and conversions:

- `convert` - Convert between time formats (MJD, JD, GPS, ISO-8601)
- `add` - Add time offsets to epochs
- `time-system-offset` - Calculate offsets between time systems
- `range` - Generate time ranges

### [orbits](orbits.md)

Orbital mechanics calculations:

- `orbital-period` - Calculate orbital period from semi-major axis
- `sma-from-period` - Calculate semi-major axis from period
- `mean-motion` - Calculate mean motion
- `anomaly-conversion` - Convert between anomaly types
- `sun-sync-inclination` - Calculate sun-synchronous inclination
- `perigee-velocity` / `apogee-velocity` - Calculate velocities at apsides

### [transform](transform.md)

Convert between coordinate systems and reference frames:
<!-- - `frame` - Transform states between ECI and ECEF frames -->
- `coordinates` - Convert between Keplerian, Cartesian, Geodetic, and Geocentric representations
<!-- - `attitude` - Convert between attitude representations (planned) -->

### [access](access.md)

Satellite access window calculations:
- `compute` - Calculate visibility windows for ground stations


## Global Options

```
--verbose, -v    Enable verbose output (INFO level)
--debug, -d      Enable debug output (DEBUG level)
--help           Show help message
```

## Features

### Constant Expressions

Many numeric arguments support mathematical expressions using Brahe constants:

```bash
# Use R_EARTH constant for semi-major axis (500km altitude LEO)
brahe orbits orbital-period "R_EARTH+500e3" --units minutes
# 94.616286
```

Available constants include:
- `R_EARTH`, `R_SUN`, `R_MOON` - Body radii (meters)
- `GM_EARTH`, `GM_SUN`, `GM_MOON` - Gravitational parameters (m³/s²)
- `DEG2RAD`, `RAD2DEG` - Angular conversions
- `MJD_ZERO`, `MJD2000`, `GPS_ZERO` - Time epoch constants

See the [Constants documentation](../constants.md) for the complete list.


## See Also

- [Python API Documentation](../../library_api/index.md)
- [Coordinate Systems](../coordinates/index.md)
- [Time Systems](../time/index.md)
- [Orbital Mechanics](../orbits/index.md)
