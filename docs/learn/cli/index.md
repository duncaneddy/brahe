# Brahe CLI

The Brahe command-line interface provides tools for orbital mechanics, time system conversions, satellite operations, and astrodynamics calculations.

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

# Calculate orbital period
brahe orbits orbital-period "R_EARTH+500e3"

# Transform coordinates between representations
brahe transform coordinates keplerian cartesian "" 6878137 0.001 97.8 0 0 0 --as-degrees

# Compute satellite access windows
brahe access compute 25544 --lat 40.7128 --lon -74.0060
```

## Command Groups

### [transform](transform.md)
Convert between coordinate systems and reference frames:
- `frame` - Transform states between ECI and ECEF frames
- `coordinates` - Convert between Keplerian, Cartesian, Geodetic, and Geocentric representations
- `attitude` - Convert between attitude representations (planned)

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

### [eop](eop.md)
Earth Orientation Parameter operations:
- `download` - Download EOP data from IERS
- `get-utc-ut1` - Get UTC-UT1 offset
- `get-polar-motion` - Get polar motion parameters
- `get-cip-offset` - Get CIP offset
- `get-lod` - Get length of day

### [access](access.md)
Satellite access window calculations:
- `compute` - Calculate visibility windows for ground stations

### [datasets](datasets.md)
Download and query satellite data:
- `celestrak` - CelesTrak TLE data operations
- `groundstations` - Ground station database operations

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
brahe orbits orbital-period "R_EARTH+500e3"

# Multiple constants in calculations
brahe orbits sma-from-period "2*3.14159*sqrt((R_EARTH+35786e3)^3/GM_EARTH)" --units seconds
```

Available constants include:
- `R_EARTH`, `R_SUN`, `R_MOON` - Body radii (meters)
- `GM_EARTH`, `GM_SUN`, `GM_MOON` - Gravitational parameters (m³/s²)
- `DEG2RAD`, `RAD2DEG` - Angular conversions
- `MJD_ZERO`, `MJD2000`, `GPS_ZERO` - Time epoch constants

See the [Constants documentation](../constants.md) for the complete list.

### Angle Formats

Commands that handle angles typically support both degrees and radians:

```bash
# Degrees (default for most commands)
brahe transform coordinates keplerian cartesian "" 6878137 0.001 97.8 0 0 45 --as-degrees

# Radians
brahe transform coordinates keplerian cartesian "" 6878137 0.001 1.706 0 0 0.785 --no-as-degrees
```

### Output Formatting

Control numeric output precision with `--format`:

```bash
# Default floating-point
brahe orbits orbital-period "R_EARTH+500e3"
# Output: 5673.281746

# High precision
brahe orbits orbital-period "R_EARTH+500e3" --format .10f
# Output: 5673.2817464420

# Scientific notation
brahe orbits orbital-period "R_EARTH+500e3" --format .3e
# Output: 5.673e+03
```

## Earth Orientation Parameters

Commands involving reference frame transformations (ECI ↔ ECEF) automatically download and cache Earth Orientation Parameter (EOP) data from IERS on first use:

```bash
# EOP data automatically downloaded for frame transformations
brahe transform frame ECI ECEF "2024-01-01T00:00:00Z" 6878137 0 0 0 7500 0

# Manually download/update EOP data
brahe eop download ~/.cache/brahe/iau2000_standard.txt --product standard
```

Cache location: `~/.cache/brahe/`

## Time Formats

Epochs can be specified in multiple formats:

**ISO-8601 strings** (most common):
```bash
brahe time convert "2024-01-01T00:00:00Z" string mjd
brahe transform frame ECI ECEF "2024-01-01T12:30:45.123Z" 6878137 0 0 0 7500 0
```

**Modified Julian Date (MJD)**:
```bash
brahe time convert 60310.0 mjd string --output-time-system UTC
```

**Julian Date (JD)**:
```bash
brahe time convert 2460310.5 jd string
```

**GPS time**:
```bash
brahe time convert "1356998418.0" gps_nanoseconds string
```

## Common Workflows

### Satellite State Conversion

```bash
# 1. Start with Keplerian elements (SSO, 500km altitude)
KEP="6878137 0.001 97.8 0 0 0"

# 2. Convert to ECI Cartesian
brahe transform coordinates keplerian cartesian "" $KEP --as-degrees

# 3. Convert to ECEF at specific epoch
brahe transform frame ECI ECEF "2024-01-01T00:00:00Z" 6871258.863 0.0 0.0 0.0 -1034.183 7549.721

# 4. Convert to geodetic coordinates
brahe transform coordinates cartesian geodetic "2024-01-01T00:00:00Z" \
  -1176064.179 -6776827.197 15961.825 6895.377 -1196.637 0.241 \
  --from-frame ECEF --as-degrees
```

### Ground Station Passes

```bash
# Find ISS passes over New York City
brahe access compute 25544 --lat 40.7128 --lon -74.0060 --duration 7

# Use ground station database
brahe access compute 25544 --gs-provider ksat --gs-name "Svalbard"

# Export to JSON
brahe access compute 25544 --lat 40.7128 --lon -74.0060 --output-file passes.json
```

### Orbital Design

```bash
# Design a sun-synchronous orbit at 600km altitude
SMA="R_EARTH+600e3"
ECC="0.001"

# Calculate required inclination
INC=$(brahe orbits sun-sync-inclination "$SMA" "$ECC" --as-degrees)

# Calculate period
brahe orbits orbital-period "$SMA" --units minutes

# Calculate mean motion
brahe orbits mean-motion "$SMA"
```

## See Also

- [Python API Documentation](../../library_api/index.md)
- [Coordinate Systems](../coordinates/index.md)
- [Time Systems](../time/index.md)
- [Orbital Mechanics](../orbits/index.md)
