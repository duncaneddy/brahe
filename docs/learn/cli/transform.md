# Transform Commands

Convert between coordinate systems and reference frames.

## Overview

The `transform` command group provides conversions between:
- **Reference frames**: ECI (Earth-Centered Inertial) ↔ ECEF (Earth-Centered Earth-Fixed)
- **Coordinate systems**: Keplerian, Cartesian, Geodetic, Geocentric
- **Attitude representations**: Quaternions, Euler angles, rotation matrices (planned)

## Commands

### `frame`

Transform state vectors between ECI and ECEF reference frames.

**Syntax:**
```bash
brahe transform frame <FROM_FRAME> <TO_FRAME> <EPOCH> <x> <y> <z> <vx> <vy> <vz> [OPTIONS]
```

**Arguments:**
- `FROM_FRAME` - Source reference frame: `ECI` or `ECEF`
- `TO_FRAME` - Target reference frame: `ECI` or `ECEF`
- `EPOCH` - Epoch for the transformation (ISO-8601 format with timezone)
- `x y z vx vy vz` - State vector [m, m, m, m/s, m/s, m/s]

**Options:**
- `--format <fmt>` - Output format string (default: `f`)

**Examples:**

Convert ECI state to ECEF at a specific epoch:
```bash
brahe transform frame ECI ECEF "2024-01-01T00:00:00Z" 6878137 0 0 0 7500 0
```
Output:
```
[-1176064.179304, -6776827.196931, 15961.825213, 6895.376569, -1196.636908, 0.240602]
```

Convert ECEF back to ECI:
```bash
brahe transform frame ECEF ECI "2024-01-01T00:00:00Z" -- -1176064.179 -6776827.197 15961.825 6895.377 -1196.637 0.241
```
Output:
```
[6878137.000016, 0.000312, -0.000213, 0.000018, 7500.000440, 0.000398]
```

High-precision output:
```bash
brahe transform frame ECI ECEF "2024-01-01T12:00:00Z" 6878137 0 0 0 7500 0 --format .6f
```
Output:
```
[6593705.875453, -2219542.362118, 15961.820947, 2257.097798, 6768.353894, 0.240601]
```

**Note:** Frame transformations require Earth Orientation Parameters (EOP). The CLI automatically downloads and caches EOP data on first use in `~/.cache/brahe/`.

---

### `coordinates`

Convert between coordinate system representations.

**Syntax:**
```bash
brahe transform coordinates [OPTIONS] <FROM_SYSTEM> <TO_SYSTEM> <EPOCH> <x1> <x2> <x3> <x4> <x5> <x6>
```

**Arguments:**
- `FROM_SYSTEM` - Source coordinate system (see below)
- `TO_SYSTEM` - Target coordinate system (see below)
- `EPOCH` - Epoch (ISO-8601 format). Use `""` if not needed for the conversion
- `x1 x2 x3 x4 x5 x6` - Coordinate values (interpretation depends on system)

**Coordinate Systems:**
- `keplerian` - Keplerian orbital elements [a, e, i, Ω, ω, ν]
- `cartesian` - Cartesian state [x, y, z, vx, vy, vz]
- `geodetic` - Geodetic coordinates [lat, lon, alt, 0, 0, 0]
- `geocentric` - Geocentric spherical [lat, lon, radius, 0, 0, 0]

**Options:**
- `--from-frame [ECI|ECEF]` - Reference frame for cartesian input (default: `ECI`)
- `--to-frame [ECI|ECEF]` - Reference frame for cartesian output (default: `ECI`)
- `--as-degrees / --no-as-degrees` - Interpret/output angles in degrees (default: `--as-degrees`)
- `--format <fmt>` - Output format string (default: `f`)

**Examples:**

#### Keplerian to Cartesian (ECI)

Convert orbital elements to ECI state (no epoch required):
```bash
brahe transform coordinates keplerian cartesian "" 6878137 0.001 97.8 0 0 0 --as-degrees
```
Output:
```
[6871258.863000, 0.000000, 0.000000, 0.000000, -1034.183142, 7549.721055]
```

With different true anomaly (45°):
```bash
brahe transform coordinates keplerian cartesian "" 6878137 0.001 97.8 0 0 45 --as-degrees
```
Output:
```
[4858852.313564, 0.000000, 4861186.164467, -5350.034024, -731.184395, 5337.540890]
```

#### Cartesian to Keplerian

Convert ECI state back to orbital elements:
```bash
brahe transform coordinates --as-degrees cartesian keplerian "" -- 6871258.863 0.0 0.0 0.0 -1034.183 7549.721
```
Output:
```
[6878136.866355, 0.001000, 97.799999, 0.000000, 0.000000, 0.000000]
```

#### Geodetic to Cartesian (ECEF)

Convert geodetic coordinates (New York City) to ECEF:
```bash
brahe transform coordinates --as-degrees --to-frame ECEF geodetic cartesian "" 40.7128 286.0060 10 0 0 0
```
Output:
```
[1334224.912305, -4651969.287142, 4140677.827068]
```

**Note:** Longitude 286° = -74° (use positive east longitude, or handle negative with `--`)

#### ECEF Cartesian to Geodetic

```bash
brahe transform coordinates --as-degrees --from-frame ECEF cartesian geodetic "2024-01-01T00:00:00Z" 1334915.0 4652372.0 4075345.0 0 0 0
```
Output:
```
[73.990114, 40.288227, -41917.492259]
```

#### Keplerian to Geodetic (via ECEF)

Convert satellite orbital elements to ground track position at epoch:
```bash
# First to cartesian ECI, then specify ECEF and geodetic
brahe transform coordinates --as-degrees --to-frame ECEF keplerian geodetic "2024-01-01T00:00:00Z" 6878137 0.001 97.8 0 0 0
```

#### Cartesian ECI to ECEF (frame change)

```bash
brahe transform coordinates --from-frame ECI --to-frame ECEF cartesian cartesian "2024-01-01T00:00:00Z" 6878137 0 0 0 7500 0
```
Output:
```
[-1176064.179304, -6776827.196931, 15961.825213, 6895.376569, -1196.636908, 0.240602]
```

**Alternative:** Use `brahe transform frame` for dedicated ECI↔ECEF transformations.

---

### `attitude`

Convert between attitude representations (quaternions, Euler angles, rotation matrices).

**Status:** Not yet implemented - planned for future release.

Planned conversions:
- Quaternion ↔ Rotation Matrix
- Quaternion ↔ Euler Angles (12 sequences)
- Quaternion ↔ Euler Axis-Angle
- And all inverse combinations

---

## Coordinate System Details

### Keplerian Elements

**Format:** `[a, e, i, Ω, ω, ν]`

- `a` - Semi-major axis (meters)
- `e` - Eccentricity (dimensionless, 0 ≤ e < 1)
- `i` - Inclination (degrees or radians)
- `Ω` - Right Ascension of Ascending Node / RAAN (degrees or radians)
- `ω` - Argument of periapsis (degrees or radians)
- `ν` - True anomaly (degrees or radians)

**Standard orbits:**
- LEO (500km): `a = R_EARTH + 500e3 = 6878137 m`
- GEO (35786km): `a = R_EARTH + 35786e3 = 42164137 m`
- SSO inclination: ~97.8° (for 500km altitude)

**Example:**
```bash
# Sun-synchronous orbit, 500km altitude, 97.8° inclination
brahe transform coordinates keplerian cartesian "" 6878137 0.001 97.8 0 0 0 --as-degrees
```

### Cartesian (ECI/ECEF)

**Format:** `[x, y, z, vx, vy, vz]`

- `x, y, z` - Position (meters)
- `vx, vy, vz` - Velocity (meters/second)

**ECI (Earth-Centered Inertial):**
- Inertial reference frame
- Z-axis aligned with Earth's rotation axis
- X-axis points to vernal equinox

**ECEF (Earth-Centered Earth-Fixed):**
- Rotating with Earth
- Z-axis aligned with rotation axis
- X-axis through 0° latitude, 0° longitude

**Example:**
```bash
# Circular equatorial orbit in ECI
brahe transform coordinates cartesian keplerian "" 6878137 0 0 0 7668 0 --as-degrees
```

### Geodetic Coordinates

**Format:** `[lat, lon, alt, 0, 0, 0]`

- `lat` - Geodetic latitude (degrees or radians)
- `lon` - Longitude (degrees or radians, positive east)
- `alt` - Altitude above WGS84 ellipsoid (meters)
- Last 3 values unused (set to 0)

**Geodetic vs Geocentric:**
- Geodetic: Perpendicular to WGS84 ellipsoid
- Geocentric: Angle from Earth's center

**Example:**
```bash
# New York City: 40.7128°N, 73.9060°W (= 286.0940°E)
brahe transform coordinates --as-degrees geodetic cartesian "" 40.7128 286.094 10 0 0 0 --to-frame ECEF
```

**Note:** For negative longitudes, either convert to positive east (add 360°) or use `--` separator:
```bash
brahe transform coordinates --as-degrees geodetic cartesian "" -- 40.7128 -73.9060 10 0 0 0 --to-frame ECEF
```

### Geocentric Coordinates

**Format:** `[lat, lon, radius, 0, 0, 0]`

- `lat` - Geocentric latitude (degrees or radians)
- `lon` - Longitude (degrees or radians, positive east)
- `radius` - Distance from Earth center (meters)
- Last 3 values unused (set to 0)

**Example:**
```bash
# Convert geocentric to geodetic
brahe transform coordinates --as-degrees geocentric geodetic "" 40.0 285.0 6478137 0 0 0
```

---

## Conversion Matrix

Which conversions are supported and what they require:

| From → To | Keplerian | Cartesian (ECI) | Cartesian (ECEF) | Geodetic | Geocentric |
|-----------|-----------|-----------------|------------------|----------|------------|
| **Keplerian** | - | Direct | Epoch† | Epoch† | Epoch† |
| **Cartesian (ECI)** | Direct | - | Epoch† | Epoch† | Epoch† |
| **Cartesian (ECEF)** | Epoch† | Epoch† | - | Direct | Direct |
| **Geodetic** | N/A* | Epoch† | Direct | - | Direct |
| **Geocentric** | N/A* | Epoch† | Direct | Direct | - |

- `Direct` = No epoch required (use `""` for EPOCH argument)
- `Epoch†` = Requires epoch in ISO-8601 format
- `N/A*` = Not physically meaningful (position-only → velocity-dependent)

---

## Common Workflows

### Satellite Ground Track

Determine where a satellite is above the Earth:

```bash
#!/bin/bash
# Satellite in Keplerian elements (SSO, 500km)
KEP="6878137 0.001 97.8 0 0 45"
EPOCH="2024-01-01T00:00:00Z"

# Convert to geodetic position
brahe transform coordinates --as-degrees --to-frame ECEF \
  keplerian geodetic "$EPOCH" $KEP
```

### Orbit Analysis at Different Anomalies

```bash
#!/bin/bash
# Orbit parameters
SMA="6878137"    # Semi-major axis (500km altitude)
ECC="0.01"       # Eccentricity (elliptical)
INC="63.4"       # Inclination (Molniya)

# State at perigee (ν = 0°)
echo "Perigee:"
brahe transform coordinates keplerian cartesian "" $SMA $ECC $INC 0 0 0 --as-degrees

# State at apogee (ν = 180°)
echo "Apogee:"
brahe transform coordinates keplerian cartesian "" $SMA $ECC $INC 0 0 180 --as-degrees
```

### Ground Station to ECI

Convert ground station location to ECI at specific epoch:

```bash
#!/bin/bash
# Svalbard ground station: 78.23°N, 15.39°E, 500m altitude
LAT="78.23"
LON="15.39"
ALT="500"
EPOCH="2024-06-21T12:00:00Z"  # Summer solstice

brahe transform coordinates --as-degrees --to-frame ECI \
  geodetic cartesian "$EPOCH" $LAT $LON $ALT 0 0 0
```

### ECI State History

Track how ECEF coordinates change over time:

```bash
#!/bin/bash
# Satellite ECI state
STATE="6878137 0 0 0 7500 0"

# Different epochs (6-hour intervals)
for hour in 0 6 12 18; do
  epoch="2024-01-01T$(printf "%02d" $hour):00:00Z"
  echo "Epoch $epoch:"
  brahe transform frame ECI ECEF "$epoch" $STATE
  echo
done
```

---

## Tips

### Handling Negative Values

Shell arguments starting with `-` are interpreted as options. To pass negative numbers:

**Method 1:** Use `--` separator (options must come before `--`):
```bash
brahe transform coordinates --as-degrees cartesian keplerian "" -- -1000 500 0 0 0 0
```

**Method 2:** Convert to positive equivalents:
```bash
# Longitude -74° = 286°
brahe transform coordinates geodetic cartesian "" 40.7128 286.0 10 0 0 0 --as-degrees
```

### When Epoch is Required

- **Frame transformations** (ECI ↔ ECEF): Always required
- **Geodetic/Geocentric** involving ECI: Always required
- **Pure Keplerian ↔ Cartesian (ECI)**: Not required (use `""`)
- **Pure Geodetic ↔ Geocentric**: Not required (use `""`)

### Angle Format Consistency

`--as-degrees` applies to **both input and output**:
```bash
# Input in degrees, output in degrees
brahe transform coordinates --as-degrees keplerian cartesian "" 6878137 0.001 97.8 0 0 0

# Input in radians, output in radians
brahe transform coordinates --no-as-degrees keplerian cartesian "" 6878137 0.001 1.706 0 0 0
```

### Using Constants

All numeric arguments support Brahe constants:
```bash
# Use R_EARTH constant
brahe transform coordinates keplerian cartesian "" "R_EARTH+500e3" 0.001 97.8 0 0 0 --as-degrees

# Use multiple constants
brahe transform coordinates keplerian cartesian "" "R_EARTH+35786e3" 0.0001 0.1 0 0 0 --as-degrees
```

### Output Precision

Control precision for debugging or analysis:
```bash
# Scientific notation
brahe transform frame ECI ECEF "2024-01-01T00:00:00Z" 6878137 0 0 0 7500 0 --format .3e

# High precision (10 decimal places)
brahe transform frame ECI ECEF "2024-01-01T00:00:00Z" 6878137 0 0 0 7500 0 --format .10f
```

---

## See Also

- [Coordinate Systems](../coordinates/index.md) - Conceptual overview
- [Reference Frames](../frame_transformations.md) - ECI/ECEF details
- [Coordinates API](../../library_api/coordinates/index.md) - Python API
- [Frames API](../../library_api/frames.md) - Frame conversion functions
- [Orbits API](../../library_api/orbits/index.md) - Orbital elements
- [Time CLI](time.md) - Time conversions
- [Orbits CLI](orbits.md) - Orbital mechanics calculations
