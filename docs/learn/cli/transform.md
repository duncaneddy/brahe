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
```bash
# [-1176064.179304, -6776827.196931, 15961.825213, 6895.376569, -1196.636908, 0.240602]
```

Convert ECEF back to ECI:
```bash
brahe transform frame ECEF ECI "2024-01-01T00:00:00Z" -- -1176064.179 -6776827.197 15961.825 6895.377 -1196.637 0.241
```
Output:
```bash
# [6878137.000016, 0.000312, -0.000213, 0.000018, 7500.000440, 0.000398]
```

Low-precision output:
```bash
brahe transform frame ECI ECEF "2024-01-01T12:00:00Z" 6878137 0 0 0 7500 0 --format .2f
```
Output:
```bash
# [1234308.01, 6766461.20, 15974.24, -6884.83, 1255.90, 0.25]
```

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
- `geodetic` - Geodetic coordinates [lon, lat, alt, 0, 0, 0]
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
```bash
# [6871258.863000, 0.000000, 0.000000, 0.000000, -1034.183142, 7549.721055]
```

With different true anomaly (45°):
```bash
brahe transform coordinates keplerian cartesian "" 6878137 0.001 97.8 0 0 45 --as-degrees
```
Output:
```bash
# [4853256.459155, -660529.749078, 4821984.763637, -5390.543500, -730.545720, 5333.113819]
```

#### Cartesian to Keplerian

Convert ECI state back to orbital elements:
```bash
brahe transform coordinates --as-degrees cartesian keplerian "" -- 6871258.863 0.0 0.0 0.0 -1034.183 7549.721
```
Output:
```bash
# [6878136.866355, 0.001000, 97.799999, 0.000000, 0.000000, 0.000000]
```

#### Geodetic to Cartesian (ECEF)

Convert geodetic coordinates (New York City) to ECEF:
```bash
brahe transform coordinates --as-degrees --to-frame ECEF geodetic cartesian "" 286.0060 40.7128 10 0 0 0
```
Output:
```bash
# [1334224.912305, -4651969.287142, 4140677.827068]
```

**Note:** Order is [lon, lat, alt]. Longitude 286° = -74° (use positive east longitude, or handle negative with `--`)

#### ECEF Cartesian to Geodetic

```bash
brahe transform coordinates --as-degrees --from-frame ECEF cartesian geodetic "2024-01-01T00:00:00Z" 1334915.0 4652372.0 4075345.0 0 0 0
```
Output:
```bash
# [73.990114, 40.288227, -41917.492259]
```

#### Keplerian to Geodetic (via ECEF)

Convert satellite orbital elements to ground track position at epoch:
```bash
# First to cartesian ECI, then specify ECEF and geodetic
brahe transform coordinates --as-degrees --to-frame ECEF keplerian geodetic "2024-01-01T00:00:00Z" 6878137 0.001 97.8 0 0 0
```
Output:
```bash
# [-99.845171, 0.133796, 493121.978692]
```

#### Cartesian ECI to ECEF (frame change)

```bash
brahe transform coordinates --from-frame ECI --to-frame ECEF cartesian cartesian "2024-01-01T00:00:00Z" 6878137 0 0 0 7500 0
```
Output:
```bash
# [-1176064.179304, -6776827.196931, 15961.825213, 6895.376569, -1196.636908, 0.240602]
```

**Alternative:** Use `brahe transform frame` for dedicated ECI↔ECEF transformations.

---

## Coordinate System Details

### Keplerian Elements

**Format:** `[a, e, i, Ω, ω, M]`

- `a` - Semi-major axis (meters)
- `e` - Eccentricity (dimensionless, 0 ≤ e < 1)
- `i` - Inclination (degrees or radians)
- `Ω` - Right Ascension of Ascending Node / RAAN (degrees or radians)
- `ω` - Argument of periapsis (degrees or radians)
- `M` - Mean anomaly (degrees or radians)

**Standard orbits:**
- LEO (500km): `a = R_EARTH + 500e3 = 6878137 m`
- GEO (35786km): `a = R_EARTH + 35786e3 = 42164137 m`
- SSO inclination: ~97.8° (for 500km altitude)

**Example:**
```bash
# Sun-synchronous orbit, 500km altitude, 97.8° inclination
brahe transform coordinates keplerian cartesian "" 6878137 0.001 97.8 0 0 0 --as-degrees
```
Output:
```bash
# [6871258.863000, 0.000000, 0.000000, 0.000000, -1034.183142, 7549.721055]
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
- Z-axis aligned with Earth's rotation axis
- X-axis through 0° latitude, 0° longitude

**Example:**
```bash
# Circular equatorial orbit in ECI
brahe transform coordinates cartesian keplerian "" 6878137 0 0 0 7668 0 --as-degrees
```
Output:
```bash
# [6980085.332943, 0.014606, 0.000000, 180.000000, 0.000000, 0.000000]
```

### Geodetic Coordinates

**Format:** `[lon, lat, alt, 0, 0, 0]`

- `lon` - Longitude (degrees or radians, positive east)
- `lat` - Geodetic latitude (degrees or radians)
- `alt` - Altitude above WGS84 ellipsoid (meters)
- Last 3 values unused (set to 0)

**Geodetic vs Geocentric:**
- Geodetic: Perpendicular to WGS84 ellipsoid
- Geocentric: Angle from Earth's center

### Geocentric Coordinates

**Format:** `[lat, lon, radius, 0, 0, 0]`

- `lat` - Geocentric latitude (degrees or radians)
- `lon` - Longitude (degrees or radians, positive east)
- `radius` - Distance from Earth center (meters)
- Last 3 values unused (set to 0)

---

## See Also

- [Coordinate Systems](../coordinates/index.md) - Conceptual overview
- [Reference Frames](../frame_transformations.md) - ECI/ECEF details
- [Coordinates API](../../library_api/coordinates/index.md) - Python API
- [Frames API](../../library_api/frames.md) - Frame conversion functions
- [Orbits API](../../library_api/orbits/index.md) - Orbital elements
- [Time CLI](time.md) - Time conversions
- [Orbits CLI](orbits.md) - Orbital mechanics calculations
