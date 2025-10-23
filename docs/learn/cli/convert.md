# Convert Subcommand

Coordinate system and reference frame conversion commands.

## Overview

The `convert` subcommand provides utilities for transforming coordinates between different coordinate systems and reference frames. This includes conversions between Keplerian and Cartesian, geodetic and ECEF, and ECI and ECEF frames.

## Commands

### `frame`

Convert state vectors between reference frames (ECI ↔ ECEF).

**Syntax:**
```bash
brahe convert frame <epoch> <x> <y> <z> <vx> <vy> <vz> <from-frame> <to-frame> [options]
```

**Arguments:**
- `epoch` - Epoch for the conversion (epoch-like string)
- `x y z vx vy vz` - State vector components [m, m, m, m/s, m/s, m/s]
- `from-frame` - Source reference frame: `ECI` or `ECEF`
- `to-frame` - Target reference frame: `ECI` or `ECEF`

**Options:**
- `--format <fmt>` - Output format string (default: `f`)

**Examples:**

```bash
# ECI to ECEF conversion
brahe convert frame "2024-01-01T00:00:00" \
    6878137 0 0 0 7500 0 \
    ECI ECEF

# ECEF to ECI conversion
brahe convert frame "2024-01-01T12:00:00" \
    6878137 0 0 0 7500 0 \
    ECEF ECI

# High precision output
brahe convert frame "2024-01-01T00:00:00" \
    6878137 0 0 0 7500 0 \
    ECI ECEF --format .3f
```

**Output:**
```
[6145123.456, 3456789.012, 0.000, -2567.891, 5678.123, 0.000]
```

**Note:** Frame conversions require Earth Orientation Parameters (EOP). The CLI automatically downloads and caches EOP data.

---

### `coordinates`

Convert between coordinate systems.

**Syntax:**
```bash
brahe convert coordinates <x1> <x2> <x3> <x4> <x5> <x6> <from-system> <to-system> [options]
```

**Arguments:**
- `x1 x2 x3 x4 x5 x6` - Coordinate values (interpretation depends on system)
- `from-system` - Source coordinate system
- `to-system` - Target coordinate system

**Coordinate Systems:**
- `keplerian` - [a, e, i, Ω, ω, ν] (meters, dimensionless, angles)
- `cartesian` / `eci` - [x, y, z, vx, vy, vz] (meters, m/s)
- `ecef` - [x, y, z, vx, vy, vz] (meters, m/s) in Earth-fixed frame
- `geodetic` - [lat, lon, alt, 0, 0, 0] (angles, meters)
- `geocentric` - [lat, lon, radius, 0, 0, 0] (angles, meters)

**Options:**
- `--epoch <epoch>` - Epoch for time-dependent conversions (required for some conversions)
- `--as-degrees` - Treat angles as degrees (default: `True`)
- `--format <fmt>` - Output format string

**Examples:**

```bash
# Keplerian to Cartesian (ECI)
brahe convert coordinates \
    6878137 0.001 97.8 0 0 0 \
    keplerian cartesian \
    --as-degrees

# Geodetic to ECEF
brahe convert coordinates \
    40.0 -105.0 1000 0 0 0 \
    geodetic ecef \
    --as-degrees

# ECEF to ECI (requires epoch)
brahe convert coordinates \
    6878137 0 0 0 7500 0 \
    ecef eci \
    --epoch "2024-01-01T00:00:00"

# Cartesian to Keplerian
brahe convert coordinates \
    6878137 0 0 0 7500 0 \
    cartesian keplerian \
    --as-degrees --format .6f
```

**Output:**
```
[6878137.000, 0.001234, 97.800000, 0.000000, 0.000000, 0.000000]
```

---

### `attitude`

Convert between attitude representations (not yet implemented).

**Status:** Planned for future release

## Coordinate System Details

### Keplerian Elements

**Format:** `[a, e, i, Ω, ω, ν]`

- `a` - Semi-major axis (meters)
- `e` - Eccentricity (dimensionless, 0 ≤ e < 1)
- `i` - Inclination (radians or degrees)
- `Ω` - Right Ascension of Ascending Node (RAAN, radians or degrees)
- `ω` - Argument of periapsis (radians or degrees)
- `ν` - True anomaly (radians or degrees)

**Example:**
```bash
# LEO sun-synchronous orbit
brahe convert coordinates \
    6878137 0.001 97.8 0 0 45 \
    keplerian cartesian --as-degrees
```

---

### Cartesian (ECI)

**Format:** `[x, y, z, vx, vy, vz]`

- `x, y, z` - Position in Earth-Centered Inertial frame (meters)
- `vx, vy, vz` - Velocity in ECI frame (m/s)

**Example:**
```bash
# Circular equatorial orbit
brahe convert coordinates \
    6878137 0 0 0 7668 0 \
    cartesian keplerian --as-degrees
```

---

### ECEF

**Format:** `[x, y, z, vx, vy, vz]`

- `x, y, z` - Position in Earth-Centered Earth-Fixed frame (meters)
- `vx, vy, vz` - Velocity in ECEF frame (m/s)

**Note:** Requires `--epoch` when converting to/from ECI.

---

### Geodetic

**Format:** `[lat, lon, alt, 0, 0, 0]`

- `lat` - Geodetic latitude (radians or degrees)
- `lon` - Longitude (radians or degrees)
- `alt` - Altitude above WGS84 ellipsoid (meters)
- Last 3 values are unused (set to 0)

**Example:**
```bash
# Boulder, Colorado
brahe convert coordinates \
    40.0150 -105.2705 1655 0 0 0 \
    geodetic ecef --as-degrees
```

---

### Geocentric

**Format:** `[lat, lon, radius, 0, 0, 0]`

- `lat` - Geocentric latitude (radians or degrees)
- `lon` - Longitude (radians or degrees)
- `radius` - Distance from Earth center (meters)
- Last 3 values are unused (set to 0)

## Common Workflows

### Satellite Position Analysis

```bash
#!/bin/bash
# Convert satellite Keplerian elements to ground position

# Satellite Keplerian elements
a=6878137
e=0.001
i=97.8
raan=0
argp=0
nu=0

# Convert to ECI
eci=$(brahe convert coordinates $a $e $i $raan $argp $nu keplerian eci --as-degrees)

echo "ECI State: $eci"

# Extract position (first 3 elements)
# Then convert to geodetic to get ground position
# (requires additional parsing - demonstration purpose)
```

### Ground Station Coordinates

```bash
#!/bin/bash
# Convert ground station locations

stations=(
    "Svalbard:78.2232:15.6267:500"
    "Singapore:1.3521:103.8198:50"
    "Hawaii:19.8968:-155.5828:100"
)

echo "Ground Station ECEF Coordinates"
echo "================================"

for station in "${stations[@]}"; do
    IFS=':' read -r name lat lon alt <<< "$station"

    ecef=$(brahe convert coordinates $lat $lon $alt 0 0 0 geodetic ecef --as-degrees --format .2f)

    echo "$name: $ecef"
done
```

### Orbit Comparison

```bash
#!/bin/bash
# Compare two orbits at different times

orbit1="6878137 0.001 97.8 0 0 0"
orbit2="6878137 0.001 97.8 0 0 90"

echo "Orbit 1 (ν=0°):"
brahe convert coordinates $orbit1 keplerian cartesian --as-degrees

echo ""
echo "Orbit 2 (ν=90°):"
brahe convert coordinates $orbit2 keplerian cartesian --as-degrees
```

## Conversion Chart

| From → To | Keplerian | Cartesian/ECI | ECEF | Geodetic | Geocentric |
|-----------|-----------|---------------|------|----------|------------|
| **Keplerian** | - | Direct | via ECI† | via ECEF† | via ECEF† |
| **Cartesian/ECI** | Direct | - | via epoch† | via ECEF† | via ECEF† |
| **ECEF** | via ECI† | via epoch† | - | Direct | Direct |
| **Geodetic** | N/A* | via ECEF† | Direct | - | via ECEF |
| **Geocentric** | N/A* | via ECEF† | Direct | via ECEF | - |

† = Requires `--epoch` parameter
\* = Not physically meaningful (position only, no velocity)

## Notes

- **Angle Units**: Use `--as-degrees` to work in degrees instead of radians
- **EOP Data**: Frame conversions (ECI ↔ ECEF) automatically download EOP data on first use
- **Epoch Required**: Conversions between ECI and ECEF reference frames require an epoch
- **Velocity Components**: Geodetic and geocentric only describe positions. Velocity components should be set to 0.
- **Format**: Use `--format` for scientific notation (`.3e`) or fixed precision (`.6f`)

## See Also

- [Coordinate Transformations](../coordinates/index.md) - Conceptual overview
- [Reference Frame Transformations](../frame_transformations.md) - ECI/ECEF details
- [Coordinates API](../../library_api/coordinates/index.md) - Python API
- [Frames API](../../library_api/frames.md) - Frame conversion functions
