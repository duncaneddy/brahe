# Orbits Subcommand

Orbital mechanics calculations for common orbital parameters and properties.

## Overview

The `orbits` subcommand provides quick calculations for orbital mechanics, including orbital periods, semi-major axes, anomaly conversions, and specialized orbit design.

## Commands

### `orbital-period`

Calculate the orbital period from semi-major axis.

**Syntax:**
```bash
brahe orbits orbital-period <semi-major-axis> [options]
```

**Arguments:**
- `semi-major-axis` - Semi-major axis in meters

**Options:**
- `--gm <value>` - Gravitational parameter (m³/s²). Default: Earth's GM
- `--units <seconds|minutes|hours|days|years>` - Output time units (default: `seconds`)
- `--format <fmt>` - Output format string (default: `f`)

**Examples:**

```bash
# LEO orbit period (500 km altitude)
brahe orbits orbital-period 6878137 --units minutes

# GEO orbit period
brahe orbits orbital-period 42164000 --units hours

# Moon orbit period with custom GM
brahe orbits orbital-period 384400000 --gm 3.986004418e14 --units days
```

**Output:**
```
94.56  # Example: minutes
```

---

### `sma-from-period`

Calculate semi-major axis from orbital period.

**Syntax:**
```bash
brahe orbits sma-from-period <period> [options]
```

**Arguments:**
- `period` - Orbital period value

**Options:**
- `--units <seconds|minutes|hours|days|years>` - Input time units (default: `seconds`)
- `--gm <value>` - Gravitational parameter (m³/s²). Default: Earth's GM
- `--format <fmt>` - Output format string

**Examples:**

```bash
# Find semi-major axis for 90-minute orbit
brahe orbits sma-from-period 90 --units minutes

# GEO altitude (24-hour period)
brahe orbits sma-from-period 24 --units hours

# Format with scientific notation
brahe orbits sma-from-period 90 --units minutes --format .3e
```

**Output:**
```
6795364.123456  # Semi-major axis in meters
```

---

### `mean-motion`

Calculate mean motion (radians/second) from semi-major axis.

**Syntax:**
```bash
brahe orbits mean-motion <semi-major-axis> [options]
```

**Arguments:**
- `semi-major-axis` - Semi-major axis in meters

**Options:**
- `--gm <value>` - Gravitational parameter
- `--format <fmt>` - Output format string

**Examples:**

```bash
# Mean motion for LEO
brahe orbits mean-motion 6878137

# High precision output
brahe orbits mean-motion 6878137 --format .10f
```

**Output:**
```
0.001058821  # rad/s
```

---

### `anomaly-conversion`

Convert between mean, eccentric, and true anomaly.

**Syntax:**
```bash
brahe orbits anomaly-conversion <anomaly> <eccentricity> <input-type> <output-type> [options]
```

**Arguments:**
- `anomaly` - Anomaly value (in radians or degrees)
- `eccentricity` - Orbital eccentricity
- `input-type` - Input anomaly type: `mean`, `eccentric`, or `true`
- `output-type` - Output anomaly type: `mean`, `eccentric`, or `true`

**Options:**
- `--as-degrees` - Treat input/output as degrees (default: radians)
- `--format <fmt>` - Output format string

**Examples:**

```bash
# Mean to true anomaly (radians)
brahe orbits anomaly-conversion 1.5708 0.001 mean true

# True to eccentric (degrees)
brahe orbits anomaly-conversion 90.0 0.1 true eccentric --as-degrees

# Eccentric to mean
brahe orbits anomaly-conversion 45.0 0.05 eccentric mean --as-degrees --format .6f
```

**Output:**
```
89.912345  # Converted anomaly
```

---

### `sun-sync-inclination`

Calculate the inclination required for a sun-synchronous orbit.

**Syntax:**
```bash
brahe orbits sun-sync-inclination <semi-major-axis> <eccentricity> [options]
```

**Arguments:**
- `semi-major-axis` - Semi-major axis in meters
- `eccentricity` - Orbital eccentricity

**Options:**
- `--as-degrees` - Output in degrees (default: `True`)
- `--format <fmt>` - Output format string

**Examples:**

```bash
# Sun-sync inclination for 600 km altitude
brahe orbits sun-sync-inclination 6978137 0.001

# LEO sun-sync with high precision
brahe orbits sun-sync-inclination 6878137 0.0001 --format .4f
```

**Output:**
```
97.8123  # Inclination in degrees
```

---

### `perigee-velocity`

Calculate velocity at perigee (closest approach).

**Syntax:**
```bash
brahe orbits perigee-velocity <semi-major-axis> <eccentricity> [options]
```

**Arguments:**
- `semi-major-axis` - Semi-major axis in meters
- `eccentricity` - Orbital eccentricity

**Options:**
- `--format <fmt>` - Output format string

**Examples:**

```bash
# Perigee velocity for elliptical orbit
brahe orbits perigee-velocity 7000000 0.1

# High eccentricity orbit
brahe orbits perigee-velocity 10000000 0.5 --format .2f
```

**Output:**
```
7845.67  # Velocity in m/s
```

---

### `apogee-velocity`

Calculate velocity at apogee (farthest point).

**Syntax:**
```bash
brahe orbits apogee-velocity <semi-major-axis> <eccentricity> [options]
```

**Arguments:**
- `semi-major-axis` - Semi-major axis in meters
- `eccentricity` - Orbital eccentricity

**Options:**
- `--format <fmt>` - Output format string

**Examples:**

```bash
# Apogee velocity for elliptical orbit
brahe orbits apogee-velocity 7000000 0.1

# Highly elliptical orbit
brahe orbits apogee-velocity 26000000 0.7 --format .2f
```

**Output:**
```
6542.31  # Velocity in m/s
```

## Common Workflows

### Orbit Design

```bash
#!/bin/bash
# Design a sun-synchronous orbit at 600 km altitude

# Constants
R_EARTH=6378137
altitude=600000
sma=$((R_EARTH + altitude))
ecc=0.001

# Calculate orbit parameters
period=$(brahe orbits orbital-period $sma --units minutes)
inclination=$(brahe orbits sun-sync-inclination $sma $ecc)
mean_motion=$(brahe orbits mean-motion $sma)

echo "Sun-Synchronous Orbit Design"
echo "============================="
echo "Altitude: ${altitude}m"
echo "Semi-major axis: ${sma}m"
echo "Period: ${period} min"
echo "Inclination: ${inclination}°"
echo "Mean motion: ${mean_motion} rad/s"
```

### Orbit Analysis

```bash
#!/bin/bash
# Analyze orbit characteristics

sma=7000000
ecc=0.15

period=$(brahe orbits orbital-period $sma --units minutes)
perigee_vel=$(brahe orbits perigee-velocity $sma $ecc)
apogee_vel=$(brahe orbits apogee-velocity $sma $ecc)

# Calculate altitude range
R_EARTH=6378137
perigee_alt=$(echo "scale=0; $sma * (1 - $ecc) - $R_EARTH" | bc)
apogee_alt=$(echo "scale=0; $sma * (1 + $ecc) - $R_EARTH" | bc)

echo "Orbit Analysis"
echo "=============="
echo "Period: ${period} min"
echo "Perigee: ${perigee_alt}m @ ${perigee_vel} m/s"
echo "Apogee: ${apogee_alt}m @ ${apogee_vel} m/s"
```

### Anomaly Tracking

```bash
#!/bin/bash
# Track satellite position through orbit

ecc=0.05

echo "Mean Anomaly -> True Anomaly"
for M in 0 30 60 90 120 150 180; do
    nu=$(brahe orbits anomaly-conversion $M $ecc mean true --as-degrees --format .2f)
    echo "$M° -> $nu°"
done
```

## Notes

- **Units**: Semi-major axis is always in meters. Use `--units` to control time output.
- **Anomalies**: By default, anomalies are in radians. Use `--as-degrees` for degree input/output.
- **Gravitational Parameter**: Default is Earth's GM (3.986004418e14 m³/s²). Use `--gm` for other bodies.
- **Circular Orbits**: For circular orbits, set eccentricity = 0.

## Orbital Element Definitions

- **Semi-major Axis (a)**: Average of periapsis and apoapsis distances
- **Eccentricity (e)**: Shape of orbit (0 = circular, 0 < e < 1 = elliptical)
- **Mean Anomaly (M)**: Fictitious angle increasing uniformly with time
- **Eccentric Anomaly (E)**: Auxiliary angle in orbital mechanics
- **True Anomaly (ν)**: Actual angle from periapsis to satellite

## See Also

- [Orbital Properties](../orbits/orbital_properties.md) - Detailed orbital mechanics
- [Keplerian Elements](../../library_api/orbits/keplerian.md) - Python API
- [Keplerian Propagation](../orbit_propagation/keplerian_propagation.md) - Orbit propagation
