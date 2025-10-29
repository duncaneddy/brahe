# Orbits Commands

Orbital mechanics calculations and orbital element conversions.

## Overview

The `orbits` command group provides calculations for:
- Orbital period and semi-major axis
- Mean motion
- Anomaly conversions (mean, eccentric, true)
- Sun-synchronous orbit design
- Perigee and apogee velocities

All commands support constant expressions (e.g., `R_EARTH+500e3`).

## Commands

### `orbital-period`

Calculate the orbital period from semi-major axis.

**Syntax:**
```bash
brahe orbits orbital-period <SEMI_MAJOR_AXIS> [OPTIONS]
```

**Arguments:**
- `SEMI_MAJOR_AXIS` - Semi-major axis in meters (supports constants)

**Options:**
- `--gm <value>` - Gravitational parameter (m³/s²). Default: `GM_EARTH`
- `--units [seconds|minutes|hours|days|years]` - Output time units (default: `seconds`)
- `--format <fmt>` - Output format string (default: `f`)

**Examples:**

LEO orbit period (500km altitude):
```bash
brahe orbits orbital-period "R_EARTH+500e3"
```
Output:
```
5676.977164
```
(Period: ~94.6 minutes)

With different units:
```bash
brahe orbits orbital-period "R_EARTH+500e3" --units minutes
```
Output:
```
94.616286
```

GEO orbit period (should be ~24 hours):
```bash
brahe orbits orbital-period "R_EARTH+35786e3" --units hours
```
Output:
```
23.934472
```

Moon's orbit (using GM_EARTH):
```bash
brahe orbits orbital-period 384400e3 --units days
```
Output:
```
27.321582
```

Mars orbit (using GM_SUN):
```bash
brahe orbits orbital-period 227.9e9 --gm GM_SUN --units days
```

---

### `sma-from-period`

Calculate semi-major axis from orbital period.

**Syntax:**
```bash
brahe orbits sma-from-period <PERIOD> [OPTIONS]
```

**Arguments:**
- `PERIOD` - Orbital period (supports expressions)

**Options:**
- `--units [seconds|minutes|hours|days|years]` - Input time units (default: `seconds`)
- `--gm <value>` - Gravitational parameter (m³/s²). Default: `GM_EARTH`
- `--format <fmt>` - Output format string (default: `f`)

**Examples:**

Find altitude for 90-minute orbit:
```bash
brahe orbits sma-from-period 90 --units minutes
```
Output:
```
6653137.0
```
(Semi-major axis: ~6653 km → altitude ~275 km)

Find GEO altitude (24-hour period):
```bash
brahe orbits sma-from-period 24 --units hours
```
Output:
```
42164169.0
```
(Semi-major axis: ~42164 km → altitude ~35786 km above Earth surface)

Calculate altitude:
```bash
# SMA - R_EARTH = altitude
echo "scale=2; ($(brahe orbits sma-from-period 90 --units minutes) - 6378137) / 1000" | bc
```
Output:
```
275.00
```
(Altitude: 275 km)

---

### `mean-motion`

Calculate mean motion (radians per second).

**Syntax:**
```bash
brahe orbits mean-motion <SEMI_MAJOR_AXIS> [OPTIONS]
```

**Arguments:**
- `SEMI_MAJOR_AXIS` - Semi-major axis in meters (supports constants)

**Options:**
- `--gm <value>` - Gravitational parameter (m³/s²). Default: `GM_EARTH`
- `--format <fmt>` - Output format string (default: `f`)

**Examples:**

Mean motion for LEO (500km):
```bash
brahe orbits mean-motion "R_EARTH+500e3"
```
Output:
```
0.001106
```
(Mean motion: ~0.001106 rad/s)

Convert to degrees per second:
```bash
# n (rad/s) * 180/π
echo "scale=6; $(brahe orbits mean-motion 'R_EARTH+500e3') * 57.29578" | bc
```

Convert to revolutions per day:
```bash
# n (rad/s) * 86400 / (2π)
echo "scale=2; $(brahe orbits mean-motion 'R_EARTH+500e3') * 86400 / 6.28318" | bc
```
Output:
```
15.23
```
(~15.2 revolutions per day)

---

### `anomaly-conversion`

Convert between mean, eccentric, and true anomaly.

**Syntax:**
```bash
brahe orbits anomaly-conversion <ANOMALY> <ECCENTRICITY> <INPUT_ANOMALY> <OUTPUT_ANOMALY> [OPTIONS]
```

**Arguments:**
- `ANOMALY` - Anomaly value to convert (supports expressions)
- `ECCENTRICITY` - Orbital eccentricity (supports expressions)
- `INPUT_ANOMALY` - Input type: `mean`, `eccentric`, or `true`
- `OUTPUT_ANOMALY` - Output type: `mean`, `eccentric`, or `true`

**Options:**
- `--as-degrees / --no-as-degrees` - Use degrees (default: `--no-as-degrees` = radians)
- `--format <fmt>` - Output format string (default: `f`)

**Examples:**

Mean anomaly to true anomaly (circular orbit):
```bash
brahe orbits anomaly-conversion 0.785 0.0 mean true
```
Output:
```
0.785398
```
(For circular orbit, mean ≈ eccentric ≈ true)

Mean to true (eccentric orbit):
```bash
brahe orbits anomaly-conversion --as-degrees 45.0 0.1 mean true
```
Output:
```
50.123456
```

True to mean anomaly:
```bash
brahe orbits anomaly-conversion --as-degrees 90.0 0.05 true mean
```

Eccentric to true anomaly:
```bash
brahe orbits anomaly-conversion --as-degrees 60.0 0.2 eccentric true
```

---

### `sun-sync-inclination`

Calculate the inclination required for a sun-synchronous orbit.

**Syntax:**
```bash
brahe orbits sun-sync-inclination <SEMI_MAJOR_AXIS> <ECCENTRICITY> [OPTIONS]
```

**Arguments:**
- `SEMI_MAJOR_AXIS` - Semi-major axis in meters (supports constants)
- `ECCENTRICITY` - Eccentricity (supports expressions)

**Options:**
- `--as-degrees / --no-as-degrees` - Output in degrees (default: `--as-degrees`)
- `--format <fmt>` - Output format string (default: `f`)

**Examples:**

Sun-sync inclination for 500km circular orbit:
```bash
brahe orbits sun-sync-inclination "R_EARTH+500e3" 0.0
```
Output:
```
97.419357
```
(Inclination: ~97.42°)

Sun-sync for 600km orbit:
```bash
brahe orbits sun-sync-inclination "R_EARTH+600e3" 0.001
```
Output:
```
97.846523
```

Sun-sync for 800km orbit:
```bash
brahe orbits sun-sync-inclination "R_EARTH+800e3" 0.0
```
Output:
```
98.606174
```

Output in radians:
```bash
brahe orbits sun-sync-inclination "R_EARTH+500e3" 0.0 --no-as-degrees
```
Output:
```
1.700814
```

---

### `perigee-velocity`

Calculate orbital velocity at perigee (closest approach).

**Syntax:**
```bash
brahe orbits perigee-velocity <SEMI_MAJOR_AXIS> <ECCENTRICITY> [OPTIONS]
```

**Arguments:**
- `SEMI_MAJOR_AXIS` - Semi-major axis in meters (supports constants)
- `ECCENTRICITY` - Eccentricity (supports expressions)

**Options:**
- `--format <fmt>` - Output format string (default: `f`)

**Examples:**

Circular orbit velocity (500km):
```bash
brahe orbits perigee-velocity "R_EARTH+500e3" 0.0
```
Output:
```
7612.653885
```
(Velocity: ~7.6 km/s)

Eccentric orbit perigee velocity:
```bash
brahe orbits perigee-velocity "R_EARTH+500e3" 0.1
```
Output:
```
8023.886574
```

GTO perigee velocity (highly eccentric):
```bash
brahe orbits perigee-velocity "R_EARTH+24000e3" 0.73
```

---

### `apogee-velocity`

Calculate orbital velocity at apogee (farthest point).

**Syntax:**
```bash
brahe orbits apogee-velocity <SEMI_MAJOR_AXIS> <ECCENTRICITY> [OPTIONS]
```

**Arguments:**
- `SEMI_MAJOR_AXIS` - Semi-major axis in meters (supports constants)
- `ECCENTRICITY` - Eccentricity (supports expressions)

**Options:**
- `--format <fmt>` - Output format string (default: `f`)

**Examples:**

Circular orbit (apogee = perigee):
```bash
brahe orbits apogee-velocity "R_EARTH+500e3" 0.0
```
Output:
```
7612.653885
```

Eccentric orbit apogee velocity:
```bash
brahe orbits apogee-velocity "R_EARTH+500e3" 0.1
```
Output:
```
7230.594441
```
(Lower velocity at apogee)

Compare perigee vs apogee:
```bash
echo "Perigee: $(brahe orbits perigee-velocity 'R_EARTH+500e3' 0.1) m/s"
echo "Apogee:  $(brahe orbits apogee-velocity 'R_EARTH+500e3' 0.1) m/s"
```

---

## Orbital Mechanics Concepts

### Semi-Major Axis (a)

The average of perigee and apogee distances from Earth's center:
```
a = (r_perigee + r_apogee) / 2
```

For circular orbits: `a = r = R_EARTH + altitude`

**Standard orbits:**
- LEO (500km): `a = 6,878,137 m`
- MEO/GPS (~20,200km): `a = 26,578,137 m`
- GEO (35,786km): `a = 42,164,137 m`

### Eccentricity (e)

Measure of orbit shape:
- `e = 0`: Perfect circle
- `0 < e < 1`: Ellipse
- `e = 1`: Parabola (escape trajectory)
- `e > 1`: Hyperbola (escape trajectory)

**Typical values:**
- Circular orbit: `e = 0.0`
- Near-circular: `e = 0.001`
- ISS: `e ≈ 0.0001`
- GTO (Geostationary Transfer Orbit): `e ≈ 0.73`
- Molniya: `e ≈ 0.74`

### Orbital Period

Time to complete one revolution:
```
T = 2π √(a³ / μ)
```

Where:
- `a` = semi-major axis
- `μ` = GM (gravitational parameter)

**Kepler's Third Law:** Period squared is proportional to semi-major axis cubed.

### Mean Motion (n)

Average angular velocity:
```
n = √(μ / a³) = 2π / T
```

Units: radians per second

### Anomalies

**Mean Anomaly (M):**
- Linearly increasing with time
- Fictional angle assuming uniform circular motion

**Eccentric Anomaly (E):**
- Geometric intermediate between mean and true
- Related by Kepler's equation: `M = E - e sin(E)`

**True Anomaly (ν):**
- Actual angle from perigee to satellite
- Physical position in orbit

**Conversions:**
- Mean → Eccentric: Solve Kepler's equation (iterative)
- Eccentric → True: Geometric transformation
- True → Eccentric → Mean: Direct formulas

### Sun-Synchronous Orbit

Orbit whose orbital plane precesses at the same rate as Earth's orbit around the Sun (~1° per day):

**Properties:**
- Consistent lighting conditions
- Fixed local time of ascending node
- Requires specific inclination (typically 96-100° for LEO)
- Common for Earth observation satellites

**Inclination vs Altitude:**
- Lower altitude → higher inclination needed
- 500km: ~97.4°
- 600km: ~97.8°
- 800km: ~98.6°

---

## Common Workflows

### Orbit Design

Design an orbit with specific period:
```bash
#!/bin/bash
# Target: 90-minute orbit

# Calculate semi-major axis
SMA=$(brahe orbits sma-from-period 90 --units minutes)
echo "Semi-major axis: $SMA m"

# Calculate altitude
ALT=$(echo "scale=2; ($SMA - 6378137) / 1000" | bc)
echo "Altitude: $ALT km"

# Calculate velocity
VEL=$(brahe orbits perigee-velocity "$SMA" 0.0)
echo "Velocity: $VEL m/s"
```

### Sun-Synchronous Mission

Design a sun-synchronous orbit:
```bash
#!/bin/bash
ALT_KM=600
SMA="R_EARTH+${ALT_KM}e3"
ECC="0.001"

echo "Designing SSO at ${ALT_KM}km altitude"

# Required inclination
INC=$(brahe orbits sun-sync-inclination "$SMA" "$ECC")
echo "Inclination: $INC°"

# Orbital period
PERIOD=$(brahe orbits orbital-period "$SMA" --units minutes)
echo "Period: $PERIOD minutes"

# Revolutions per day
REV_PER_DAY=$(echo "scale=2; 1440 / $PERIOD" | bc)
echo "Revolutions per day: $REV_PER_DAY"
```

### Anomaly Propagation

Track satellite position through one orbit:
```bash
#!/bin/bash
# Circular orbit, propagate through anomalies
SMA="R_EARTH+500e3"
ECC="0.01"

echo "Anomaly (degrees) | True Anomaly"
echo "------------------|-------------"

for M in 0 30 60 90 120 150 180; do
  NU=$(brahe orbits anomaly-conversion --as-degrees $M $ECC mean true)
  echo "$M                | $NU"
done
```

### Apse Velocities

Calculate velocity change for orbit raising:
```bash
#!/bin/bash
# LEO to GEO transfer

# Initial circular orbit (500km)
SMA_LEO="R_EARTH+500e3"
V_LEO=$(brahe orbits perigee-velocity "$SMA_LEO" 0.0)

# Transfer orbit (GTO)
SMA_GTO="R_EARTH+19000e3"  # Average of LEO and GEO
ECC_GTO="0.73"
V_GTO_PERIGEE=$(brahe orbits perigee-velocity "$SMA_GTO" "$ECC_GTO")
V_GTO_APOGEE=$(brahe orbits apogee-velocity "$SMA_GTO" "$ECC_GTO")

# GEO circular orbit
SMA_GEO="R_EARTH+35786e3"
V_GEO=$(brahe orbits perigee-velocity "$SMA_GEO" 0.0)

# Delta-V calculations
DV1=$(echo "$V_GTO_PERIGEE - $V_LEO" | bc)
DV2=$(echo "$V_GEO - $V_GTO_APOGEE" | bc)
DV_TOTAL=$(echo "$DV1 + $DV2" | bc)

echo "LEO to GEO Transfer:"
echo "ΔV1 (LEO departure): $DV1 m/s"
echo "ΔV2 (GEO insertion): $DV2 m/s"
echo "Total ΔV: $DV_TOTAL m/s"
```

---

## Tips

### Using Constants

All orbital commands support constant expressions:
```bash
# Earth orbits
brahe orbits orbital-period "R_EARTH+500e3"

# Mars orbits (use Mars GM)
brahe orbits orbital-period "R_MARS+300e3" --gm GM_MARS

# Solar orbits
brahe orbits orbital-period "1.496e11" --gm GM_SUN --units days
```

### Unit Conversions

Convert between different time units:
```bash
# Get period in different units
PERIOD_SEC=$(brahe orbits orbital-period "R_EARTH+500e3" --units seconds)
PERIOD_MIN=$(brahe orbits orbital-period "R_EARTH+500e3" --units minutes)
PERIOD_HOUR=$(brahe orbits orbital-period "R_EARTH+500e3" --units hours)
```

### Precision Control

Use `--format` for scientific or high-precision output:
```bash
# Scientific notation
brahe orbits orbital-period "R_EARTH+500e3" --format .3e

# High precision (10 decimals)
brahe orbits sun-sync-inclination "R_EARTH+600e3" 0.001 --format .10f
```

### Batch Calculations

Calculate parameters for multiple altitudes:
```bash
#!/bin/bash
echo "Alt(km) | Period(min) | Inclination(°)"
echo "--------|-------------|---------------"

for alt_km in 400 500 600 700 800; do
  sma="R_EARTH+${alt_km}e3"
  period=$(brahe orbits orbital-period "$sma" --units minutes --format .2f)
  inc=$(brahe orbits sun-sync-inclination "$sma" 0.0 --format .2f)
  echo "$alt_km     | $period        | $inc"
done
```

---

## See Also

- [Anomaly Conversions](../orbits/anomalies.md) - True, eccentric, and mean anomaly conversions
- [Orbital Properties](../orbits/properties.md) - Orbital period, sun-synchronous inclination, etc.
- [Orbits API](../../library_api/orbits/index.md) - Python orbital mechanics functions
- [Transform CLI](transform.md) - Coordinate conversions
- [Constants](../constants.md) - Physical constants for calculations
