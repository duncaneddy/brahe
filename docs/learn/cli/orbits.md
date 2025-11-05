# Orbits Commands

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
```bash
# 5676.977164
```
(Period: ~94.6 minutes)

With different units:
```bash
brahe orbits orbital-period "R_EARTH+500e3" --units minutes
```
Output:
```bash
# 94.616286
```

GEO orbit period (should be ~24 hours):
```bash
brahe orbits orbital-period "R_EARTH+35786e3" --units hours
```
Output:
```bash
# 23.934441
```

Moon's orbit (using GM_EARTH):
```bash
brahe orbits orbital-period 384400e3 --units days
```
Output:
```bash
# 27.451894
```

Mars orbit (using GM_SUN):
```bash
brahe orbits orbital-period 227.9e9 --gm GM_SUN --units days
```
Output:
```bash
# 686.794481
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
```bash
# 6652555.699659
```
(Semi-major axis: ~6653 km → altitude ~275 km)

Find GEO altitude (24-hour period):
```bash
brahe orbits sma-from-period 24 --units hours
```
Output:
```bash
# 42241095.663660
```
(Semi-major axis: ~42164 km → altitude ~35786 km above Earth surface)

Calculate altitude:
```bash
# SMA - R_EARTH = altitude
echo "scale=2; ($(brahe orbits sma-from-period 90 --units minutes) - 6378137) / 1000" | bc
```
Output:
```bash
# 274.41
```

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
```bash
# 0.001107
```

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
```bash
# 0.785000
```
(For circular orbit, mean = eccentric = true)

Mean to true (eccentric orbit):
```bash
brahe orbits anomaly-conversion --as-degrees 45.0 0.1 mean true
```
Output:
```bash
# 53.849399
```

True to mean anomaly:
```bash
brahe orbits anomaly-conversion --as-degrees 90.0 0.05 true mean
```
Output:
```bash
# 84.272810
```

Eccentric to true anomaly:
```bash
brahe orbits anomaly-conversion --as-degrees 60.0 0.2 eccentric true
```
Output:
```bash
# 70.528779
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
```bash
# 97.401744
```
(Inclination: ~97.42°)

Sun-sync for 600km orbit:
```bash
brahe orbits sun-sync-inclination "R_EARTH+600e3" 0.001
```
Output:
```bash
# 97.787587
```

Sun-sync for 800km orbit:
```bash
brahe orbits sun-sync-inclination "R_EARTH+800e3" 0.0
```
Output:
```bash
# 98.603036
```

Output in radians:
```bash
brahe orbits sun-sync-inclination "R_EARTH+500e3" 0.0 --no-as-degrees
```
Output:
```bash
# 1.699981
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
```bash
# 7612.608558
```


Eccentric orbit perigee velocity:
```bash
brahe orbits perigee-velocity "R_EARTH+500e3" 0.1
```
Output:
```bash
# 8416.055421
```

GTO perigee velocity (highly eccentric):
```bash
brahe orbits perigee-velocity "R_EARTH+24000e3" 0.73
```
Output:
```bash
# 9169.158794
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
```bash
# 7612.608558
```

Eccentric orbit apogee velocity:
```bash
brahe orbits apogee-velocity "R_EARTH+500e3" 0.1
```
Output:
```bash
# 6885.863526
```
(Lower velocity at apogee)

Compare perigee vs apogee:
```bash
echo "Perigee: $(brahe orbits perigee-velocity 'R_EARTH+500e3' 0.1) m/s"
echo "Apogee:  $(brahe orbits apogee-velocity 'R_EARTH+500e3' 0.1) m/s"
```
Output:
```bash
# Perigee: 8416.055421 m/s
# Apogee:  6885.863526 m/s
```

---

---

## See Also

- [Anomaly Conversions](../orbits/anomalies.md) - True, eccentric, and mean anomaly conversions
- [Orbital Properties](../orbits/properties.md) - Orbital period, sun-synchronous inclination, etc.
- [Orbits API](../../library_api/orbits/index.md) - Python orbital mechanics functions
- [Transform CLI](transform.md) - Coordinate conversions
- [Constants](../constants.md) - Physical constants for calculations
