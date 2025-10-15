# Physical Constants

Physical properties of celestial bodies and universal constants. All values use SI base units.

## Universal Constants

### C_LIGHT

::: brahe.C_LIGHT
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

**Value**: `299792458.0` m/s

Speed of light in vacuum.

**Example**:
```python
import brahe as bh

# Time for light to travel 1 AU
distance = bh.AU
time_seconds = distance / bh.C_LIGHT  # ~499.0 seconds (8.3 minutes)
```

---

### AU

::: brahe.AU
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

**Value**: `1.495978707e11` m

Astronomical Unit - mean distance of Earth from the Sun. TDB-compatible value.

**Example**:
```python
import brahe as bh

# Express orbital radius in AU
orbital_radius_m = 2.0e11  # meters
orbital_radius_au = orbital_radius_m / bh.AU  # ~1.34 AU
```

---

### P_SUN

::: brahe.P_SUN
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

**Value**: `4.56e-6` N/m²

Solar radiation pressure at 1 AU.

**Example**:
```python
import brahe as bh

# Solar radiation pressure force on a satellite
area = 10.0  # m²
reflectivity = 1.3  # coefficient
force = bh.P_SUN * area * reflectivity  # Newtons
```

---

## Earth Constants

### Geometry

#### R_EARTH

::: brahe.R_EARTH
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `6378136.3` m

Earth's equatorial radius (GGM05 gravity model).

---

#### WGS84_A

::: brahe.WGS84_A
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `6378137.0` m

Earth's semi-major axis as defined by WGS84 geodetic system.

---

#### WGS84_F

::: brahe.WGS84_F
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `0.0033528106647474805` (dimensionless)

Earth's ellipsoidal flattening. WGS84 value: 1/298.257223563

---

#### ECC_EARTH

::: brahe.ECC_EARTH
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `0.081819190842622` (dimensionless)

Earth's first eccentricity (WGS84 value).

---

### Gravitational Properties

#### GM_EARTH

::: brahe.GM_EARTH
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `3.986004415e14` m³/s²

Earth's gravitational parameter (μ = G × M).

**Example**:
```python
import brahe as bh
import math

# Calculate orbital period for LEO satellite
altitude = 400e3  # 400 km altitude
radius = bh.R_EARTH + altitude
period = 2 * math.pi * math.sqrt(radius**3 / bh.GM_EARTH)
print(f"Orbital period: {period/60:.1f} minutes")  # ~92.5 minutes
```

---

#### J2_EARTH

::: brahe.J2_EARTH
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `0.0010826358191967` (dimensionless)

Earth's J2 zonal harmonic coefficient (GGM05s gravity model). Represents Earth's oblateness.

**Example**:
```python
import brahe as bh

# J2 perturbations are significant for orbit propagation
# Used in analytical orbit propagators for secular effects
```

---

#### OMEGA_EARTH

::: brahe.OMEGA_EARTH
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `7.292115146706979e-05` rad/s

Earth's axial rotation rate.

**Example**:
```python
import brahe as bh
import math

# Sidereal day
sidereal_day = 2 * math.pi / bh.OMEGA_EARTH  # ~86164.1 seconds
```

---

## Celestial Body Gravitational Parameters

Gravitational parameters (μ = G × M) for major solar system bodies in m³/s².

### Sun

#### GM_SUN

::: brahe.GM_SUN
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `1.32712440041939e20` m³/s²

---

#### R_SUN

::: brahe.R_SUN
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `6.9634e8` m

Solar radius.

---

### Moon

#### GM_MOON

::: brahe.GM_MOON
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `4.9028e12` m³/s²

---

#### R_MOON

::: brahe.R_MOON
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `1.738e6` m

Lunar radius.

---

### Inner Planets

#### GM_MERCURY

::: brahe.GM_MERCURY
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `2.2031868551e13` m³/s²

---

#### GM_VENUS

::: brahe.GM_VENUS
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `3.257e14` m³/s²

---

#### GM_MARS

::: brahe.GM_MARS
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `4.305e13` m³/s²

---

### Outer Planets

#### GM_JUPITER

::: brahe.GM_JUPITER
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `1.268e17` m³/s²

---

#### GM_SATURN

::: brahe.GM_SATURN
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `3.794e16` m³/s²

---

#### GM_URANUS

::: brahe.GM_URANUS
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `5.794e15` m³/s²

---

#### GM_NEPTUNE

::: brahe.GM_NEPTUNE
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `6.837e15` m³/s²

---

### Dwarf Planets

#### GM_PLUTO

::: brahe.GM_PLUTO
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `9.77e11` m³/s²

---

## Usage Example

```python
import brahe as bh
import numpy as np

# Third-body perturbation calculation
def third_body_accel(r_sat, r_body, gm_body):
    """Calculate third-body gravitational acceleration."""
    r_sat_body = r_body - r_sat
    r_sat_body_mag = np.linalg.norm(r_sat_body)
    r_body_mag = np.linalg.norm(r_body)

    return gm_body * (r_sat_body / r_sat_body_mag**3 - r_body / r_body_mag**3)

# Example: Moon's perturbation on LEO satellite
r_sat = np.array([bh.R_EARTH + 400e3, 0, 0])  # LEO satellite position
r_moon = np.array([384400e3, 0, 0])  # Approximate Moon distance
a_moon = third_body_accel(r_sat, r_moon, bh.GM_MOON)
```
