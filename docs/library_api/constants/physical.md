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

---

### AU

::: brahe.AU
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

**Value**: `1.495978707e11` m

Astronomical Unit - mean distance of Earth from the Sun. TDB-compatible value.

---

### P_SUN

::: brahe.P_SUN
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

**Value**: `4.56e-6` N/m²

Solar radiation pressure at 1 AU.

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

---

#### J2_EARTH

::: brahe.J2_EARTH
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `0.0010826358191967` (dimensionless)

Earth's J2 zonal harmonic coefficient (GGM05s gravity model). Represents Earth's oblateness.

---

#### OMEGA_EARTH

::: brahe.OMEGA_EARTH
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `7.292115146706979e-05` rad/s

Earth's axial rotation rate.

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
