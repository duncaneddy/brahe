# Mathematical Constants

Conversion factors for angles and other mathematical operations.

## Angle Conversions

### DEG2RAD

::: brahe.DEG2RAD
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

**Value**: `0.017453292519943295` rad/deg

Converts degrees to radians. Equivalent to π/180.

**Example**:
```python
import brahe as bh

angle_deg = 45.0
angle_rad = angle_deg * bh.DEG2RAD  # 0.7853981633974483 radians
```

---

### RAD2DEG

::: brahe.RAD2DEG
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

**Value**: `57.29577951308232` deg/rad

Converts radians to degrees. Equivalent to 180/π.

**Example**:
```python
import brahe as bh
import math

angle_rad = math.pi / 4
angle_deg = angle_rad * bh.RAD2DEG  # 45.0 degrees
```

---

### AS2RAD

::: brahe.AS2RAD
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

**Value**: `4.84813681109536e-06` rad/arcsec

Converts arc seconds to radians. Equivalent to π/(180 × 3600).

**Example**:
```python
import brahe as bh

angle_as = 3600.0  # 1 degree in arcseconds
angle_rad = angle_as * bh.AS2RAD  # 0.017453292519943295 radians
```

---

### RAD2AS

::: brahe.RAD2AS
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

**Value**: `206264.80624709636` arcsec/rad

Converts radians to arc seconds. Equivalent to (180 × 3600)/π.

**Example**:
```python
import brahe as bh
import math

angle_rad = math.pi / 180  # 1 degree
angle_as = angle_rad * bh.RAD2AS  # 3600.0 arcseconds
```
