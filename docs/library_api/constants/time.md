# Time Constants

Constants related to time systems, epochs, and time conversions.

## Julian Date References

### MJD_ZERO

::: brahe.MJD_ZERO
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

**Value**: `2400000.5` days

Offset of Modified Julian Date (MJD) with respect to Julian Date (JD). For any time t:
```
MJD_ZERO = JD - MJD
```

---

### MJD2000

::: brahe.MJD2000
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

**Value**: `51544.5` days

Modified Julian Date of January 1, 2000 12:00:00 (J2000.0 epoch). Value is independent of time system.

---

### GPS_ZERO

::: brahe.GPS_ZERO
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

**Value**: `44244.0` days

Modified Julian Date of the start of GPS time (January 6, 1980 00:00:00 UTC).

---

## Time System Offsets

All offset values are in seconds.

### GPS ↔ TAI

#### GPS_TAI

::: brahe.GPS_TAI
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `-19.0` seconds

Offset of GPS time with respect to TAI: `GPS = TAI + GPS_TAI`

---

#### TAI_GPS

::: brahe.TAI_GPS
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `19.0` seconds

Offset of TAI time with respect to GPS: `TAI = GPS + TAI_GPS`

---

### TT ↔ TAI

#### TT_TAI

::: brahe.TT_TAI
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `32.184` seconds

Offset of Terrestrial Time with respect to TAI: `TT = TAI + TT_TAI`

---

#### TAI_TT

::: brahe.TAI_TT
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `-32.184` seconds

Offset of TAI with respect to Terrestrial Time: `TAI = TT + TAI_TT`

---

### GPS ↔ TT

#### GPS_TT

::: brahe.GPS_TT
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `13.184` seconds

Offset of GPS time with respect to TT: `GPS = TT + GPS_TT`

Computed as: `GPS_TAI + TAI_TT`

---

#### TT_GPS

::: brahe.TT_GPS
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 5

**Value**: `-13.184` seconds

Offset of TT with respect to GPS time: `TT = GPS + TT_GPS`

---
