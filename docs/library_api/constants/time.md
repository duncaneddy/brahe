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

**Example**:
```python
import brahe as bh

# Convert between JD and MJD
jd = 2460000.0
mjd = jd - bh.MJD_ZERO  # 60000.5
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

**Example**:
```python
import brahe as bh

# Days since J2000
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
days_since_j2000 = epoch.mjd() - bh.MJD2000
```

---

### GPS_ZERO

::: brahe.GPS_ZERO
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

**Value**: `44244.0` days

Modified Julian Date of the start of GPS time (January 6, 1980 00:00:00 UTC).

**Example**:
```python
import brahe as bh

# Create GPS epoch zero
gps_start = bh.Epoch.from_mjd(bh.GPS_ZERO, bh.TimeSystem.GPS)
```

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

## Time System Conversion Example

```python
import brahe as bh

# These offsets are used internally by Epoch for time system conversions
utc_epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Convert to different time systems (handled automatically)
tai_jd = utc_epoch.jd_tai()  # Julian Date in TAI
gps_jd = utc_epoch.jd_gps()  # Julian Date in GPS
tt_jd = utc_epoch.jd_tt()    # Julian Date in TT
```
