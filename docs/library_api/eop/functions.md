# EOP Functions

Global EOP management and query functions.

**Module**: `brahe.eop`

## Setting Global EOP Provider

### set_global_eop_provider_from_file_provider

::: brahe.set_global_eop_provider_from_file_provider
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

Set a file-based EOP provider as the global provider.

**Example**:
```python
import brahe as bh

provider = bh.FileEOPProvider.from_default_standard()
bh.set_global_eop_provider_from_file_provider(provider)
```

---

### set_global_eop_provider_from_static_provider

::: brahe.set_global_eop_provider_from_static_provider
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

Set a static EOP provider as the global provider.

**Example**:
```python
import brahe as bh

provider = bh.StaticEOPProvider.from_zero()
bh.set_global_eop_provider_from_static_provider(provider)
```

---

## Querying Global EOP Data

### get_global_eop

::: brahe.get_global_eop
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

Get all EOP values for a specific Modified Julian Date.

**Returns**: Tuple of (ut1_utc, pm_x, pm_y, dx, dy, lod)

---

### get_global_ut1_utc

::: brahe.get_global_ut1_utc
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

Get UT1-UTC offset in seconds.

---

### get_global_pm

::: brahe.get_global_pm
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

Get polar motion (x, y) in radians.

---

### get_global_dxdy

::: brahe.get_global_dxdy
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

Get celestial pole offsets (dx, dy) in radians.

---

### get_global_lod

::: brahe.get_global_lod
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

Get length of day offset in seconds.

---

## EOP Metadata

### get_global_eop_type

::: brahe.get_global_eop_type
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

Get the type of global EOP provider ("file" or "static").

---

### get_global_eop_initialization

::: brahe.get_global_eop_initialization
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

Check if global EOP provider has been initialized.

---

### get_global_eop_interpolation

::: brahe.get_global_eop_interpolation
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

Check if interpolation is enabled.

---

### get_global_eop_extrapolation

::: brahe.get_global_eop_extrapolation
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

Get extrapolation method ("Hold", "Zero", or "Error").

---

### get_global_eop_len

::: brahe.get_global_eop_len
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

Get number of EOP data points in provider.

---

### get_global_eop_mjd_min

::: brahe.get_global_eop_mjd_min
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

Get minimum (earliest) MJD in EOP data.

---

### get_global_eop_mjd_max

::: brahe.get_global_eop_mjd_max
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

Get maximum (latest) MJD in EOP data.

---

### get_global_eop_mjd_last_lod

::: brahe.get_global_eop_mjd_last_lod
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

Get MJD of last LOD (Length of Day) data point.

---

### get_global_eop_mjd_last_dxdy

::: brahe.get_global_eop_mjd_last_dxdy
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

Get MJD of last dX/dY (celestial pole offset) data point.

---

## Downloading EOP Files

### download_standard_eop_file

::: brahe.download_standard_eop_file
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

Download latest Standard format EOP file from IERS.

**Example**:
```python
import brahe as bh

# Download to specific directory
filepath = bh.download_standard_eop_file("./eop_data")

# Use downloaded file
provider = bh.FileEOPProvider.from_standard_file(filepath)
```

---

### download_c04_eop_file

::: brahe.download_c04_eop_file
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

Download latest C04 format EOP file from IERS.

**Example**:
```python
import brahe as bh

# Download to specific directory
filepath = bh.download_c04_eop_file("./eop_data")

# Use downloaded file
provider = bh.FileEOPProvider.from_c04_file(filepath)
```

---

## Complete Example

```python
import brahe as bh

# Download and set up file-based EOP
eop_file = bh.download_standard_eop_file("./data")
provider = bh.FileEOPProvider.from_standard_file(
    eop_file,
    interpolate=True,
    extrapolate="Hold"
)
bh.set_global_eop_provider_from_file_provider(provider)

# Check provider status
print(f"EOP Type: {bh.get_global_eop_type()}")
print(f"Data points: {bh.get_global_eop_len()}")
print(f"Date range: MJD {bh.get_global_eop_mjd_min():.1f} to {bh.get_global_eop_mjd_max():.1f}")
print(f"Interpolation: {bh.get_global_eop_interpolation()}")
print(f"Extrapolation: {bh.get_global_eop_extrapolation()}")

# Query EOP for specific epoch
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
mjd = epoch.mjd()

ut1_utc, pm_x, pm_y, dx, dy, lod = bh.get_global_eop(mjd)
print(f"\nEOP for MJD {mjd}:")
print(f"  UT1-UTC: {ut1_utc:.6f} s")
print(f"  Polar Motion: ({pm_x*1e6:.3f}, {pm_y*1e6:.3f}) μrad")
print(f"  dX, dY: ({dx*1e6:.3f}, {dy*1e6:.3f}) μrad")
print(f"  LOD: {lod*1e3:.6f} ms")
```

## See Also

- [FileEOPProvider](file_provider.md)
- [StaticEOPProvider](static_provider.md)
- [Frames](../frames.md) - Frame transformations that use EOP
