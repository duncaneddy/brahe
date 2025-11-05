# EOP Functions

Global EOP management and query functions.

**Module**: `brahe.eop`

## Setting Global EOP Provider

### initialize_eop

::: brahe.initialize_eop
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

**Recommended**: Initialize the global EOP provider with sensible defaults. This is the easiest way to get started with EOP data for most applications.

---

### set_global_eop_provider

::: brahe.set_global_eop_provider
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

Set the global EOP provider using any supported provider type (StaticEOPProvider, FileEOPProvider, or CachingEOPProvider).

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

---

### download_c04_eop_file

::: brahe.download_c04_eop_file
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 4

---

## Usage Examples

### Quick Start (Recommended)

```python
import brahe as bh

# Initialize EOP with recommended defaults - easiest way to get started!
bh.initialize_eop()

# Query EOP for specific epoch
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
mjd = epoch.mjd()

ut1_utc, pm_x, pm_y, dx, dy, lod = bh.get_global_eop(mjd)
print(f"EOP for MJD {mjd}:")
print(f"  UT1-UTC: {ut1_utc:.6f} s")
print(f"  Polar Motion: ({pm_x*1e6:.3f}, {pm_y*1e6:.3f}) μrad")
print(f"  dX, dY: ({dx*1e6:.3f}, {dy*1e6:.3f}) μrad")
print(f"  LOD: {lod*1e3:.6f} ms")
```

### Custom Provider Setup

```python
import brahe as bh

# Download and set up file-based EOP with custom settings
eop_file = bh.download_standard_eop_file("./data")
provider = bh.FileEOPProvider.from_standard_file(
    eop_file,
    interpolate=True,
    extrapolate="Hold"
)
bh.set_global_eop_provider(provider)

# Check provider status
print(f"EOP Type: {bh.get_global_eop_type()}")
print(f"Data points: {bh.get_global_eop_len()}")
print(f"Date range: MJD {bh.get_global_eop_mjd_min():.1f} to {bh.get_global_eop_mjd_max():.1f}")
print(f"Interpolation: {bh.get_global_eop_interpolation()}")
print(f"Extrapolation: {bh.get_global_eop_extrapolation()}")
```

---

## See Also

- [FileEOPProvider](file_provider.md)
- [StaticEOPProvider](static_provider.md)
- [Frames](../frames.md) - Frame transformations that use EOP
