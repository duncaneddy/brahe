# EOP Subcommand

Earth Orientation Parameter (EOP) data management commands for downloading and querying EOP data.

## Overview

The `eop` subcommand provides tools for managing Earth Orientation Parameters, which are essential for accurate coordinate transformations between celestial and terrestrial reference frames.

## Commands

### `download`

Download EOP data files from IERS.

**Syntax:**
```bash
brahe eop download <filepath> --product <standard|c04>
```

**Arguments:**
- `filepath` - Path where the EOP data file will be saved

**Options:**
- `--product` - Type of EOP product to download:
  - `standard` - Standard rapid data (default, updated daily)
  - `c04` - C04 finals data (high precision, updated less frequently)

**Examples:**

```bash
# Download standard EOP data
brahe eop download eop_standard.txt --product standard

# Download C04 EOP data
brahe eop download eop_c04.txt --product c04
```

---

### `get-utc-ut1`

Get the UT1-UTC offset for a specific epoch.

**Syntax:**
```bash
brahe eop get-utc-ut1 <epoch> [options]
```

**Arguments:**
- `epoch` - Epoch-like string (ISO 8601, MJD, or JD)

**Options:**
- `--product <standard|c04>` - EOP product type (default: `standard`)
- `--source <default|file>` - Data source (default: `default`)
- `--filepath <path>` - Path to EOP file (only used if `--source file`)

**Examples:**

```bash
# Get UT1-UTC for a specific date
brahe eop get-utc-ut1 "2024-01-01T00:00:00"

# Use C04 product
brahe eop get-utc-ut1 "2024-01-01T00:00:00" --product c04

# Use custom EOP file
brahe eop get-utc-ut1 "2024-01-01T00:00:00" --source file --filepath my_eop.txt
```

**Output:**
```
-0.0234567  # UT1-UTC offset in seconds
```

---

### `get-polar-motion`

Get polar motion parameters (x, y) for a specific epoch.

**Syntax:**
```bash
brahe eop get-polar-motion <epoch> [options]
```

**Arguments:**
- `epoch` - Epoch-like string

**Options:**
- Same as `get-utc-ut1`

**Examples:**

```bash
# Get polar motion parameters
brahe eop get-polar-motion "2024-01-01T00:00:00"
```

**Output:**
```
0.123456, 0.234567  # pm_x, pm_y in arcseconds
```

---

### `get-cip-offset`

Get Celestial Intermediate Pole (CIP) offsets (dX, dY) for a specific epoch.

**Syntax:**
```bash
brahe eop get-cip-offset <epoch> [options]
```

**Arguments:**
- `epoch` - Epoch-like string

**Options:**
- Same as `get-utc-ut1`

**Examples:**

```bash
# Get CIP offsets
brahe eop get-cip-offset "2024-01-01T00:00:00"
```

**Output:**
```
0.000123, 0.000234  # dX, dY in arcseconds
```

---

### `get-lod`

Get Length of Day (LOD) excess for a specific epoch.

**Syntax:**
```bash
brahe eop get-lod <epoch> [options]
```

**Arguments:**
- `epoch` - Epoch-like string

**Options:**
- Same as `get-utc-ut1`

**Examples:**

```bash
# Get LOD
brahe eop get-lod "2024-01-01T00:00:00"
```

**Output:**
```
0.0012345  # LOD in milliseconds
```

## Common Workflows

### Initial Setup

Download EOP data for offline use:

```bash
# Create data directory
mkdir -p ~/.brahe/eop

# Download standard EOP data
brahe eop download ~/.brahe/eop/standard.txt --product standard

# Download C04 EOP data for higher precision
brahe eop download ~/.brahe/eop/c04.txt --product c04
```

### Querying EOP Data

```bash
# Check current UT1-UTC offset
brahe eop get-utc-ut1 "2024-01-01T00:00:00"

# Get all parameters for a specific epoch
epoch="2024-01-01T00:00:00"
ut1_utc=$(brahe eop get-utc-ut1 "$epoch")
pm=$(brahe eop get-polar-motion "$epoch")
cip=$(brahe eop get-cip-offset "$epoch")
lod=$(brahe eop get-lod "$epoch")

echo "Epoch: $epoch"
echo "UT1-UTC: $ut1_utc s"
echo "Polar Motion: $pm arcsec"
echo "CIP Offset: $cip arcsec"
echo "LOD: $lod ms"
```

## Notes

- **EOP Data Updates**: Standard EOP data is updated daily. Download fresh data regularly for current operations.
- **C04 vs Standard**: C04 provides higher precision but is updated less frequently. Use standard for near-real-time applications.
- **Data Range**: EOP data is only available for specific date ranges. Commands will error if the requested epoch is out of range.
- **Default Source**: If `--source` is not specified, the command downloads and caches data automatically.

## See Also

- [Earth Orientation Data](../eop/index.md) - Conceptual overview
- [Managing EOP Data](../eop/managing_eop_data.md) - Detailed guide
- [FileEOPProvider API](../../library_api/eop/file_provider.md) - Python API
