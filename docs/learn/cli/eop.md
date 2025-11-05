# EOP Commands

The `eop` command group provides access to Earth Orientation Parameters (EOP) data from IERS (International Earth Rotation and Reference Systems Service). EOP data is required for accurate transformations between ECI and ECEF reference frames.

## Commands

### `download`

Download EOP data from IERS and save to file.

**Syntax:**
```bash
brahe eop download <FILEPATH> --product <PRODUCT>
```

**Arguments:**
- `FILEPATH` - Output file path for EOP data

**Options:**
- `--product [standard|c04]` - Data product type (required)
  - `standard` - Standard rapid EOP data (daily updates, ~1 year of predictions)
  - `c04` - EOP 14 C04 long-term series (high accuracy, historical)

**Examples:**

Download standard EOP data:
```bash
brahe eop download ~/.cache/brahe/eop/iau2000_standard.txt --product standard
```

Download C04 long-term series:
```bash
brahe eop download ~/.cache/brahe/eop/iau2000_c04_20.txt --product c04
```

Update local EOP file:
```bash
brahe eop download /path/to/eop_data.txt --product standard
```

!!! tip
    It's usually not necessary to manually download EOP data. If you are using a caching file provider, the package data will automatically download and cache EOP data as needed.

---

### `get-utc-ut1`

Get the UTC-UT1 offset (ΔUT1) at a specific epoch.

**Syntax:**
```bash
brahe eop get-utc-ut1 <EPOCH> [OPTIONS]
```

**Arguments:**
- `EPOCH` - Epoch to query (ISO-8601 format)

**Options:**
- `--product [standard|c04]` - EOP data product (default: `standard`)
- `--source [default|file]` - EOP data source (default: `default`)
- `--filepath <path>` - Custom EOP file path (if `--source file`)

**Examples:**

Get ΔUT1 for a specific date:
```bash
brahe eop get-utc-ut1 "2024-01-01T00:00:00Z"
# 0.0087837
```

Use custom EOP file:
```bash
brahe eop get-utc-ut1 "2024-01-01T00:00:00Z" --source file --filepath /path/to/eop.txt
```

Use C04 product:
```bash
brahe eop get-utc-ut1 "2024-01-01T00:00:00Z" --product c04
# 0.0087572
```

---

### `get-polar-motion`

Get polar motion parameters (x_p, y_p) at a specific epoch.

**Syntax:**
```bash
brahe eop get-polar-motion <EPOCH> [OPTIONS]
```

**Arguments:**
- `EPOCH` - Epoch to query (ISO-8601 format)

**Options:**
- `--product [standard|c04]` - EOP data product (default: `standard`)
- `--source [default|file]` - EOP data source (default: `default`)
- `--filepath <path>` - Custom EOP file path (if `--source file`)

**Examples:**

Get polar motion parameters:
```bash
brahe eop get-polar-motion "2024-01-01T00:00:00Z"
# 6.63768107080688e-07, 9.802447818353709e-07
```

---

### `get-cip-offset`

Get Celestial Intermediate Pole (CIP) offset (dX, dY) at a specific epoch.

**Syntax:**
```bash
brahe eop get-cip-offset <EPOCH> [OPTIONS]
```

**Arguments:**
- `EPOCH` - Epoch to query (ISO-8601 format)

**Options:**
- `--product [standard|c04]` - EOP data product (default: `standard`)
- `--source [default|file]` - EOP data source (default: `default`)
- `--filepath <path>` - Custom EOP file path (if `--source file`)

**Examples:**

Get CIP offset:
```bash
brahe eop get-cip-offset "2024-01-01T00:00:00Z"
# 1.4302003592731312e-09, -4.6057299705405924e-10
```

---

### `get-lod`

Get Length of Day (LOD) variation at a specific epoch.

**Syntax:**
```bash
brahe eop get-lod <EPOCH> [OPTIONS]
```

**Arguments:**
- `EPOCH` - Epoch to query (ISO-8601 format)

**Options:**
- `--product [standard|c04]` - EOP data product (default: `standard`)
- `--source [default|file]` - EOP data source (default: `default`)
- `--filepath <path>` - Custom EOP file path (if `--source file`)

**Examples:**

Get LOD variation:
```bash
brahe eop get-lod "2024-01-01T00:00:00Z"
# 0.00023750000000000003
```

---

---

## See Also

- [Earth Orientation Data](../eop/what_is_eop_data.md) - Conceptual overview
- [Reference Frames](../frame_transformations.md) - ECI/ECEF transformations
- [Transform CLI](transform.md) - Frame transformations (use EOP)
- [Time CLI](time.md) - UT1 time system
- [EOP API](../../library_api/eop/index.md) - Python EOP provider classes
