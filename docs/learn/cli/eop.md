# EOP Commands

Earth Orientation Parameter operations and queries.

## Overview

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
brahe eop download ~/.cache/brahe/iau2000_standard.txt --product standard
```

Download C04 long-term series:
```bash
brahe eop download ~/.cache/brahe/iau2000_c04_20.txt --product c04
```

Update existing EOP file:
```bash
brahe eop download /path/to/eop_data.txt --product standard
```

**Default Location:**
The CLI automatically uses `~/.cache/brahe/iau2000_standard.txt` for frame transformations.

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
```
Output:
```
-0.123456
```

Use custom EOP file:
```bash
brahe eop get-utc-ut1 "2024-01-01T00:00:00Z" --source file --filepath /path/to/eop.txt
```

Use C04 product:
```bash
brahe eop get-utc-ut1 "2024-01-01T00:00:00Z" --product c04
```

**Note:** ΔUT1 varies irregularly due to Earth rotation variations. Typical values: -0.9s to +0.9s.

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
```
Output:
```
x_p: 0.123456 (arcseconds)
y_p: 0.234567 (arcseconds)
```

**Note:** Polar motion describes the movement of Earth's rotation axis relative to its crust. Typical magnitudes: ~0.3 arcseconds.

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
```

**Note:** CIP offset represents deviations from the IAU 2006/2000A precession-nutation model.

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
```
Output:
```
0.001234 (seconds)
```

**Note:** LOD represents the deviation of the day length from exactly 86400 seconds due to Earth rotation variations.

---

## Earth Orientation Parameters (EOP)

### What is EOP?

Earth Orientation Parameters describe the relationship between the terrestrial (ECEF) and celestial (ECI) reference frames. These parameters are measured because Earth's rotation is not perfectly uniform:

- **Polar motion** - Movement of Earth's rotation axis relative to crust
- **UT1-UTC** - Variation in Earth rotation speed
- **Nutation offsets** - Deviations from theoretical nutation model
- **LOD** - Length of day variations

### Why EOP is Needed

Accurate transformations between ECI and ECEF require EOP data:

```bash
# This uses EOP data internally
brahe transform frame ECI ECEF "2024-01-01T00:00:00Z" 6878137 0 0 0 7500 0
```

Without EOP:
- Position errors: meters to kilometers
- Velocity errors: millimeters/second to meters/second
- Unacceptable for mission operations, collision avoidance, precise orbit determination

### EOP Products

**Standard (Rapid Service):**
- Updated daily
- ~1 year of historical data
- ~1 year of predictions
- Accuracy: ~0.1 mas (polar motion), ~0.01 ms (UT1-UTC)
- Use for: Near real-time operations, recent data

**C04 (Long-term Series):**
- Complete historical record (1962-present)
- Updated monthly
- Higher accuracy for historical data
- Use for: Historical analysis, long-term studies

### Automatic EOP Handling

The CLI automatically manages EOP data:

1. **First use**: Downloads standard EOP to `~/.cache/brahe/`
2. **Subsequent uses**: Uses cached data
3. **Updates**: Run `brahe eop download` to refresh

---

## Common Workflows

### Update EOP Data

```bash
#!/bin/bash
# Update EOP data for accurate transformations

echo "Updating EOP data..."
brahe eop download ~/.cache/brahe/iau2000_standard.txt --product standard

echo "EOP data updated successfully"
```

### EOP Parameter Timeline

```bash
#!/bin/bash
# Query EOP parameters over a time range

START="2024-01-01T00:00:00Z"
END="2024-01-07T00:00:00Z"
STEP=86400  # 1 day

echo "Date | ΔUT1 (s)"
echo "-----|----------"

for epoch in $(brahe time range "$START" "$END" "$STEP"); do
  dut1=$(brahe eop get-utc-ut1 "$epoch")
  echo "$epoch | $dut1"
done
```

### EOP Data Quality Check

```bash
#!/bin/bash
# Check if EOP data is available for mission timeframe

MISSION_START="2024-06-01T00:00:00Z"
MISSION_END="2024-12-31T23:59:59Z"

echo "Checking EOP coverage for mission:"
echo "Start: $MISSION_START"
echo "End:   $MISSION_END"

# Try to get EOP for mission start
brahe eop get-utc-ut1 "$MISSION_START" 2>/dev/null
if [ $? -eq 0 ]; then
  echo "✓ EOP data available for mission start"
else
  echo "✗ EOP data NOT available for mission start"
fi

# Try for mission end
brahe eop get-utc-ut1 "$MISSION_END" 2>/dev/null
if [ $? -eq 0 ]; then
  echo "✓ EOP data available for mission end"
else
  echo "✗ EOP data NOT available - download updated EOP"
fi
```

---

## Tips

### When to Update EOP

- **Daily operations**: Update weekly
- **Mission planning**: Update before critical events
- **Historical analysis**: Use C04 product for best accuracy
- **Future predictions**: Standard product provides ~1 year predictions

### EOP File Location

Default location: `~/.cache/brahe/iau2000_standard.txt`

To use custom location:
```bash
export BRAHE_EOP_FILE=/path/to/custom/eop.txt
```

### EOP Data Sources

IERS provides EOP data from:
- IERS Rapid Service/Prediction Center (standard product)
- IERS Earth Orientation Centre (C04 product)

Data is automatically downloaded from IERS servers.

### Offline Usage

Download EOP data while online:
```bash
brahe eop download ~/.cache/brahe/iau2000_standard.txt --product standard
```

Then work offline - the CLI will use cached data.

---

## See Also

- [Earth Orientation Data](../earth_orientation_data.md) - Conceptual overview
- [Reference Frames](../frame_transformations.md) - ECI/ECEF transformations
- [Transform CLI](transform.md) - Frame transformations (use EOP)
- [Time CLI](time.md) - UT1 time system
- [EOP API](../../library_api/eop/index.md) - Python EOP provider classes
