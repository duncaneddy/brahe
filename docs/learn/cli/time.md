# Time Subcommand

Time system conversions and epoch operations.

## Overview

The `time` subcommand provides utilities for converting between different time systems and epoch formats, performing time arithmetic, and generating time ranges.

## Commands

### `convert`

Convert epochs between different formats and time systems.

**Syntax:**
```bash
brahe time convert <epoch> <input-format> <output-format> [options]
```

**Arguments:**
- `epoch` - The epoch value to convert
- `input-format` - Format of the input epoch:
  - `string` - ISO 8601 string (e.g., "2024-01-01T12:00:00")
  - `mjd` - Modified Julian Date
  - `jd` - Julian Date
  - `gps_date` - GPS date (days since GPS epoch)
  - `gps_nanoseconds` - GPS nanoseconds
- `output-format` - Desired output format (same options as input-format)

**Options:**
- `--input-time-system <UTC|GPS|TAI|UT1|TT>` - Time system of input (default: TAI)
- `--output-time-system <UTC|GPS|TAI|UT1|TT>` - Time system of output (default: TAI)

**Examples:**

```bash
# Convert ISO string to MJD in UTC
brahe time convert "2024-01-01T00:00:00" string mjd --output-time-system UTC

# Convert MJD (TAI) to Julian Date (UTC)
brahe time convert 60310.5 mjd jd --input-time-system TAI --output-time-system UTC

# Convert between time systems (same format)
brahe time convert 60310.5 mjd mjd --input-time-system UTC --output-time-system GPS

# Convert GPS nanoseconds to ISO string
brahe time convert 1325376000000000000 gps_nanoseconds string
```

**Output:**
```
60310.0  # Example MJD output
```

---

### `add`

Add seconds to an epoch and output the result.

**Syntax:**
```bash
brahe time add <epoch> <seconds> [options]
```

**Arguments:**
- `epoch` - Epoch-like string (ISO 8601, MJD, JD, etc.)
- `seconds` - Number of seconds to add (can be negative)

**Options:**
- `--output-format <format>` - Output format (default: `string`)
- `--output-time-system <system>` - Output time system (default: `UTC`)

**Examples:**

```bash
# Add 1 hour (3600 seconds) to an epoch
brahe time add "2024-01-01T00:00:00" 3600

# Add 1 day in seconds, output as MJD
brahe time add "2024-01-01T00:00:00" 86400 --output-format mjd

# Subtract time (negative seconds)
brahe time add "2024-01-01T12:00:00" -43200 --output-format string
```

**Output:**
```
2024-01-01T01:00:00  # Example: original + 1 hour
```

---

### `time-system-offset`

Calculate the time offset between two time systems at a specific epoch.

**Syntax:**
```bash
brahe time time-system-offset <epoch> <source> <target>
```

**Arguments:**
- `epoch` - Epoch-like string
- `source` - Source time system (UTC, GPS, TAI, UT1, TT)
- `target` - Target time system

**Examples:**

```bash
# Get TAI - UTC offset
brahe time time-system-offset "2024-01-01T00:00:00" UTC TAI

# Get GPS - UTC offset
brahe time time-system-offset "2024-01-01T00:00:00" UTC GPS

# Get TT - TAI offset (constant 32.184 seconds)
brahe time time-system-offset "2024-01-01T00:00:00" TAI TT
```

**Output:**
```
37.0  # Offset in seconds
```

---

### `range`

Generate a sequence of epochs over a time range.

**Syntax:**
```bash
brahe time range <epoch-start> <epoch-end> <step>
```

**Arguments:**
- `epoch-start` - Starting epoch (epoch-like string)
- `epoch-end` - Ending epoch (epoch-like string)
- `step` - Step size in seconds

**Examples:**

```bash
# Generate hourly epochs for one day
brahe time range "2024-01-01T00:00:00" "2024-01-02T00:00:00" 3600

# Generate 10-minute intervals
brahe time range "2024-01-01T00:00:00" "2024-01-01T06:00:00" 600
```

**Output:**
```
2024-01-01T00:00:00
2024-01-01T01:00:00
2024-01-01T02:00:00
...
```

## Time Systems

Brahe supports the following time systems:

- **UTC** - Coordinated Universal Time (includes leap seconds)
- **TAI** - International Atomic Time (continuous, no leap seconds)
- **GPS** - GPS Time (continuous since GPS epoch: 1980-01-06)
- **TT** - Terrestrial Time (uniform time scale for solar system dynamics)
- **UT1** - Universal Time 1 (based on Earth's rotation, requires EOP data)

### Time System Relationships

```
TAI = UTC + (leap seconds)
GPS = TAI - 19 seconds
TT  = TAI + 32.184 seconds
UT1 = UTC + UT1-UTC (from EOP data)
```

## Epoch Formats

### ISO 8601 String
```bash
"2024-01-01T12:00:00"
"2024-12-31T23:59:60"  # Leap second
```

### Modified Julian Date (MJD)
```bash
60310.5  # Days since 1858-11-17T00:00:00
```

### Julian Date (JD)
```bash
2460310.5  # Days since -4713-11-24T12:00:00
```

### GPS Formats
```bash
# GPS Date (days since GPS epoch)
9862

# GPS Nanoseconds
1325376000000000000
```

## Common Workflows

### Time Zone Conversions

```bash
# Convert UTC to different time systems
utc_epoch="2024-01-01T00:00:00"
tai=$(brahe time convert "$utc_epoch" string mjd --input-time-system UTC --output-time-system TAI)
gps=$(brahe time convert "$utc_epoch" string mjd --input-time-system UTC --output-time-system GPS)
tt=$(brahe time convert "$utc_epoch" string mjd --input-time-system UTC --output-time-system TT)

echo "UTC: $utc_epoch"
echo "TAI: $tai MJD"
echo "GPS: $gps MJD"
echo "TT: $tt MJD"
```

### Mission Planning

```bash
# Generate orbit epoch sequence
start="2024-01-01T00:00:00"
end="2024-01-01T01:30:00"
period=5400  # 90-minute orbit

echo "Orbit Epochs:"
brahe time range "$start" "$end" "$period"
```

### Time Arithmetic

```bash
# Calculate epoch 1 week from now
current="2024-01-01T00:00:00"
week_seconds=$((7 * 24 * 3600))
future=$(brahe time add "$current" $week_seconds)

echo "Current: $current"
echo "Future:  $future"
```

## Notes

- **Leap Seconds**: UTC includes leap seconds; TAI and GPS do not. The offset changes when leap seconds are added.
- **UT1 Requires EOP**: Conversions to/from UT1 require Earth Orientation Parameters.
- **Precision**: All time calculations maintain nanosecond precision internally.
- **Epoch Parsing**: The CLI automatically detects and parses various epoch formats when used as arguments.

## See Also

- [Time Systems](../time/time_systems.md) - Detailed explanation of time systems
- [Epoch Class](../../library_api/time/epoch.md) - Python API
- [Time Conversions](../time/time_conversions.md) - Conversion formulas
