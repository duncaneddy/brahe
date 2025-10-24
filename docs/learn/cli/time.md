# Time Commands

Time system operations, conversions, and epoch manipulation.

## Overview

The `time` command group provides:
- Conversion between time formats (MJD, JD, ISO-8601, GPS)
- Conversion between time systems (UTC, TAI, GPS, UT1, TT)
- Time arithmetic (adding/subtracting durations)
- Time range generation

## Commands

### `convert`

Convert between time formats and time systems.

**Syntax:**
```bash
brahe time convert <EPOCH> <INPUT_FORMAT> <OUTPUT_FORMAT> [OPTIONS]
```

**Arguments:**
- `EPOCH` - Time value to convert
- `INPUT_FORMAT` - Format of input: `mjd`, `jd`, `string`, `gps_date`, `gps_nanoseconds`
- `OUTPUT_FORMAT` - Desired output format (same options as input)

**Options:**
- `--input-time-system [UTC|GPS|TAI|UT1|TT]` - Time system of input
- `--output-time-system [UTC|GPS|TAI|UT1|TT]` - Time system of output

**Examples:**

Convert ISO-8601 string to Modified Julian Date:
```bash
brahe time convert "2024-01-01T00:00:00Z" string mjd --input-time-system UTC --output-time-system UTC
```
Output:
```
60310.0
```

Convert MJD to Julian Date:
```bash
brahe time convert 60310.0 mjd jd --input-time-system UTC --output-time-system UTC
```
Output:
```
2460310.5
```

Convert between time systems (UTC to TAI):
```bash
brahe time convert "2024-01-01T00:00:00Z" string string --input-time-system UTC --output-time-system TAI
```
Output:
```
2024-01-01 00:00:37.000 TAI
```

Convert GPS time to UTC:
```bash
brahe time convert "1356998418000000000" gps_nanoseconds string --output-time-system UTC
```

---

### `add`

Add a time offset to an epoch.

**Syntax:**
```bash
brahe time add <EPOCH> <SECONDS> [OPTIONS]
```

**Arguments:**
- `EPOCH` - Starting epoch (ISO-8601 string, MJD, or JD)
- `SECONDS` - Number of seconds to add (can be negative)

**Options:**
- `--output-format [mjd|jd|string|gps_date|gps_nanoseconds]` - Output format (default: `string`)
- `--output-time-system [UTC|GPS|TAI|UT1|TT]` - Output time system (default: `UTC`)

**Examples:**

Add 1 hour (3600 seconds):
```bash
brahe time add "2024-01-01T00:00:00Z" 3600
```
Output:
```
2024-01-01 01:00:00.000 UTC
```

Add 1 day (86400 seconds):
```bash
brahe time add "2024-01-01T00:00:00Z" 86400
```
Output:
```
2024-01-02 00:00:00.000 UTC
```

Subtract 30 minutes (negative seconds):
```bash
brahe time add "2024-01-01T12:00:00Z" -1800
```
Output:
```
2024-01-01 11:30:00.000 UTC
```

Output as MJD:
```bash
brahe time add "2024-01-01T00:00:00Z" 86400 --output-format mjd
```
Output:
```
60311.0
```

---

### `time-system-offset`

Calculate the offset between two time systems at a given epoch.

**Syntax:**
```bash
brahe time time-system-offset <EPOCH> <SOURCE> <TARGET>
```

**Arguments:**
- `EPOCH` - Epoch to calculate offset at (ISO-8601 string)
- `SOURCE` - Source time system: `UTC`, `GPS`, `TAI`, `UT1`, `TT`
- `TARGET` - Target time system (same options)

**Examples:**

UTC to TAI offset:
```bash
brahe time time-system-offset "2024-01-01T00:00:00Z" UTC TAI
```
Output:
```
37.0
```
(TAI is 37 seconds ahead of UTC in 2024)

GPS to UTC offset:
```bash
brahe time time-system-offset "2024-01-01T00:00:00Z" GPS UTC
```
Output:
```
-18.0
```
(GPS is 18 seconds behind UTC... wait, that's the leap second offset)

TAI to TT offset:
```bash
brahe time time-system-offset "2024-01-01T00:00:00Z" TAI TT
```
Output:
```
32.184
```
(TT is always 32.184 seconds ahead of TAI)

---

### `range`

Generate a sequence of epochs over a time range.

**Syntax:**
```bash
brahe time range <EPOCH_START> <EPOCH_END> <STEP>
```

**Arguments:**
- `EPOCH_START` - Start of time range (ISO-8601 string)
- `EPOCH_END` - End of time range (ISO-8601 string)
- `STEP` - Step size in seconds

**Examples:**

Generate epochs every 30 minutes for 1 hour:
```bash
brahe time range "2024-01-01T00:00:00Z" "2024-01-01T01:00:00Z" 1800
```
Output:
```
2024-01-01 00:00:00.000 UTC
2024-01-01 00:30:00.000 UTC
```

Generate epochs every 6 hours for 1 day:
```bash
brahe time range "2024-01-01T00:00:00Z" "2024-01-02T00:00:00Z" 21600
```
Output:
```
2024-01-01 00:00:00.000 UTC
2024-01-01 06:00:00.000 UTC
2024-01-01 12:00:00.000 UTC
2024-01-01 18:00:00.000 UTC
```

Generate epochs every minute for 5 minutes:
```bash
brahe time range "2024-01-01T12:00:00Z" "2024-01-01T12:05:00Z" 60
```
Output:
```
2024-01-01 12:00:00.000 UTC
2024-01-01 12:01:00.000 UTC
2024-01-01 12:02:00.000 UTC
2024-01-01 12:03:00.000 UTC
2024-01-01 12:04:00.000 UTC
```

---

## Time Systems

### UTC (Coordinated Universal Time)
- Civil time standard
- Includes leap seconds
- Most common for human-readable timestamps

### TAI (International Atomic Time)
- Continuous atomic time scale
- No leap seconds
- Currently 37 seconds ahead of UTC (as of 2024)

### GPS (Global Positioning System Time)
- Used by GPS satellites
- Started at 1980-01-06 00:00:00 UTC
- 19 seconds behind TAI (fixed offset)

### UT1 (Universal Time 1)
- Based on Earth's rotation
- Irregular due to Earth rotation variations
- Requires Earth Orientation Parameters (EOP)

### TT (Terrestrial Time)
- Ideal time for Earth-based observations
- Always 32.184 seconds ahead of TAI

### Offset Relationships

```
TT  = TAI + 32.184s
TAI = GPS + 19s
TAI = UTC + (leap seconds, currently 37s)
UT1 = UTC + (DUT1, from EOP data)
```

---

## Time Formats

### ISO-8601 String (`string`)
Human-readable format with timezone:
```
2024-01-01T00:00:00Z
2024-12-31T23:59:59.123Z
```

### Modified Julian Date (`mjd`)
Days since 1858-11-17 00:00:00 UTC:
```
60310.0         # 2024-01-01 00:00:00 UTC
60310.5         # 2024-01-01 12:00:00 UTC
60310.25        # 2024-01-01 06:00:00 UTC
```

### Julian Date (`jd`)
Days since -4712-01-01 12:00:00 UTC:
```
2460310.5       # 2024-01-01 00:00:00 UTC
2460311.0       # 2024-01-01 12:00:00 UTC
```

**Relationship:** `JD = MJD + 2400000.5`

### GPS Date (`gps_date`)
GPS week number and seconds:
```
2295:0.0        # GPS Week 2295, 0 seconds
```

### GPS Nanoseconds (`gps_nanoseconds`)
Nanoseconds since GPS epoch (1980-01-06 00:00:00 UTC):
```
1356998418000000000
```

---

## Common Workflows

### Mission Planning Timeline

```bash
#!/bin/bash
# Create mission timeline with 1-hour intervals
START="2024-06-01T00:00:00Z"
END="2024-06-01T24:00:00Z"
STEP=3600  # 1 hour

echo "Mission timeline:"
brahe time range "$START" "$END" "$STEP"
```

### Time System Comparison

```bash
#!/bin/bash
# Compare all time systems at a specific epoch
EPOCH="2024-01-01T00:00:00Z"

echo "Epoch: $EPOCH"
echo "UTC to TAI: $(brahe time time-system-offset "$EPOCH" UTC TAI)s"
echo "UTC to GPS: $(brahe time time-system-offset "$EPOCH" UTC GPS)s"
echo "UTC to TT:  $(brahe time time-system-offset "$EPOCH" UTC TT)s"
```

### Satellite Pass Duration

```bash
#!/bin/bash
# Calculate pass duration
START="2024-01-01T14:23:15Z"
END="2024-01-01T14:35:42Z"

# Convert to MJD
MJD_START=$(brahe time convert "$START" string mjd --input-time-system UTC --output-time-system UTC)
MJD_END=$(brahe time convert "$END" string mjd --input-time-system UTC --output-time-system UTC)

# Calculate duration (MJD is in days)
DURATION=$(echo "($MJD_END - $MJD_START) * 86400" | bc)
echo "Pass duration: ${DURATION} seconds"
```

### Propagation Timeline

```bash
#!/bin/bash
# Generate epochs for 24-hour propagation with 10-minute steps
START="2024-01-01T00:00:00Z"
DURATION_HOURS=24
STEP_SECONDS=600  # 10 minutes

# Calculate end time
END=$(brahe time add "$START" $((DURATION_HOURS * 3600)) --output-format string)

# Generate timeline
brahe time range "$START" "$END" "$STEP_SECONDS" > propagation_epochs.txt
```

---

## Tips

### Leap Seconds

UTC includes leap seconds, which are added irregularly:
- As of 2024-01-01: 37 leap seconds have been added since 1972
- TAI - UTC = 37 seconds
- GPS is 19 seconds behind TAI (fixed)

Use TAI or GPS for continuous time calculations without discontinuities.

### High-Precision Timestamps

ISO-8601 strings support nanosecond precision:
```bash
brahe time convert "2024-01-01T00:00:00.123456789Z" string mjd
```

### Negative Time Offsets

Use negative seconds to subtract time:
```bash
# Go back 1 week
brahe time add "2024-01-08T00:00:00Z" -604800
```

### Batch Processing

Use shell loops for batch time conversions:
```bash
# Convert multiple epochs
for epoch in "2024-01-01T00:00:00Z" "2024-06-01T00:00:00Z" "2024-12-31T00:00:00Z"; do
  echo "$epoch -> $(brahe time convert "$epoch" string mjd --input-time-system UTC --output-time-system UTC)"
done
```

---

## See Also

- [Time Systems](../time.md) - Conceptual overview
- [Earth Orientation Data](../earth_orientation_data.md) - EOP and UT1
- [Epoch API](../../library_api/time/epoch.md) - Python Epoch class
- [EOP CLI](eop.md) - Earth Orientation Parameters
- [Transform CLI](transform.md) - Coordinate transformations (require epochs)
