# Access Commands

Satellite access window computation for ground stations.

## Overview

The `access` command group calculates when satellites are visible from ground locations, considering elevation constraints and other visibility criteria.

## Commands

### `compute`

Compute satellite access windows for a ground location.

**Syntax:**
```bash
brahe access compute <NORAD_ID> [OPTIONS]
```

**Arguments:**
- `NORAD_ID` - NORAD catalog ID of the satellite (integer)

**Location Options** (choose one):

Location coordinates:
- `--lat <degrees>` - Latitude in degrees (-90 to 90)
- `--lon <degrees>` - Longitude in degrees (-180 to 180)
- `--alt <meters>` - Altitude above WGS84 ellipsoid (default: 0.0)

Or ground station lookup:
- `--gs-provider <name>` - Ground station provider (e.g., 'ksat', 'atlas', 'aws')
- `--gs-name <name>` - Ground station name to lookup

**Time Range Options:**

- `--start-time <epoch>` - Start time (ISO-8601). Default: now
- `--end-time <epoch>` - End time (ISO-8601)
- `--duration <days>` - Duration in days (default: 7)

**Constraint Options:**

- `--min-elevation <degrees>` - Minimum elevation angle (default: 10.0)

**Output Options:**

- `--output-format [table|rich|simple]` - Output format (default: `table`)
- `--sort-by [contact_number|start_time|end_time|duration|max_elevation|start_azimuth|end_azimuth]` - Sort field (default: `start_time`)
- `--sort-order [ascending|descending]` - Sort order (default: `ascending`)
- `--max-results <count>` - Maximum number of windows to display
- `--output-file <path>` - Export results to JSON file

**Examples:**

ISS passes over New York City (next 7 days):
```bash
brahe access compute 25544 --lat 40.7128 --lon -74.0060
```
Output (table format):
```
Contact | Start Time           | End Time             | Duration | Max El. | Start Az | End Az
--------|----------------------|----------------------|----------|---------|----------|--------
1       | 2024-01-01 06:23:15  | 2024-01-01 06:31:42  | 507s     | 45.2°   | 185° (S) | 78° (E)
2       | 2024-01-01 08:01:33  | 2024-01-01 08:09:18  | 465s     | 38.7°   | 230° (SW)| 25° (NE)
...
```

GPS satellite passes (15° minimum elevation):
```bash
brahe access compute 32260 --lat 40.7128 --lon -74.0060 --min-elevation 15
```

Custom time range:
```bash
brahe access compute 25544 --lat 40.7128 --lon -74.0060 \
  --start-time "2024-06-01T00:00:00Z" \
  --duration 1
```

Use ground station database:
```bash
brahe access compute 25544 --gs-provider ksat --gs-name "Svalbard"
```

Simple output format:
```bash
brahe access compute 25544 --lat 40.7128 --lon -74.0060 --output-format simple
```

Export to JSON:
```bash
brahe access compute 25544 --lat 40.7128 --lon -74.0060 --output-file passes.json
```

Sort by maximum elevation (highest first):
```bash
brahe access compute 25544 --lat 40.7128 --lon -74.0060 \
  --sort-by max_elevation --sort-order descending --max-results 5
```

Sort by duration (longest passes first):
```bash
brahe access compute 25544 --lat 40.7128 --lon -74.0060 \
  --sort-by duration --sort-order descending
```

---

## Understanding Access Windows

### Satellite Visibility

A satellite is visible from a ground location when:
1. Above the horizon (elevation > 0°)
2. Meets minimum elevation constraint (default: 10°)
3. Not in Earth's shadow (for optical observations)

### Elevation Angle

Angle between the satellite and the local horizontal plane:
- `0°` - On the horizon
- `90°` - Directly overhead (zenith)
- Higher elevation = better viewing conditions
- Typical minimum: 5-10° (atmospheric effects near horizon)

### Azimuth Angle

Compass direction to the satellite:
- `0°` / `360°` - North
- `90°` - East
- `180°` - South
- `270°` - West

Output shows start and end azimuth with cardinal directions: `185° (S)`, `78° (E)`

### Access Window Components

Each access window includes:
- **Start Time** - When satellite rises above minimum elevation
- **End Time** - When satellite sets below minimum elevation
- **Duration** - Contact time in seconds
- **Max Elevation** - Highest elevation during pass
- **Start/End Azimuth** - Entry and exit directions

---

## Common Workflows

### Daily ISS Passes

```bash
#!/bin/bash
# Get today's ISS passes over a location

LAT="40.7128"
LON="-74.0060"
LOCATION_NAME="New York City"

echo "ISS passes over $LOCATION_NAME (next 24 hours):"
brahe access compute 25544 --lat $LAT --lon $LON --duration 1 --output-format rich
```

### High-Elevation Passes Only

```bash
#!/bin/bash
# Find only the best passes (elevation > 45°)

brahe access compute 25544 --lat 40.7128 --lon -74.0060 \
  --min-elevation 45 \
  --sort-by max_elevation --sort-order descending
```

### Multi-Satellite Constellation

```bash
#!/bin/bash
# Check access for multiple satellites (e.g., GPS constellation)

GPS_SATS=(32260 32384 38833 40105)  # Example GPS PRNs
LAT="40.7128"
LON="-74.0060"

for norad in "${GPS_SATS[@]}"; do
  echo "=== NORAD $norad ==="
  brahe access compute $norad --lat $LAT --lon $LON \
    --duration 1 --min-elevation 15 --output-format simple
  echo
done
```

### Export for Analysis

```bash
#!/bin/bash
# Export access windows to JSON for further processing

OUTPUT_DIR="./access_windows"
mkdir -p $OUTPUT_DIR

# ISS passes
brahe access compute 25544 --lat 40.7128 --lon -74.0060 \
  --output-file "$OUTPUT_DIR/iss_passes.json"

# HST passes
brahe access compute 20580 --lat 40.7128 --lon -74.0060 \
  --output-file "$OUTPUT_DIR/hst_passes.json"

echo "Access windows exported to $OUTPUT_DIR"
```

### Ground Station Network

```bash
#!/bin/bash
# Check satellite visibility from multiple ground stations

SATELLITE=25544  # ISS
STATIONS=("Svalbard" "Singapore" "Troll")

for station in "${STATIONS[@]}"; do
  echo "=== $station ==="
  brahe access compute $SATELLITE --gs-provider ksat --gs-name "$station" --duration 7
  echo
done
```

### Next Visible Pass

```bash
#!/bin/bash
# Find the next visible pass (max 1 result)

brahe access compute 25544 --lat 40.7128 --lon -74.0060 \
  --max-results 1 --output-format simple
```

---

## Tips

### NORAD Catalog IDs

Find NORAD IDs using the datasets command:
```bash
# Search for satellite by name
brahe datasets celestrak lookup "ISS"

# Show TLE with NORAD ID
brahe datasets celestrak show 25544
```

Common satellites:
- ISS: 25544
- Hubble Space Telescope: 20580
- Starlink satellites: 40000+

### Choosing Minimum Elevation

**Radio communications:**
- `5°` - Minimum for most radio links (atmospheric distortion)
- `10°` - Good compromise (default)
- `15°` - High-quality links

**Optical observations:**
- `15-20°` - Minimum for photography
- `30°+` - Best conditions (less atmosphere)

### Negative Longitudes

For western longitudes, use negative values or `--`:
```bash
# Method 1: Negative longitude
brahe access compute 25544 --lat 40.7128 --lon=-74.0060

# Method 2: After -- separator
brahe access compute 25544 -- --lat 40.7128 --lon -74.0060
```

### Output Formats

**table** (default):
- ASCII table with columns
- Good for terminal viewing
- Aligned columns

**rich**:
- Enhanced table with colors
- Better readability in modern terminals

**simple**:
- Plain text, one line per pass
- Easy to parse with scripts

**JSON** (via `--output-file`):
- Machine-readable
- Complete data structure
- For further processing

### Performance

Computing access windows requires:
1. Downloading TLE data from CelesTrak (cached)
2. Downloading EOP data (cached)
3. Propagating orbit over time range

Longer durations take more time:
- 1 day: ~1-2 seconds
- 7 days: ~5-10 seconds
- 30 days: ~20-30 seconds

---

## See Also

- [Ground Contacts Example](../../examples/ground_contacts.md) - Detailed access computation examples
- [Datasets CLI](datasets.md) - Find NORAD IDs and download TLEs
- [Groundstations Dataset](../datasets/groundstations.md) - Ground station database
- [Access API](../../library_api/access/index.md) - Python access computation functions
