# Access Subcommand

Compute satellite access windows for ground locations.

## Overview

The `access` subcommand provides command-line tools for computing when satellites are visible from ground locations. It fetches TLE data from CelesTrak, propagates the orbit, and computes access windows based on elevation constraints.

## Commands

### `compute`

Compute access windows between a satellite and a ground location.

**Syntax:**
```bash
brahe access compute <norad-id> <latitude> <longitude> [options]
```

**Arguments:**
- `norad-id` - NORAD catalog ID of the satellite (e.g., 25544 for ISS)
- `latitude` - Latitude in degrees (-90 to 90)
- `longitude` - Longitude in degrees (-180 to 180)

**Options:**
- `--alt <meters>` - Altitude above WGS84 ellipsoid in meters (default: 0)
- `--start-time <time>` - Start time (ISO-8601 or epoch string, default: now)
- `--end-time <time>` - End time (ISO-8601 or epoch string)
- `--duration <days>` - Duration in days (default: 7, only used if end-time not specified)
- `--min-elevation <deg>` - Minimum elevation angle in degrees (default: 10)
- `--max-results <n>` - Maximum number of access windows to display
- `--output-format <format>` - Output format: 'rich' or 'simple' (default: rich)
- `--output-file <path>` - Export results to JSON file

## Examples

### Basic Usage

Compute next 7 days of ISS passes over New York City:
```bash
brahe access compute 25544 40.7128 -74.0060
```

### Custom Time Range

GPS satellite passes over a specific time period:
```bash
brahe access compute 32260 40.7128 -74.0060 \
    --start-time "2024-01-01T00:00:00" \
    --duration 1 \
    --min-elevation 15
```

### Simple Output Format

Use simple text output instead of rich tables:
```bash
brahe access compute 25544 40.7128 -74.0060 --output-format simple
```

### Export to JSON

Save results to a JSON file:
```bash
brahe access compute 25544 40.7128 -74.0060 --output-file passes.json
```

### High Elevation Only

Only show passes with high maximum elevation:
```bash
brahe access compute 25544 40.7128 -74.0060 \
    --min-elevation 45 \
    --max-results 5
```

### Ground Station with Altitude

Station at 1000m altitude in the mountains:
```bash
brahe access compute 25544 46.5197 6.6323 --alt 1000
```

## Output Formats

### Rich Format (Default)

Displays a formatted table with:
- Start Time (UTC)
- End Time (UTC)
- Duration (human-readable)
- Maximum Elevation
- Azimuth at window open
- Azimuth at window close

### Simple Format

One line per access window:
```
1. 2024-01-01 12:34:56 | 2024-01-01 12:45:23 | 10 minutes and 27.00 seconds | Max Elev: 45.2° | Az: 45°-315°
```

### JSON Export

Complete access window data including:
- Satellite information (name, NORAD ID)
- Location parameters
- Constraint settings
- Access window properties (times, azimuth, elevation, off-nadir, local time, look direction, ascending/descending)

## Time Range Behavior

The command handles time ranges as follows:

1. **No times specified**: Uses current time to 7 days from now
2. **Start time only**: Uses start time to start + duration (default 7 days)
3. **Start and end times**: Uses exact range specified
4. **End time only**: Error (start time required)

## Common NORAD IDs

- **25544** - International Space Station (ISS)
- **43013** - Starlink-30
- **38771** - NOAA-20
- **43226** - Sentinel-6 Michael Freilich
- **32260** - GPS BIIA-28 (SVN 49)

Use the [CelesTrak SATCAT Search](https://celestrak.org/satcat/search.php) to find other NORAD IDs.

## Python API Alternative

For more advanced access computations with custom constraints, use the Python API:

```python
import brahe as bh
import numpy as np

# Setup EOP
eop = bh.FileEOPProvider.from_default_standard(bh.EarthOrientationFileType.STANDARD, True)
bh.set_global_eop_provider(eop)

# Define epoch and orbit
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.01, np.radians(97.8), 0.0, 0.0, 0.0])
state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)

# Create propagator
prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0).with_name("Satellite")

# Define ground station
location = bh.PointLocation(np.radians(40.7128), np.radians(-74.0060), 0.0).with_name("NYC")

# Create constraint
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)

# Compute access windows
windows = bh.location_accesses(
    [location],
    [prop],
    epoch,
    epoch + 86400.0,  # 24 hours
    constraint
)

# Print results
print(f"Found {len(windows)} access windows:")
for i, window in enumerate(windows):
    print(f"  Window {i+1}:")
    print(f"    Open:  {window.window_open}")
    print(f"    Close: {window.window_close}")
    print(f"    Duration: {window.window_close - window.window_open:.1f} seconds")
    print(f"    Max Elevation: {window.properties.elevation_max:.1f}°")
```

## Use the Python API

For complete access computation functionality, see:

- **[Access Computation](../access_computation/index.md)** - Conceptual overview
- **[Locations](../access_computation/locations.md)** - Defining ground locations
- **[Constraints](../access_computation/constraints.md)** - Access constraints
- **[Computation](../access_computation/computation.md)** - Computing access windows
- **[Access API](../../library_api/access/index.md)** - Python API reference
- **[Ground Contacts Example](../../examples/ground_contacts.md)** - Complete example

## Contributing

If you're interested in implementing CLI access commands, contributions are welcome! See the [Contributing Guide](../../contributing.md) for details on how to get started.

## See Also

- [Access Module](../../library_api/access/index.md) - Python API
- [Predicting Ground Contacts](../../examples/ground_contacts.md) - Example workflow
- [Access Constraints](../access_computation/constraints.md) - Constraint types
