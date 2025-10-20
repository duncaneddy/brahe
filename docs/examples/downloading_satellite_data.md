# How to Download Satellite Ephemeris Data

This guide shows how to download, process, and use satellite ephemeris data from CelesTrak using Brahe's datasets module.

## Prerequisites

```python
import brahe as bh
import numpy as np
from pathlib import Path
```

## Quick Start: Download and Use

The simplest workflow downloads ephemeris data and creates propagators in one step:

```python
# Get GNSS satellites as ready-to-use propagators
propagators = bh.datasets.celestrak.get_ephemeris_as_propagators(
    "gnss",
    step_size=60.0  # Internal propagation step in seconds
)

print(f"Loaded {len(propagators)} GNSS satellites")

# Propagate all satellites to a specific time
epoch = bh.Epoch.from_datetime(2024, 6, 15, 12, 0, 0, tsys="UTC")
for i, prop in enumerate(propagators[:5]):  # First 5 satellites
    state = prop.state_eci(epoch)
    position = state[:3]  # Extract position [x, y, z]
    print(f"Satellite {i+1}: {position / 1000:.1f} km")
```

## Download Raw Ephemeris Data

Get ephemeris as tuples for custom processing:

```python
# Download GNSS constellation ephemeris
ephemeris = bh.datasets.celestrak.get_ephemeris("gnss")

# ephemeris is a list of (name, line1, line2) tuples
print(f"Downloaded {len(ephemeris)} satellites")

# Process first satellite
name, line1, line2 = ephemeris[0]
print(f"\nSatellite: {name}")
print(f"Line 1: {line1}")
print(f"Line 2: {line2}")

# Create SGP propagator
prop = bh.SGPPropagator.from_tle(line1, line2)

# Extract epoch
epoch = bh.epoch_from_tle(line1)
print(f"Epoch: {epoch}")

# Convert to Keplerian elements
epoch_tle, keplerian = bh.keplerian_elements_from_tle(line1, line2)
a, e, i, raan, argp, M = keplerian
print(f"Semi-major axis: {a/1000:.1f} km")
print(f"Eccentricity: {e:.6f}")
print(f"Inclination: {i:.2f} deg")  # Already in degrees
```

## Save Ephemeris to Files

### Save as Plain Text

Use standard TLE format for compatibility with other tools:

```python
# Download and save GPS satellites as text
bh.datasets.celestrak.download_ephemeris(
    "gps-ops",
    "gps_satellites.txt",
    content_format="3le",  # Include satellite names
    file_format="txt"      # Plain text format
)

print("Saved GPS satellites to gps_satellites.txt")
```

File format:
```
GPS BIIA-10 (PRN 32)
1 20959U 90103A   24167.50000000  .00000012  00000+0  00000+0 0  9998
2 20959  54.9876 123.4567 0123456 234.5678 123.4567  2.00561234567890

GPS BIIR-2  (PRN 13)
1 24876U 97035A   24167.50000000  .00000011  00000+0  00000+0 0  9997
2 24876  55.1234 234.5678 0098765 345.6789 012.3456  2.00557890123456
```

### Save as CSV

For spreadsheet analysis or database import:

```python
# Download Starlink satellites as CSV
bh.datasets.celestrak.download_ephemeris(
    "starlink",
    "starlink_satellites.csv",
    content_format="3le",
    file_format="csv"
)

print("Saved Starlink satellites to CSV")

# Load with pandas for analysis
import pandas as pd
df = pd.read_csv("starlink_satellites.csv")
print(f"Loaded {len(df)} Starlink satellites")
print(df.head())
```

CSV format:
```csv
Name,Line1,Line2
STARLINK-1007,1 44713U 19074A   24167.50000000  .00001234  00000-0  12345-4 0  9991,2 44713  53.0543 123.4567 0001234  90.1234 269.9876 15.06391234567890
STARLINK-1008,1 44714U 19074B   24167.50000000  .00001235  00000-0  12346-4 0  9992,2 44714  53.0544 123.4568 0001235  90.1235 269.9877 15.06391345678901
```

### Save as JSON

For web applications or structured processing:

```python
# Download space stations as JSON
bh.datasets.celestrak.download_ephemeris(
    "stations",
    "space_stations.json",
    content_format="3le",
    file_format="json"
)

print("Saved space stations to JSON")

# Load and parse
import json
with open("space_stations.json", "r") as f:
    stations = json.load(f)

for station in stations:
    print(f"{station['name']}: NORAD ID {station['line1'].split()[1]}")
```

JSON format:
```json
[
  {
    "name": "ISS (ZARYA)",
    "line1": "1 25544U 98067A   24167.50000000  .00002182  00000-0  41420-4 0  9990",
    "line2": "2 25544  51.6461 339.8014 0002571  34.5857 120.4689 15.48919393265104"
  },
  {
    "name": "TIANGONG",
    "line1": "1 48274U 21035A   24167.50000000  .00001234  00000-0  23456-4 0  9991",
    "line2": "2 48274  41.4700 123.4560 0001234  56.7890  12.3456 15.59876543012345"
  }
]
```

## Work with Multiple Satellite Groups

Combine ephemeris from several groups:

```python
# Download all navigation constellations
navigation_groups = ["gps-ops", "glonass-ops", "galileo", "beidou"]
all_ephemeris = []

for group in navigation_groups:
    ephemeris = bh.datasets.celestrak.get_ephemeris(group)
    all_ephemeris.extend(ephemeris)
    print(f"{group}: {len(ephemeris)} satellites")

print(f"\nTotal navigation satellites: {len(all_ephemeris)}")

# Save combined data
import json
output = [
    {"name": name, "line1": line1, "line2": line2}
    for name, line1, line2 in all_ephemeris
]
with open("all_navigation_satellites.json", "w") as f:
    json.dump(output, f, indent=2)
```

## Filter Satellites by Criteria

Select specific satellites from ephemeris data:

```python
# Get all active satellites
all_sats = bh.datasets.celestrak.get_ephemeris("active")

# Filter for LEO satellites (period < 128 minutes = ~8000 km altitude)
leo_sats = []
for name, line1, line2 in all_sats:
    # Extract mean-Keplerian elements at epoch
    epoch_tle, elements = bh.keplerian_elements_from_tle(line1, line2)

    # Calculate orbital period
    a = elements[0]  # Semi-major axis in meters
    period = bh.orbital_period(a)  # Period in seconds

    if period < 128 * 60:  # 128 minutes in seconds
        leo_sats.append((name, line1, line2))

print(f"Found {len(leo_sats)} LEO satellites out of {len(all_sats)} total")
```

## Compute Satellite Visibility

Use downloaded ephemeris to predict visibility from a ground station:

```python
# Download satellites of interest
propagators = bh.datasets.celestrak.get_ephemeris_as_propagators(
    "weather",
    step_size=60.0
)

# Ground station location (geodetic coordinates)
station_lat = 40.7128   # New York City latitude
station_lon = -74.0060  # New York City longitude
station_alt = 10.0      # Altitude in meters (approximate)

# Convert to ECEF
station_ecef = bh.position_geodetic_to_ecef(
    np.array([station_lat, station_lon, station_alt]),
    bh.DEGREES
)

# Check visibility at specific time
epoch = bh.Epoch.from_datetime(2024, 6, 15, 18, 0, 0, tsys="UTC")

visible_satellites = []
for i, prop in enumerate(propagators):
    # Get satellite position in ECI
    state_eci = prop.propagate(epoch)
    pos_eci = state_eci[:3]

    # Convert to ECEF
    pos_ecef = bh.position_eci_to_ecef(pos_eci, epoch)

    # Compute relative position
    relative_ecef = pos_ecef - station_ecef

    # Convert to topocentric (ENZ)
    relative_enz = bh.relative_position_ecef_to_enz(
        station_lat,
        station_lon,
        relative_ecef
    )

    # Convert to azimuth/elevation
    az, el, _ = bh.position_enz_to_azel(relative_enz)

    # Check if above horizon (elevation > 0)
    if el > 0:
        visible_satellites.append({
            "index": i,
            "azimuth": np.rad2deg(az),
            "elevation": np.rad2deg(el)
        })

print(f"Visible satellites: {len(visible_satellites)}")
for sat in visible_satellites[:5]:
    print(f"Satellite {sat['index']}: f"Az={sat['azimuth']:.1f}°, El={sat['elevation']:.1f}°")
```

## Handle Errors Gracefully

Implement robust error handling and retries:

```python
import time

def download_with_retry(group, filepath, max_retries=3):
    """Download ephemeris with exponential backoff"""
    for attempt in range(max_retries):
        try:
            bh.datasets.celestrak.download_ephemeris(
                group,
                filepath,
                content_format="3le",
                file_format="json"
            )
            print(f"Successfully downloaded {group}")
            return True

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"Attempt {attempt + 1} failed: {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"All {max_retries} attempts failed for {group}")
                return False

# Use with error handling
success = download_with_retry("gnss", "gnss_satellites.json")
if not success:
    print("Failed to download")
    # Load from previous download or use fallback
```

## See Also

- [Datasets Overview](../learn/datasets/index.md) - Understanding datasets module
- [CelesTrak Details](../learn/datasets/celestrak.md) - CelesTrak data source specifics
- [TLE and SGP](../learn/tle_and_sgp.md) - Understanding TLE format and SGP propagation
- [Datasets API Reference](../library_api/datasets/index.md) - Complete function reference
