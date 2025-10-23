# Datasets Subcommand

Download satellite ephemeris data and groundstation information.

## Overview

The `datasets` subcommand provides access to curated satellite and groundstation datasets, including TLE data from CelesTrak and groundstation network information.

## Subcommands

### `celestrak`

Download satellite ephemeris data from CelesTrak.

#### `celestrak download`

Download satellite TLE data from CelesTrak and save to file.

**Syntax:**
```bash
brahe datasets celestrak download <filepath> [options]
```

**Arguments:**
- `filepath` - Output file path

**Options:**
- `--group <name>` - Satellite group name (default: `active`)
- `--content-format <tle|3le>` - Content format:
  - `tle` - Two-line elements (classic TLE format)
  - `3le` - Three-line elements (includes satellite names)
- `--file-format <txt|csv|json>` - File format (default: `txt`)

**Common Groups:**
- `active` - All active satellites
- `stations` - Space stations
- `gnss` - GPS, GLONASS, Galileo, BeiDou
- `starlink` - SpaceX Starlink constellation
- `oneweb` - OneWeb constellation
- `planet` - Planet Labs satellites
- `weather` - Weather satellites
- `science` - Scientific satellites
- `geo` - Geostationary satellites
- `amateur` - Amateur radio satellites

**Examples:**

```bash
# Download active satellites as JSON
brahe datasets celestrak download satellites.json \
    --group active \
    --content-format 3le \
    --file-format json

# Download GNSS satellites as text
brahe datasets celestrak download gnss.txt \
    --group gnss \
    --content-format tle \
    --file-format txt

# Download Starlink constellation
brahe datasets celestrak download starlink.csv \
    --group starlink \
    --file-format csv

# Download weather satellites
brahe datasets celestrak download weather.json \
    --group weather \
    --content-format 3le \
    --file-format json
```

**Output Formats:**

**Text (TLE):**
```
1 25544U 98067A   21001.00000000  .00002182  00000-0  41420-4 0  9990
2 25544  51.6461 339.8014 0002571  34.5857 120.4689 15.48919393265104
```

**Text (3LE):**
```
ISS (ZARYA)
1 25544U 98067A   21001.00000000  .00002182  00000-0  41420-4 0  9990
2 25544  51.6461 339.8014 0002571  34.5857 120.4689 15.48919393265104
```

**JSON:**
```json
[
  {
    "name": "ISS (ZARYA)",
    "line1": "1 25544U 98067A   21001.00000000  .00002182  00000-0  41420-4 0  9990",
    "line2": "2 25544  51.6461 339.8014 0002571  34.5857 120.4689 15.48919393265104"
  }
]
```

---

### `groundstations`

Access groundstation network information.

#### `groundstations list`

List available groundstation providers.

**Syntax:**
```bash
brahe datasets groundstations list
```

**Examples:**

```bash
brahe datasets groundstations list
```

**Output:**
```
Available groundstation providers:
  - ksat
  - atlas
  - aws
  - azure
  - leaf
  - ....
```

---

#### `groundstations show`

Show groundstations for a specific provider.

**Syntax:**
```bash
brahe datasets groundstations show <provider> [options]
```

**Arguments:**
- `provider` - Provider name (e.g., `ksat`, `atlas`, `aws`)

**Options:**
- `--properties` / `-p` - Show station properties (frequency bands, etc.)

**Examples:**

```bash
# Show KSAT stations
brahe datasets groundstations show ksat

# Show Atlas stations with properties
brahe datasets groundstations show atlas --properties

# Show AWS ground station network
brahe datasets groundstations show aws -p
```

**Output:**
```
KSAT Groundstations (20 total):
--------------------------------------------------------------------------------

Svalbard
  Location:   15.627° lon,  78.223° lat,    500 m alt

Troll
  Location:    2.535° lon, -72.011° lat,   1270 m alt

...

✓ Loaded 20 groundstations from ksat
```

**With Properties:**
```
Svalbard
  Location:   15.627° lon,  78.223° lat,    500 m alt
  Frequency bands: S, X, Ka
  Provider: KSAT
```

---

#### `groundstations show-all`

Show groundstations from all providers.

**Syntax:**
```bash
brahe datasets groundstations show-all [options]
```

**Options:**
- `--properties` / `-p` - Show station properties

**Examples:**

```bash
# Show all groundstations
brahe datasets groundstations show-all

# Show all with properties
brahe datasets groundstations show-all --properties
```

**Output:**
```
All Groundstations (150 total):
================================================================================

ksat (20 stations):
--------------------------------------------------------------------------------

  Svalbard
    Location:   15.627° lon,  78.223° lat,    500 m alt

  Troll
    Location:    2.535° lon, -72.011° lat,   1270 m alt

atlas (30 stations):
--------------------------------------------------------------------------------

  Awarua
    Location:  168.386° lon, -46.518° lat,     10 m alt

...

✓ Loaded 150 groundstations from all providers
```

## Common Workflows

### Daily TLE Updates

```bash
#!/bin/bash
# Download fresh TLE data daily

data_dir="$HOME/.brahe/tle"
mkdir -p "$data_dir"

# Download various satellite groups
brahe datasets celestrak download "$data_dir/active.json" \
    --group active --file-format json

brahe datasets celestrak download "$data_dir/gnss.json" \
    --group gnss --file-format json

brahe datasets celestrak download "$data_dir/starlink.json" \
    --group starlink --file-format json

echo "TLE data updated: $(date)"
```

### Mission Planning Setup

```bash
#!/bin/bash
# Set up data for mission planning

# Create directory structure
mkdir -p mission_data/{satellites,groundstations}

# Download satellite data
echo "Downloading satellite data..."
brahe datasets celestrak download mission_data/satellites/active.json \
    --group active --content-format 3le --file-format json

# Get groundstation info
echo "Getting groundstation information..."
brahe datasets groundstations show-all > mission_data/groundstations/all_stations.txt

echo "Mission data ready in mission_data/"
```

### Constellation Analysis

```bash
#!/bin/bash
# Analyze satellite constellations

echo "Constellation Sizes"
echo "==================="

for group in starlink oneweb planet; do
    brahe datasets celestrak download /tmp/${group}.json \
        --group $group --file-format json 2>/dev/null

    count=$(cat /tmp/${group}.json | grep -c "name")

    echo "$group: $count satellites"
done
```

### Ground Network Coverage

```bash
#!/bin/bash
# Analyze ground network coverage by provider

echo "Ground Network Summary"
echo "======================"

providers=$(brahe datasets groundstations list 2>/dev/null | tail -n +2 | sed 's/  - //')

for provider in $providers; do
    output=$(brahe datasets groundstations show $provider 2>/dev/null)
    count=$(echo "$output" | grep -oP '\d+(?= total)' | head -1)

    echo "$provider: $count stations"
done
```

## File Format Comparison

### Use Cases

| Format | Best For | Pros | Cons |
|--------|----------|------|------|
| **TXT** | Legacy systems, simple parsing | Standard format, widely supported | No metadata, harder to parse |
| **CSV** | Spreadsheet analysis, batch processing | Structured, easy to import | Limited nested data |
| **JSON** | Modern applications, APIs | Flexible, includes metadata | Larger file size |

### Content Format Comparison

| Format | Includes Name | Use Case |
|--------|---------------|----------|
| **TLE** (2-line) | No | Minimal data, standard format |
| **3LE** (3-line) | Yes | Easier identification, recommended |

## CelesTrak Satellite Groups

### Full Group List

Common groups include:

- **Mega-Constellations**: `starlink`, `oneweb`
- **Navigation**: `gnss`, `gps-ops`, `glonass-ops`, `galileo`, `beidou`
- **Communication**: `geo`, `intelsat`, `ses`, `iridium`, `orbcomm`
- **Earth Observation**: `planet`, `spire`, `capella`
- **Science**: `science`, `geodetic`, `engineering`
- **Weather**: `weather`, `noaa`, `goes`
- **Space Stations**: `stations`
- **Special**: `active`, `analyst`, `classified`

For a complete list, visit [CelesTrak](https://celestrak.org/NORAD/elements/).

## Notes

- **Update Frequency**: CelesTrak updates TLE data multiple times per day. Download fresh data regularly for accurate propagation.
- **File Size**: JSON format with 3LE content is larger but more usable than TXT.
- **Caching**: The CLI does not cache downloaded data. Implement your own caching if needed.
- **Rate Limits**: CelesTrak has rate limits. Avoid excessive requests.
- **Groundstation Data**: Groundstation coordinates are from public sources and may not reflect current operational status.

## See Also

- [CelesTrak Dataset](../../library_api/datasets/celestrak.md) - Python API
- [Groundstations Dataset](../../library_api/datasets/groundstations.md) - Python API
- [TLE Format](../orbits/two_line_elements.md) - TLE explanation
- [Downloading TLE Data Example](../../examples/downloading_tle_data.md) - Complete workflow
