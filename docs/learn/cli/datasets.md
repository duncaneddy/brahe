# Datasets Commands

Download and query satellite ephemeris data and ground station information.

## Overview

The `datasets` command group provides access to:
- **CelesTrak** - Satellite TLE (Two-Line Element) data
- **Ground Stations** - Commercial ground station network databases

## CelesTrak Commands

### `celestrak download`

Download satellite ephemeris data from CelesTrak and save to file.

**Syntax:**
```bash
brahe datasets celestrak download <GROUP> <FILEPATH>
```

**Arguments:**
- `GROUP` - Satellite group name (e.g., 'stations', 'starlink', 'gps-ops')
- `FILEPATH` - Output file path for TLE data

**Examples:**

Download space station TLEs:
```bash
brahe datasets celestrak download stations ~/satellite_data/stations.txt
```

Download Starlink constellation:
```bash
brahe datasets celestrak download starlink ~/satellite_data/starlink.txt
```

Download GPS satellites:
```bash
brahe datasets celestrak download gps-ops ~/satellite_data/gps.txt
```

See available groups:
```bash
brahe datasets celestrak list-groups
```

---

### `celestrak lookup`

Look up a satellite by name and display its NORAD ID and TLE.

**Syntax:**
```bash
brahe datasets celestrak lookup <NAME>
```

**Arguments:**
- `NAME` - Satellite name (partial match supported)

**Examples:**

Find ISS:
```bash
brahe datasets celestrak lookup "ISS"
```
Output:
```
ISS (ZARYA)
NORAD ID: 25544
TLE:
1 25544U 98067A   24001.12345678  .00001234  00000-0  12345-4 0  9991
2 25544  51.6400 123.4567 0001234  12.3456 347.8901 15.50000000123456
```

Find Hubble Space Telescope:
```bash
brahe datasets celestrak lookup "HST"
```

Find by partial name:
```bash
brahe datasets celestrak lookup "STARLINK"
```
(Shows first match)

---

### `celestrak show`

Display TLE information and computed orbital parameters for a satellite.

**Syntax:**
```bash
brahe datasets celestrak show <NORAD_ID>
```

**Arguments:**
- `NORAD_ID` - NORAD catalog ID (integer)

**Examples:**

Show ISS TLE and orbit info:
```bash
brahe datasets celestrak show 25544
```
Output:
```
Satellite: ISS (ZARYA)
NORAD ID: 25544

TLE:
1 25544U 98067A   24001.12345678  .00001234  00000-0  12345-4 0  9991
2 25544  51.6400 123.4567 0001234  12.3456 347.8901 15.50000000123456

Orbital Parameters:
  Epoch:           2024-01-01 02:57:46 UTC
  Inclination:     51.6400°
  RAAN:            123.4567°
  Eccentricity:    0.0001234
  Arg. of Perigee: 12.3456°
  Mean Anomaly:    347.8901°
  Mean Motion:     15.50000000 rev/day

Computed:
  Period:          92.9 minutes
  Semi-major axis: 6797.1 km
  Apogee altitude: 420.2 km
  Perigee altitude: 417.1 km
```

Show GPS satellite:
```bash
brahe datasets celestrak show 32260
```

---

### `celestrak list-groups`

List commonly used CelesTrak satellite groups.

**Syntax:**
```bash
brahe datasets celestrak list-groups
```

**Examples:**
```bash
brahe datasets celestrak list-groups
```
Output:
```
Available CelesTrak Groups:

  Navigation:
    gps-ops           - GPS Operational Satellites
    gps-ops-gl        - GLONASS Operational Satellites
    galileo           - Galileo Satellites
    beidou            - BeiDou Satellites

  Communication:
    starlink          - Starlink Constellation
    oneweb            - OneWeb Constellation
    iridium           - Iridium Satellites

  Earth Observation:
    resource          - Earth Resource Satellites
    weather           - Weather Satellites
    noaa              - NOAA Satellites

  Science:
    stations          - Space Stations (ISS, Tiangong)
    science           - Scientific Satellites

  Special Interest:
    active            - All Active Satellites
    analyst           - Analyst Satellites
    2024-launches     - 2024 Launches
```

---

### `celestrak search`

Search for satellites by name pattern within a group.

**Syntax:**
```bash
brahe datasets celestrak search <GROUP> <PATTERN>
```

**Arguments:**
- `GROUP` - Satellite group name
- `PATTERN` - Name search pattern (case-insensitive)

**Examples:**

Search for Starlink satellites:
```bash
brahe datasets celestrak search starlink "1234"
```

Search for specific GPS satellite:
```bash
brahe datasets celestrak search gps-ops "GPS II"
```

---

## Ground Station Commands

### `groundstations list-providers`

List available ground station providers.

**Syntax:**
```bash
brahe datasets groundstations list-providers
```

**Examples:**
```bash
brahe datasets groundstations list-providers
```
Output:
```
Available Ground Station Providers:
  - ksat (Kongsberg Satellite Services)
  - atlas (Atlas Space Operations)
  - aws (AWS Ground Station)
  - leaf (Leaf Space)
```

---

### `groundstations list-stations`

List ground stations, optionally filtered by provider.

**Syntax:**
```bash
brahe datasets groundstations list-stations [OPTIONS]
```

**Options:**
- `--provider <name>` - Filter by provider name

**Examples:**

List all ground stations:
```bash
brahe datasets groundstations list-stations
```

List KSAT stations only:
```bash
brahe datasets groundstations list-stations --provider ksat
```
Output:
```
KSAT Ground Stations:

  Svalbard, Norway
    Latitude:  78.23° N
    Longitude: 15.39° E
    Altitude:  500 m

  Troll, Antarctica
    Latitude:  72.01° S
    Longitude: 2.53° E
    Altitude:  1270 m

  Singapore
    Latitude:  1.34° N
    Longitude: 103.99° E
    Altitude:  15 m
```

List AWS Ground Stations:
```bash
brahe datasets groundstations list-stations --provider aws
```

---

### `groundstations show`

Show ground stations for a specific provider (deprecated - use `list-stations`).

**Syntax:**
```bash
brahe datasets groundstations show <PROVIDER>
```

**Examples:**
```bash
brahe datasets groundstations show ksat
```

**Note:** This command is deprecated. Use `list-stations --provider <name>` instead.

---

### `groundstations show-all`

Show ground stations from all providers.

**Syntax:**
```bash
brahe datasets groundstations show-all
```

**Examples:**
```bash
brahe datasets groundstations show-all
```

**Note:** Equivalent to `list-stations` without filters.

---

## Common Workflows

### Download Satellite Data

```bash
#!/bin/bash
# Download TLE data for mission-relevant satellites

DATA_DIR="./satellite_data"
mkdir -p "$DATA_DIR"

echo "Downloading satellite TLE data..."

# Space stations
brahe datasets celestrak download stations "$DATA_DIR/stations.txt"

# GPS constellation
brahe datasets celestrak download gps-ops "$DATA_DIR/gps.txt"

# Starlink
brahe datasets celestrak download starlink "$DATA_DIR/starlink.txt"

# Weather satellites
brahe datasets celestrak download weather "$DATA_DIR/weather.txt"

echo "Download complete!"
```

### Find Satellite NORAD ID

```bash
#!/bin/bash
# Find NORAD ID for satellite access computation

SATELLITE_NAME="ISS"

echo "Looking up: $SATELLITE_NAME"
brahe datasets celestrak lookup "$SATELLITE_NAME"

# Extract just the NORAD ID (for scripting)
NORAD_ID=$(brahe datasets celestrak lookup "$SATELLITE_NAME" | grep "NORAD ID:" | awk '{print $3}')
echo "NORAD ID: $NORAD_ID"

# Compute access windows
brahe access compute $NORAD_ID --lat 40.7128 --lon -74.0060
```

### Orbital Parameter Survey

```bash
#!/bin/bash
# Survey orbital parameters for satellite constellation

SATS=(25544 20580 43013)  # ISS, HST, sample Starlink

echo "Satellite | Altitude | Inclination | Period"
echo "----------|----------|-------------|--------"

for norad in "${SATS[@]}"; do
  echo "Processing NORAD $norad..."
  brahe datasets celestrak show $norad | grep -E "Altitude|Inclination|Period"
done
```

### Ground Station Selection

```bash
#!/bin/bash
# Select ground stations for satellite mission

SATELLITE=25544  # ISS
PROVIDERS=("ksat" "aws" "atlas")

echo "Evaluating ground stations for NORAD $SATELLITE"

for provider in "${PROVIDERS[@]}"; do
  echo ""
  echo "=== $provider stations ==="

  # List stations
  brahe datasets groundstations list-stations --provider $provider

  # Would compute access for each station (manual selection needed)
done
```

### TLE Data Update

```bash
#!/bin/bash
# Regularly update TLE data for accuracy

DATA_DIR="./satellite_data"
GROUPS=("stations" "gps-ops" "active")

echo "Updating TLE data: $(date)"

for group in "${GROUPS[@]}"; do
  echo "Updating $group..."
  brahe datasets celestrak download $group "$DATA_DIR/${group}.txt"
done

echo "TLE data updated: $(date)"
```

---

## Tips

### TLE Data Freshness

TLEs degrade over time:
- **LEO satellites**: Update daily for accurate orbit prediction
- **MEO/GEO satellites**: Update weekly acceptable
- **Mission-critical**: Update before each operation

### NORAD Catalog IDs

- Assigned sequentially by USSPACECOM
- Unique for each satellite
- Required for most Brahe access computations
- Find via `celestrak lookup` or online databases

### CelesTrak Groups

Common group names:
- `stations` - ISS, Tiangong, etc.
- `starlink` - Starlink constellation
- `gps-ops` - GPS operational satellites
- `galileo` - Galileo GNSS
- `active` - All active satellites
- `2024-launches` - Recent launches (year-specific)

### Ground Station Providers

**KSAT (Kongsberg Satellite Services):**
- Polar ground stations (Svalbard, Troll)
- Excellent coverage for polar orbits

**AWS Ground Station:**
- Global network
- Cloud-integrated

**Atlas Space Operations:**
- Freedom Network
- Multiple global sites

**Leaf Space:**
- European and global coverage

### Scripting with Datasets

Extract specific info:
```bash
# Get just NORAD ID
brahe datasets celestrak lookup "ISS" | grep "NORAD ID:" | awk '{print $3}'

# Get orbital period
brahe datasets celestrak show 25544 | grep "Period:" | awk '{print $2, $3}'
```

---

## See Also

- [CelesTrak](https://celestrak.org) - Official TLE data source
- [TLE Format](../tle_format.md) - Understanding Two-Line Elements
- [SGP4 Propagation](../sgp4.md) - TLE-based orbit propagation
- [Access CLI](access.md) - Compute satellite passes (uses TLE data)
- [Datasets API](../../library_api/datasets/index.md) - Python dataset functions
