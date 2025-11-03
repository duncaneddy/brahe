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
brahe datasets celestrak download --group stations ~/satellite_data/stations.txt
```
Output:
```bash
✓ Downloaded stations satellites to ~/satellite_data/stations.txt
```

Download Starlink constellation:
```bash
brahe datasets celestrak download --group starlink ~/satellite_data/starlink.txt
```
Output:
```bash
✓ Downloaded stations satellites to ~/satellite_data/starlink.txt
```

Download GPS satellites:
```bash
brahe datasets celestrak download --group gps-ops ~/satellite_data/gps.txt
```
Output:
```bash
✓ Downloaded stations satellites to ~/satellite_data/gps.txt
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
```bash
# ISS (ZARYA) [NORAD ID: 25544]

# TLE Lines:
#   1 25544U 98067A   25306.42331346  .00010070  00000+0  18610-3 0  9998
#   2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601

# ✓ Found satellite 'ISS (ZARYA)'
```

Find Hubble Space Telescope:
```bash
brahe datasets celestrak lookup "HST"
```
Output:
```bash
# HST [NORAD ID: 20580]

# TLE Lines:
#   1 20580U 90037B   25306.35384425  .00007204  00000+0  25387-3 0  9994
#   2 20580  28.4668 187.2184 0001915 134.9270 225.1481 15.27276544753814

# ✓ Found satellite 'HST'
```

Find by partial name:
```bash
brahe datasets celestrak lookup "CAPELLA"
```
Output:
```bash
# CAPELLA-11 (ACADIA-1) [NORAD ID: 57693]

# TLE Lines:
#   1 57693U 23126A   25306.12190313 -.00002119  00000+0 -25509-3 0  9998
#   2 57693  53.0104  84.6427 0002550 126.4571 233.6640 14.78979627118491

# ✓ Found satellite 'CAPELLA-11 (ACADIA-1)'
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
brahe datasets celestrak show 25544 -s
```
Output:
```bash
# ISS (ZARYA) [NORAD ID: 25544]

# TLE Lines:
#   Line 1: 1 25544U 98067A   25306.42331346  .00010070  00000+0  18610-3 0  9998
#   Line 2: 2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601

# Orbital Elements:
#   Epoch:              2025-11-02 10:09:34.283 UTC
#   Ephemeris Age:      21h 16m 13s
#   Semi-major axis:    6795.7 km
#   Eccentricity:       0.0004969
#   Inclination:        51.6344°
#   RAAN:               342.0717°
#   Arg of Perigee:     8.9436°
#   Mean Anomaly:       351.1640°

# Orbital Characteristics:
#   Orbital Period:     92.9 min (1.55 hours)
#   Mean Motion:        15.497 rev/day
#   Perigee Altitude:   414.2 km
#   Apogee Altitude:    421.0 km
```

Show GPS satellite:
```bash
# brahe datasets celestrak show 32260
```
Output:
```bash
# ╭──────────────────────────────────────────────────────────────────────────────────────╮
# │ NAVSTAR 60 (USA 196)                                                                 │
# │ NORAD ID: 32260                                                                      │
# ╰──────────────────────────────────────────────────────────────────────────────────────╯
# ╭───────────────────────────────────── TLE Lines ──────────────────────────────────────╮
# │  Line 1:  1 32260U 07047A   25305.87057871  .00000045  00000+0  00000+0 0  9996      │
# │  Line 2:  2 32260  53.9265  89.4464 0164696  84.6485 277.2299  2.00566111132250      │
# ╰──────────────────────────────────────────────────────────────────────────────────────╯
# ╭────────────────────────────────── Orbital Elements ──────────────────────────────────╮
# │  Epoch            2025-11-01 20:53:38.001 UTC                                        │
# │  Ephemeris Age                 1d 10h 32m 23s                                        │
# │  Semi-major axis                   26560.1 km                                        │
# │  Eccentricity                       0.0164696                                        │
# │  Inclination                         53.9265°                                        │
# │  RAAN                                89.4464°                                        │
# │  Arg of Perigee                      84.6485°                                        │
# │  Mean Anomaly                       277.2299°                                        │
# ╰──────────────────────────────────────────────────────────────────────────────────────╯
# ╭────────────────────────────── Orbital Characteristics ───────────────────────────────╮
# │  Orbital Period    718.0 min (11.97 hours)                                           │
# │  Mean Motion                 2.006 rev/day                                           │
# │  Perigee Altitude               19744.6 km                                           │
# │  Apogee Altitude                20619.4 km                                           │
# ╰──────────────────────────────────────────────────────────────────────────────────────╯
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
```bash
Common CelesTrak Satellite Groups

┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Group Name           ┃ Description                                         ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ active               │ All active satellites                               │
│ stations             │ Space stations (ISS, Tiangong, etc.)                │
│ last-30-days         │ Satellites launched in the last 30 days             │
│ gnss                 │ All GNSS satellites (GPS, Galileo, GLONASS, Beidou) │
│ gps-ops              │ Operational GPS satellites                          │
│ galileo              │ Galileo navigation satellites                       │
│ beidou               │ Beidou navigation satellites                        │
│ glo-ops              │ Operational GLONASS satellites                      │
│ geo                  │ Geostationary satellites                            │
│ gpz                  │ Geostationary protected zone satellites             │
│ gpz-plus             │ Geostationary protected zone plus satellites        │
│ weather              │ Weather satellites                                  │
│ noaa                 │ NOAA satellites                                     │
│ goes                 │ GOES weather satellites                             │
│ starlink             │ SpaceX Starlink constellation                       │
│ oneweb               │ OneWeb constellation                                │
│ kuiper               │ Amazon Kuiper constellation                         │
│ qianfan              │ Qianfan constellation                               │
│ hulianwang           │ Hulianwang constellation                            │
│ planet               │ Planet Labs imaging satellites                      │
│ iridium              │ Iridium constellation                               │
│ iridium-NEXT         │ Iridium NEXT constellation                          │
│ intelsat             │ Intelsat satellites                                 │
│ eutelsat             │ Eutelsat satellites                                 │
│ ses                  │ SES satellites                                      │
│ orbcomm              │ Orbcomm satellites                                  │
│ globalstar           │ Globalstar satellites                               │
│ sarsat               │ Search and rescue satellites                        │
│ cubesat              │ CubeSats                                            │
│ amateur              │ Amateur radio satellites                            │
│ science              │ Science satellites                                  │
│ weather              │ Weather satellites                                  │
│ geodetic             │ Geodetic satellites                                 │
│ cosmos-2251-debris   │ Debris from Cosmos 2251 collision                   │
│ iridium-33-debris    │ Debris from Iridium 33 collision                    │
│ fengyun-1c-debris    │ Debris from Fengyun-1C ASAT test                    │
│ cosmos-1408-debris   │ Debris from Cosmos 1408 ASAT test                   │
└──────────────────────┴─────────────────────────────────────────────────────┘
```

---

### `celestrak search`

Search for satellites by name pattern within a group.

**Syntax:**
```bash
brahe datasets celestrak search <PATTERN> [OPTIONS]
```

**Arguments:**
- `PATTERN` - Name search pattern (case-insensitive)

**Options:**
- `--group <name>` - Satellite group to search (default: "active")
- `--table, -t` - Display results as table
- `--columns <preset>` - Columns to display: 'minimal', 'default', 'all', or comma-separated list

**Examples:**

Search for Capella satellites:
```bash
brahe datasets celestrak search "Capella"
```
Output:
```bash
# CAPELLA-11 (ACADIA-1) (NORAD: 57693)
# CAPELLA-14 (ACADIA-4) (NORAD: 59444)
# CAPELLA-13 (ACADIA-3) (NORAD: 60419)
# CAPELLA-15 (ACADIA-5) (NORAD: 60544)
# CAPELLA-17 (ACADIA-7) (NORAD: 64583)
# CAPELLA-16 (ACADIA-6) (NORAD: 65318)
```

Search for GPS satellites in specific group:
```bash
brahe datasets celestrak search "GPS II" --group gps-ops
```
Output:
```bash
# GPS BIIR-2  (PRN 13) (NORAD: 24876)
# GPS BIIR-4  (PRN 20) (NORAD: 26360)
# GPS BIIR-5  (PRN 22) (NORAD: 26407)
# GPS BIIR-8  (PRN 16) (NORAD: 27663)
# GPS BIIR-11 (PRN 19) (NORAD: 28190)
# GPS BIIR-13 (PRN 02) (NORAD: 28474)
# GPS BIIRM-1 (PRN 17) (NORAD: 28874)
# GPS BIIRM-2 (PRN 31) (NORAD: 29486)
# GPS BIIRM-3 (PRN 12) (NORAD: 29601)
# GPS BIIRM-4 (PRN 15) (NORAD: 32260)
# GPS BIIRM-5 (PRN 29) (NORAD: 32384)
# GPS BIIRM-6 (PRN 07) (NORAD: 32711)
# GPS BIIRM-8 (PRN 05) (NORAD: 35752)
```

Search with table output:
```bash
brahe datasets celestrak search "CAPELLA" --group active --table
```
Output:
```bash
# ┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┓
# ┃ Name               ┃ ID     ┃ Epoch                ┃ Age        ┃ Period (min) ┃ SMA (km)  ┃ Ecc      ┃ Inc (°) ┃ RAAN (°)  ┃ ArgP (°)  ┃ MA (°)  ┃
# ┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━┩
# │ CAPELLA-11         │ 57693  │ 2025-11-02           │ 1d 5h 13m  │ 97.4         │ 7010.7    │ 0.000255 │ 53.01   │ 84.64     │ 126.46    │ 233.66  │
# │ (ACADIA-1)         │        │ 02:55:32.430 UTC     │ 39s        │              │           │          │         │           │           │         │
# │ CAPELLA-14         │ 59444  │ 2025-11-02           │ 20h 33m 2s │ 95.7         │ 6930.2    │ 0.000330 │ 45.61   │ 6.37      │ 226.89    │ 133.18  │
# │ (ACADIA-4)         │        │ 11:36:09.001 UTC     │            │              │           │          │         │           │           │         │
# │ CAPELLA-13         │ 60419  │ 2025-11-01           │ 1d 23h 48m │ 96.8         │ 6983.4    │ 0.000158 │ 53.00   │ 332.08    │ 93.30     │ 266.82  │
# │ (ACADIA-3)         │        │ 08:20:23.000 UTC     │ 48s        │              │           │          │         │           │           │         │
# │ CAPELLA-15         │ 60544  │ 2025-11-01           │ 2d 4h 11m  │ 96.5         │ 6968.5    │ 0.000478 │ 97.70   │ 18.46     │ 121.04    │ 239.13  │
# │ (ACADIA-5)         │        │ 03:57:45.954 UTC     │ 25s        │              │           │          │         │           │           │         │
# │ CAPELLA-17         │ 64583  │ 2025-11-02           │ 1d 31m 29s │ 96.4         │ 6963.2    │ 0.000327 │ 97.76   │ 58.71     │ 21.22     │ 338.91  │
# │ (ACADIA-7)         │        │ 07:37:41.806 UTC     │            │              │           │          │         │           │           │         │
# │ CAPELLA-16         │ 65318  │ 2025-11-02           │ 1d 3h 57m  │ 96.6         │ 6973.1    │ 0.000296 │ 97.76   │ 19.81     │ 275.26    │ 84.83   │
# │ (ACADIA-6)         │        │ 04:11:12.548 UTC     │ 58s        │              │           │          │         │           │           │         │
# └────────────────────┴────────┴──────────────────────┴────────────┴──────────────┴───────────┴──────────┴─────────┴───────────┴───────────┴─────────┘
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
```bash
# Available groundstation providers:
#   - atlas
#   - aws
#   - ksat
#   - leaf
#   - ssc
#   - viasat
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
Output:
```bash
# All Groundstations (96 total):
#   Absheron: 40.470° lat, 49.490° lon, 0 m alt [S, X]
#   Accra: 5.600° lat, -0.300° lon, 0 m alt [L, S, X, Ka]
```

List KSAT stations only:
```bash
brahe datasets groundstations list-stations --provider ksat
```
Output:
```bash
# KSAT Groundstations (36 total):
#   Athens: 37.850° lat, 22.620° lon, 0 m alt [S, X, Optical]
#   Awarua: -46.530° lat, 168.380° lon, 0 m alt [S, X, Ka]
```

---

## See Also

- [CelesTrak](https://celestrak.org) - Official TLE data source
- [Two-Line Elements](../orbits/two_line_elements.md) - Understanding Two-Line Elements
- [SGP Propagation](../orbit_propagation/sgp_propagation.md) - TLE-based orbit propagation
- [Access CLI](access.md) - Compute satellite passes (uses TLE data)
- [Datasets API](../../library_api/datasets/index.md) - Python dataset functions
