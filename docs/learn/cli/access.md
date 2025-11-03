# Access Commands

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
Output:
```bash
# Access Windows for ISS (ZARYA) (NORAD ID: 25544)
# Location: 40.7128° lat, -74.0060° lon, 0 m alt
# Minimum elevation: 10.0°
# Found 18 access window(s)
...
```

GPS satellite passes (15° minimum elevation):
```bash
brahe access compute 32260 --lat 40.7128 --lon -74.0060 --min-elevation 15
```
Output:
```bash
# Access Windows for NAVSTAR 60 (USA 196) (NORAD ID: 32260)
# Location: 40.7128° lat, -74.0060° lon, 0 m alt
# Minimum elevation: 15.0°
# Found 8 access window(s)
```

Custom time range:
```bash
brahe access compute 25544 --lat 40.7128 --lon -74.0060 \
  --start-time "2024-06-01T00:00:00Z" \
  --duration 1
```
Output:
```bash
# Access Windows for ISS (ZARYA) (NORAD ID: 25544)
# Location: 40.7128° lat, -74.0060° lon, 0 m alt
# Minimum elevation: 10.0°
# Found 2 access window(s)
```

Use ground station database:
```bash
brahe access compute 25544 --gs-provider ksat --gs-name "Svalbard"
```
Output:
```bash
# Access Windows for ISS (ZARYA) (NORAD ID: 25544)
# Location: 78.2300° lat, 15.4100° lon, 0 m alt
# Minimum elevation: 10.0°
# Found 19 access window(s)
```

Simple output format:
```bash
brahe access compute 25544 --lat 40.7128 --lon -74.0060 --output-format simple
```
Output:
```bash
# Access Windows for ISS (ZARYA) (NORAD ID: 25544)
# Location: 40.7128° lat, -74.0060° lon, 0 m alt
# Minimum elevation: 10.0°
# Found 18 access window(s)

# 1. 2025-11-03 18:36:18.197 UTC | 2025-11-03 18:38:16.000 UTC | 1m 57s | Max Elev: 11.9° | Az: 150°-114°
# 2. 2025-11-03 20:11:37.226 UTC | 2025-11-03 20:16:16.000 UTC | 4m 38s | Max Elev: 20.6° | Az: 255°-351°
```

Export to JSON:
```bash
brahe access compute 25544 --lat 40.7128 --lon -74.0060 --output-file passes.json
```
Output:
```bash
# Access Windows for ISS (ZARYA) (NORAD ID: 25544)
# Location: 40.7128° lat, -74.0060° lon, 0 m alt
# Minimum elevation: 10.0°
# Found 18 access window(s)
```

Sort by maximum elevation (highest first):
```bash
brahe access compute 25544 --lat 40.7128 --lon -74.0060 \
  --sort-by max_elevation --sort-order descending --max-results 5
```
Output:
```bash
# Access Windows for ISS (ZARYA) (NORAD ID: 25544)
# Location: 40.7128° lat, -74.0060° lon, 0 m alt
# Minimum elevation: 10.0°
# Found 5 access window(s)

# ┏━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
# ┃         ┃ Start   ┃ End    ┃         ┃    Max ┃   Start ┃        ┃
# ┃ Contact ┃ Time    ┃ Time   ┃         ┃   Elev ┃      Az ┃ End Az ┃
# ┃       # ┃ (UTC)   ┃ (UTC)  ┃ Durati… ┃  (deg) ┃   (deg) ┃  (deg) ┃
# ┡━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
# │       1 │ 2025-1… │ 2025-… │ 6       │   77.0 │     321 │    148 │
# │         │ 05:22:… │ 05:28… │ minutes │        │         │        │
# │         │ UTC     │ UTC    │ and     │        │         │        │
# │         │         │        │ 31.74   │        │         │        │
# │         │         │        │ seconds │        │         │        │
# │       2 │ 2025-1… │ 2025-… │ 6       │   69.8 │     319 │    151 │
# │         │ 06:58:… │ 07:04… │ minutes │        │         │        │
# │         │ UTC     │ UTC    │ and     │        │         │        │
# │         │         │        │ 30.04   │        │         │        │
# │         │         │        │ seconds │        │         │        │
# │       3 │ 2025-1… │ 2025-… │ 6       │   57.9 │     206 │     46 │
# │         │ 18:34:… │ 18:40… │ minutes │        │         │        │
# │         │ UTC     │ UTC    │ and     │        │         │        │
# │         │         │        │ 7.79    │        │         │        │
# │         │         │        │ seconds │        │         │        │
# │       4 │ 2025-1… │ 2025-… │ 6       │   52.9 │     204 │     48 │
# │         │ 16:58:… │ 17:04… │ minutes │        │         │        │
# │         │ UTC     │ UTC    │ and     │        │         │        │
# │         │         │        │ 9.76    │        │         │        │
# │         │         │        │ seconds │        │         │        │
# │       5 │ 2025-1… │ 2025-… │ 5       │   52.6 │     227 │     22 │
# │         │ 17:46:… │ 17:52… │ minutes │        │         │        │
# │         │ UTC     │ UTC    │ and     │        │         │        │
# │         │         │        │ 59.45   │        │         │        │
# │         │         │        │ seconds │        │         │        │
# └─────────┴─────────┴────────┴─────────┴────────┴─────────┴────────┘
```

Sort by duration (longest passes first):
```bash
brahe access compute 25544 --lat 40.7128 --lon -74.0060 \
  --sort-by duration --sort-order descending
```
Output:
```bash
# Access Windows for ISS (ZARYA) (NORAD ID: 25544)
# Location: 40.7128° lat, -74.0060° lon, 0 m alt
# Minimum elevation: 10.0°
# Found 18 access window(s)

# ┏━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
# ┃         ┃ Start   ┃ End    ┃         ┃    Max ┃   Start ┃        ┃
# ┃ Contact ┃ Time    ┃ Time   ┃         ┃   Elev ┃      Az ┃ End Az ┃
# ┃       # ┃ (UTC)   ┃ (UTC)  ┃ Durati… ┃  (deg) ┃   (deg) ┃  (deg) ┃
# ┡━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
# │       1 │ 2025-1… │ 2025-… │ 6       │   53.8 │     204 │     47 │
# │         │ 16:58:… │ 17:05… │ minutes │        │         │        │
# │         │ UTC     │ UTC    │ and     │        │         │        │
# │         │         │        │ 24.76   │        │         │        │
# │         │         │        │ seconds │        │         │        │
# │       2 │ 2025-1… │ 2025-… │ 6       │   59.5 │     206 │     45 │
# │         │ 18:34:… │ 18:41… │ minutes │        │         │        │
# │         │ UTC     │ UTC    │ and     │        │         │        │
# └─────────┴─────────┴────────┴─────────┴────────┴─────────┴────────┘
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

See [Datasets CLI](datasets.md) for more details.

### Negative Longitudes

For westerly longitudes, use negative values with `=` or `--`:
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
