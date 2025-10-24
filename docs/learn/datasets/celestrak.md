# CelesTrak Data Source

CelesTrak is a public source for satellite Two-Line Element (TLE) data, maintained by T.S. Kelso since 1985. It provides free, frequently updated orbital element sets for thousands of satellites, making it a useful resource for satellite tracking, orbit determination, and space situational awareness.

## Overview

**Website**: [https://celestrak.org](https://celestrak.org)

**Maintainer**: Dr. T.S. Kelso

**Update frequency**: Multiple times daily (varies by satellite priority)

**Access**: No registration or API key required


## Why CelesTrak?

### Advantages

**Free and Open**:
- No registration required
- No rate limits for reasonable use
- Publicly accessible worldwide
- Stable, long-term availability

**Well-Organized**:
- Satellites grouped by function and constellation
- Consistent naming conventions
- Regular group updates as missions change
- Special groupings for recent launches and interesting events


### Limitations

**Update Latency**:
- Data comes from Space-Track.org with some delay
- Not as current as direct Space-Track access
- Typically 1-6 hours behind during active tracking periods

**Limited Historical Data**:
- Focuses on current/recent TLEs
- Historical archives available but not comprehensive
- For deep historical analysis, use Space-Track.org directly

## Satellite Groups

CelesTrak organizes satellites into logical groups accessible via simple names. These groups are maintained as constellations evolve.

### Temporal Groups

| Group | Description | Typical Count |
|-------|-------------|---------------|
| `active` | All active satellites | ~5,000+ |
| `last-30-days` | Recently launched satellites | 20-100 (varies) |
| `tle-new` | Newly added TLEs (last 15 days) | Variable |

### Communications

| Group | Description |
|-------|-------------|
| `starlink` | SpaceX Starlink constellation |
| `oneweb` | OneWeb constellation |
| `kuiper` | Amazon Kuiper constellation |
| `intelsat` | Intelsat satellites |
| `eutelsat` | Eutelsat constellation |
| `orbcomm` | ORBCOMM constellation |
| `telesat` | Telesat constellation |
| `globalstar` | Globalstar constellation |
| `iridium-NEXT` | Iridium constellation |
| `qianfan` | Qianfan constellation |
| `hulianwang` | Hulianwang Digui constellation |


### Earth Observation

| Group | Description |
|-------|-------------|
| `weather` | Weather satellites (NOAA, GOES, Metop, etc.) |
| `earth-resources` | Earth observation (Landsat, Sentinel, etc.) |
| `planet` | Planet Labs imaging satellites |
| `spire` | Spire Global satellites |

### Navigation

| Group | Description |
|-------|-------------|
| `gnss` | All navigation satellites (GPS, GLONASS, Galileo, BeiDou, QZSS, IRNSS) |
| `gps-ops` | Operational GPS satellites only |
| `glonass-ops` | Operational GLONASS satellites only |
| `galileo` | European Galileo constellation |
| `beidou` | Chinese BeiDou/COMPASS constellation |
| `sbas` | Satellite-Based Augmentation System (WAAS/EGNOS/MSAS) |

### Scientific and Special Purpose

| Group | Description |
|-------|-------------|
| `science` | Scientific research satellites |
| `noaa` | NOAA satellites |
| `stations` | Space stations (ISS, Tiangong) |
| `analyst` | Analyst satellites (tracking placeholder IDs) |
| `visual` | 100 (or so) brightest objects |
| `gpz` | Geostationary Protected Zone |
| `gpz-plus` | Geostationary Protected Zone Plus |

**Note**: Group names and contents evolve as missions launch, deorbit, or change status. Visit [CelesTrak GP Element Sets](https://celestrak.org/NORAD/elements/gp.php) for the current complete list.

## Data Quality and Currency

TLE updates depend on the object itself, it's tracking priority, how difficult to track it is, and how often it's maneuvering. CelesTrak provides [statistics](https://celestrak.org/NORAD/elements/gp-statistics.php) on the median and mean age of TLEs by group. This has been about 0.8 and 1.5 days respectively in recent years, but can vary.

### Accuracy Considerations

For most operational applications TLEs should be updated daily. For research or general analysis, it's generally fine to just download and save the ephemeris once.

**Propagation accuracy**:
- Best within hours of epoch
- Degrades beyond 3-7 days for LEO
- Acceptable for 1-2 weeks for MEO/GEO

**When to refresh**:
- LEO tracking: Daily updates recommended
- GNSS analysis: Weekly updates acceptable
- GEO operations: Monthly updates sufficient

## Access Methods

### Brahe Integration

Brahe provides convenient API for CelesTrak access:

```python
import brahe as bh

# Get ephemeris data
ephemeris = bh.datasets.celestrak.get_ephemeris("gnss")
# Returns: List of (name, line1, line2) tuples

# Get as propagators
propagators = bh.datasets.celestrak.get_ephemeris_as_propagators(
    "gnss",
    step_size=60.0
)

# Download and save
bh.datasets.celestrak.download_ephemeris(
    "gnss",
    "gnss_satellites.json",
    content_format="3le",
    file_format="json"
)
```

## Best Practices

### Respectful Usage

While CelesTrak has no strict rate limits, follow these guidelines:

**DO**:
- Cache downloaded data locally
- Refresh on reasonable schedules (hourly at most)
- Use appropriate update intervals for your orbit regime
- Implement exponential backoff on errors

**DON'T**:
- Poll every minute (excessive)
- Download all groups when you need one
- Ignore errors and retry immediately
- Run unattended scripts without rate limiting

### Error Handling

Implement robust error handling:

```python
import brahe as bh
import time

def download_with_retry(group, max_retries=3):
    """Download ephemeris with exponential backoff"""
    for attempt in range(max_retries):
        try:
            ephemeris = bh.datasets.celestrak.get_ephemeris(group)
            return ephemeris
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Attempt {attempt + 1} failed: {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"All retries failed for {group}")
                raise

# Use with error handling
try:
    ephemeris = download_with_retry("gnss")
except Exception as e:
    print(f"Failed to download: {e}")
    # Fall back to cached data
```

### Caching Strategy

Implement local caching to reduce load:

```python
import brahe as bh
from pathlib import Path
import time
import json

def get_cached_ephemeris(group, cache_dir="./cache", max_age_hours=24):
    """Get ephemeris from cache or download if stale"""
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    cache_file = cache_path / f"{group}.json"

    # Check if cache exists and is fresh
    if cache_file.exists():
        age_seconds = time.time() - cache_file.stat().st_mtime
        age_hours = age_seconds / 3600

        if age_hours < max_age_hours:
            # Use cached data
            with open(cache_file, 'r') as f:
                return json.load(f)

    # Cache miss or stale - download fresh data
    ephemeris = bh.datasets.celestrak.get_ephemeris(group)

    # Save to cache
    with open(cache_file, 'w') as f:
        json.dump(ephemeris, f)

    return ephemeris

# Use cached ephemeris
ephemeris = get_cached_ephemeris("gnss", max_age_hours=6)
```

## See Also

- [Datasets Overview](index.md) - Understanding satellite ephemeris datasets
- [Two-Line Elements](../orbits/two_line_elements.md) - TLE and 3LE format details
- [Downloading TLE Data](../../examples/downloading_tle_data.md) - Practical examples
- [CelesTrak API Reference](../../library_api/datasets/celestrak.md) - Function documentation
