# Ephemeris Data Sources

Brahe provides typed clients for two satellite ephemeris data sources: [CelesTrak](celestrak.md) and [Space-Track](spacetrack/index.md). Both clients return `GPRecord` for General Perturbations queries, enabling code that works interchangeably with either data source.

## GPRecord

`GPRecord` is the shared data type for General Perturbations (OMM) data returned by both `CelestrakClient.query_gp()` and `SpaceTrackClient.query_gp()`. It contains 40 fields organized into three categories based on their data type.

### Metadata Fields (String)

These fields contain textual metadata about the object and data record:

<div class="center-table" markdown="1">

| Field | Python Type | Description |
|-------|-------------|-------------|
| `ccsds_omm_vers` | `Optional[str]` | CCSDS OMM version |
| `comment` | `Optional[str]` | Comment field |
| `creation_date` | `Optional[str]` | Record creation date |
| `originator` | `Optional[str]` | Data originator |
| `object_name` | `Optional[str]` | Satellite common name |
| `object_id` | `Optional[str]` | International designator |
| `center_name` | `Optional[str]` | Center name (typically "EARTH") |
| `ref_frame` | `Optional[str]` | Reference frame (typically "TEME") |
| `time_system` | `Optional[str]` | Time system (typically "UTC") |
| `mean_element_theory` | `Optional[str]` | Mean element theory (typically "SGP4") |
| `epoch` | `Optional[str]` | Epoch of the orbital elements |
| `classification_type` | `Optional[str]` | Classification (U=Unclassified, C=Classified, S=Secret) |
| `object_type` | `Optional[str]` | Object type (PAYLOAD, ROCKET BODY, DEBRIS, etc.) |
| `rcs_size` | `Optional[str]` | Radar cross-section size category (SMALL, MEDIUM, LARGE) |
| `country_code` | `Optional[str]` | Country code of the launching state |
| `launch_date` | `Optional[str]` | Launch date |
| `site` | `Optional[str]` | Launch site code |
| `decay_date` | `Optional[str]` | Decay date (if decayed) |
| `tle_line0` | `Optional[str]` | TLE line 0 (object name) |
| `tle_line1` | `Optional[str]` | TLE line 1 |
| `tle_line2` | `Optional[str]` | TLE line 2 |

</div>

### Orbital Element Fields (Numeric)

These fields contain orbital mechanics parameters. In Rust, they are typed as `Option<f64>` and deserialized from either string (SpaceTrack) or numeric (CelesTrak) JSON values.

<div class="center-table" markdown="1">

| Field | Python Type | Rust Type | Description |
|-------|-------------|-----------|-------------|
| `mean_motion` | `Optional[float]` | `Option<f64>` | Mean motion (rev/day) |
| `eccentricity` | `Optional[float]` | `Option<f64>` | Orbital eccentricity |
| `inclination` | `Optional[float]` | `Option<f64>` | Orbital inclination (deg) |
| `ra_of_asc_node` | `Optional[float]` | `Option<f64>` | Right ascension of ascending node (deg) |
| `arg_of_pericenter` | `Optional[float]` | `Option<f64>` | Argument of pericenter (deg) |
| `mean_anomaly` | `Optional[float]` | `Option<f64>` | Mean anomaly (deg) |
| `bstar` | `Optional[float]` | `Option<f64>` | BSTAR drag coefficient |
| `mean_motion_dot` | `Optional[float]` | `Option<f64>` | First derivative of mean motion |
| `mean_motion_ddot` | `Optional[float]` | `Option<f64>` | Second derivative of mean motion |
| `semimajor_axis` | `Optional[float]` | `Option<f64>` | Semi-major axis (km) |
| `period` | `Optional[float]` | `Option<f64>` | Orbital period (min) |
| `apoapsis` | `Optional[float]` | `Option<f64>` | Apoapsis altitude (km) |
| `periapsis` | `Optional[float]` | `Option<f64>` | Periapsis altitude (km) |

</div>

### Identifier Fields (Integer)

These fields contain numeric identifiers. Like the orbital element fields, they accept both string and numeric JSON representations.

<div class="center-table" markdown="1">

| Field | Python Type | Rust Type | Description |
|-------|-------------|-----------|-------------|
| `norad_cat_id` | `Optional[int]` | `Option<u32>` | NORAD catalog identifier |
| `element_set_no` | `Optional[int]` | `Option<u16>` | Element set number |
| `rev_at_epoch` | `Optional[int]` | `Option<u32>` | Revolution number at epoch |
| `ephemeris_type` | `Optional[int]` | `Option<u8>` | Ephemeris type |
| `file` | `Optional[int]` | `Option<u64>` | File number |
| `gp_id` | `Optional[int]` | `Option<u32>` | GP record identifier |

</div>

!!! info "Flexible Deserialization"
    SpaceTrack returns all JSON values as strings (e.g., `"NORAD_CAT_ID": "25544"`), while CelesTrak returns numeric fields as JSON numbers (e.g., `"NORAD_CAT_ID": 25544`). GPRecord uses custom deserializers that accept both formats transparently, so the same code works with data from either source.

## Operator Functions

The `operators` module provides functions that generate operator-prefixed strings for use in query filters. These operators work with both `SpaceTrackQuery.filter()` and `CelestrakQuery.filter()`:

<div class="center-table" markdown="1">

| Function | Output | Example |
|----------|--------|---------|
| `greater_than(v)` | `">v"` | `">25544"` |
| `less_than(v)` | `"<v"` | `"<0.01"` |
| `not_equal(v)` | `"<>v"` | `"<>DEBRIS"` |
| `inclusive_range(a, b)` | `"a--b"` | `"25544--25600"` |
| `like(v)` | `"~~v"` | `"~~STARLINK"` |
| `startswith(v)` | `"^v"` | `"^NOAA"` |
| `now()` | `"now"` | `"now"` |
| `now_offset(days)` | `"now-N"` / `"now+N"` | `"now-7"` |
| `null_val()` | `"null-val"` | `"null-val"` |
| `or_list(vals)` | `"v1,v2,v3"` | `"25544,48274"` |

</div>

Operators compose naturally. For example, `greater_than(now_offset(-7))` produces `">now-7"`.

In Python, access these via `brahe.spacetrack.operators`:

```python
from brahe.spacetrack import operators as op
op.greater_than("25544")  # ">25544"
```

## CelesTrak vs Space-Track

<div class="center-table" markdown="1">

| Feature | CelesTrak | Space-Track |
|---------|-----------|-------------|
| **Authentication** | None required | Free account required |
| **GP data** | `GPRecord` | `GPRecord` |
| **SATCAT data** | `CelestrakSATCATRecord` | `SATCATRecord` |
| **Server-side filtering** | Limited (GROUP, CATNR, NAME, INTDES) | Full (any field) |
| **Client-side filtering** | Supported via operators | Not needed (server filters) |
| **File operations** | Not available | FileShare, SP Ephemeris, Public Files |
| **Rate limiting** | 6-hour client cache | Built-in sliding window limiter |
| **Output formats** | JSON, 3LE, CSV, XML | JSON, TLE, CSV, XML, KVN |
| **Supplemental GP** | SupGP endpoint (constellation operators) | Not available |

</div>

## Subpages

- [CelesTrak](celestrak.md) -- Public ephemeris data source (no account required)
- [Space-Track](spacetrack/index.md) -- Authoritative catalog data (account required)

---

## See Also

- [Ephemeris API Reference](../../library_api/ephemeris/index.md) -- Complete function documentation
- [Two-Line Elements](../orbits/two_line_elements.md) -- TLE and 3LE format details
