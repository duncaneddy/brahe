# OEM — Orbit Ephemeris Message

An Orbit Ephemeris Message (OEM) contains time-ordered sequences of spacecraft state vectors (position and velocity), optionally with accelerations and covariance matrices. OEM is the standard format for exchanging ephemeris data — for example, when handing off a trajectory between mission planning and flight dynamics teams, distributing predicted or definitive ephemerides, or providing data for conjunction screening.

## Structure

An OEM message consists of:

- **Header** — format version, creation date, originator, optional classification
- **One or more segments**, each containing:
    - **Metadata** — object name/ID, center body, reference frame, time system, time span, interpolation settings
    - **State vectors** — epoch + position + velocity (+ optional acceleration) per line
    - **Covariance blocks** (optional) — 6$\times$6 symmetric covariance matrices with optional epoch and reference frame

Multiple segments allow a single file to cover different time spans, reference frames, or trajectory arcs (e.g., before and after a maneuver).

## Parsing

```python
from brahe.ccsds import OEM

# From file (KVN, XML, or JSON — auto-detected)
oem = OEM.from_file("ephemeris.oem")

# From string
oem = OEM.from_str(kvn_content)
```

## Accessing Data

### Header and Metadata

```python
oem.format_version()   # float, e.g. 3.0
oem.originator()       # str, e.g. "NASA/JPL"
oem.classification()   # str or None

oem.num_segments()     # int
oem.object_name(0)     # str — name for segment 0
oem.object_id(0)       # str — international designator
oem.center_name(0)     # str — e.g. "EARTH", "MARS BARYCENTER"
oem.ref_frame(0)       # str — e.g. "EME2000", "GCRF", "ITRF2000"
```

### State Vectors

```python
oem.num_states(0)      # int — number of state vectors in segment 0

# Get a single state vector as a dictionary
sv = oem.state(0, 0)   # segment 0, state 0
# sv = {
#     "epoch": "1996-12-18T12:00:00.331",
#     "position": [2789619.0, -280045.0, -1746755.0],   # meters
#     "velocity": [4733.72, -2495.86, -1041.95],         # m/s
#     "acceleration": None                                # or [ax, ay, az] in m/s²
# }
```

### Covariance

```python
oem.num_covariances(0)  # int — number of covariance blocks in segment 0
```

Covariance matrices are accessible through `to_dict()` at the segment level.

### Full Serialization

```python
d = oem.to_dict()
# d = {
#     "header": { "format_version": 3.0, "originator": "NASA/JPL", ... },
#     "segments": [
#         {
#             "metadata": { "object_name": "...", "ref_frame": "...", ... },
#             "states": [ { "epoch": "...", "position": [...], "velocity": [...] }, ... ],
#             "comments": [...],
#             "num_covariances": 0
#         },
#         ...
#     ]
# }
```

## Writing

```python
# Write to KVN string
kvn = oem.to_string("KVN")

# Write to file
oem.to_file("output.oem", "KVN")
```

!!! info "Round-Trip Fidelity"
    Writing and re-parsing an OEM preserves all metadata, state vectors, and covariance data. Numeric precision may vary slightly due to floating-point formatting, but values are preserved within the precision of the output format.

## KVN Format Example

A minimal OEM KVN file looks like:

```text
CCSDS_OEM_VERS = 3.0
CREATION_DATE = 2024-01-15T00:00:00
ORIGINATOR = BRAHE

META_START
OBJECT_NAME = MY SATELLITE
OBJECT_ID = 2024-001A
CENTER_NAME = EARTH
REF_FRAME = EME2000
TIME_SYSTEM = UTC
START_TIME = 2024-01-15T00:00:00
STOP_TIME = 2024-01-15T01:00:00
META_STOP

2024-01-15T00:00:00  6878.137  0.000  0.000  0.000  7.612  0.000
2024-01-15T00:30:00  -3439.068  5957.355  0.000  -6.593  -3.806  0.000
2024-01-15T01:00:00  -3439.068  -5957.355  0.000  6.593  -3.806  0.000
```

The data lines contain epoch followed by position (km) and velocity (km/s), space-separated.

---

## See Also

- [API Reference — OEM](../../library_api/ccsds/oem.md)
- [CCSDS Data Formats](index.md) — Overview of all message types
- [Trajectories](../trajectories/index.md) — Brahe trajectory containers
