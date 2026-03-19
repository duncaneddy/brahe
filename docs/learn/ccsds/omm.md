# OMM — Orbit Mean-elements Message

An Orbit Mean-elements Message (OMM) contains mean Keplerian elements and associated TLE parameters used for SGP4/SDP4 analytical propagation. OMM is the CCSDS-standardized representation of General Perturbations (GP) data — the same orbital information traditionally distributed as Two-Line Element sets (TLEs), but in a structured, self-describing format.

Data sources like CelesTrak and Space-Track distribute GP data as OMM messages (typically in XML or JSON), making OMM the modern successor to the fixed-width TLE format.

## Structure

An OMM message consists of:

- **Header** — format version, creation date, originator
- **Metadata** — object name/ID, center body, reference frame, time system, mean element theory (e.g., "SGP/SGP4")
- **Mean elements** — epoch, mean motion (or semi-major axis), eccentricity, inclination, RAAN, argument of pericenter, mean anomaly, optional GM
- **TLE parameters** (optional) — NORAD catalog ID, classification, element set number, revolution count, BSTAR, mean motion derivatives
- **Spacecraft parameters** (optional) — mass, drag/SRP areas and coefficients
- **Covariance** (optional) — 6$\times$6 covariance matrix
- **User-defined parameters** (optional) — arbitrary key-value pairs

## Parsing

```python
from brahe.ccsds import OMM

# From file (KVN, XML, or JSON — auto-detected)
omm = OMM.from_file("gp_data.omm")

# From string
omm = OMM.from_str(kvn_content)
```

## Accessing Data

### Metadata

```python
omm.format_version()       # 3.0
omm.object_name()          # "GOES 9"
omm.object_id()            # "1995-025A"
omm.center_name()          # "EARTH"
omm.ref_frame()            # "TEME"
omm.time_system()          # "UTC"
omm.mean_element_theory()  # "SGP/SGP4"
```

### Mean Elements

Mean elements use CCSDS/TLE-native units — mean motion in rev/day, angles in degrees:

```python
omm.mean_motion()          # 1.00273272 (rev/day), or None
omm.eccentricity()         # 0.0005013
omm.inclination()          # 3.0539 (degrees)
omm.ra_of_asc_node()       # 81.7939 (degrees)
omm.arg_of_pericenter()    # 249.2363 (degrees)
omm.mean_anomaly()         # 150.1602 (degrees)
omm.gm()                   # 3.986008e+14 (m³/s², converted from km³/s²)
```

!!! info "Unit Convention for OMM"
    Mean motion, angles, and TLE drag terms are kept in their CCSDS/TLE-native units (rev/day, degrees, etc.) because these values are needed as-is for TLE generation and SGP4 initialization. Only GM is converted to SI (m$^3$/s$^2$).

### TLE Parameters

```python
omm.norad_cat_id()         # 23581
omm.classification_type()  # "U"
omm.ephemeris_type()       # 0
omm.element_set_no()       # 925
omm.rev_at_epoch()         # 4316
omm.bstar()                # 0.0001
omm.mean_motion_dot()      # -1.13e-06 (rev/day²)
omm.mean_motion_ddot()     # 0.0 (rev/day³)
```

### Full Serialization

```python
d = omm.to_dict()
# d = {
#     "header": { "format_version": 3.0, "originator": "NOAA/USA", ... },
#     "metadata": { "object_name": "GOES 9", "ref_frame": "TEME", ... },
#     "mean_elements": { "epoch": "...", "mean_motion": 1.00273272, ... },
#     "tle_parameters": { "norad_cat_id": 23581, "bstar": 0.0001, ... },
#     "spacecraft_parameters": { ... },  # if present
#     "user_defined": { ... }            # if present
# }
```

The `to_dict()` method returns all data in the message, including optional sections that may not be present in every file. Missing optional sections are omitted from the dictionary.

## KVN Format Example

A minimal OMM KVN file:

```text
CCSDS_OMM_VERS = 3.0
CREATION_DATE = 2024-01-15T00:00:00
ORIGINATOR = EXAMPLE

OBJECT_NAME = ISS (ZARYA)
OBJECT_ID = 1998-067A
CENTER_NAME = EARTH
REF_FRAME = TEME
TIME_SYSTEM = UTC
MEAN_ELEMENT_THEORY = SGP/SGP4

EPOCH = 2024-01-15T12:00:00
MEAN_MOTION = 15.50100000
ECCENTRICITY = 0.0006180
INCLINATION = 51.6413
RA_OF_ASC_NODE = 289.5820
ARG_OF_PERICENTER = 36.5102
MEAN_ANOMALY = 323.6298

EPHEMERIS_TYPE = 0
CLASSIFICATION_TYPE = U
NORAD_CAT_ID = 25544
ELEMENT_SET_NO = 999
REV_AT_EPOCH = 43210
BSTAR = 0.000035000
MEAN_MOTION_DOT = 0.00001200
MEAN_MOTION_DDOT = 0.0
```

Note that OMM KVN does not use `META_START`/`META_STOP` markers — all keywords appear in a flat sequence.

---

## See Also

- [API Reference — OMM](../../library_api/ccsds/omm.md)
- [CCSDS Data Formats](index.md) — Overview of all message types
- [Two-Line Elements](../orbits/two_line_elements.md) — Traditional TLE format
