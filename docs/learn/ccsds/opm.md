# OPM — Orbit Parameter Message

An Orbit Parameter Message (OPM) contains a single spacecraft state (position and velocity) at a specific epoch, optionally accompanied by Keplerian elements, spacecraft physical parameters, maneuver specifications, and covariance data. OPM is used when exchanging a complete snapshot of an object's orbital state — for example, as initial conditions for propagation, in maneuver planning documents, or as part of a state handoff between organizations.

## Structure

An OPM message consists of:

- **Header** — format version, creation date, originator, comments
- **Metadata** — object name/ID, center body, reference frame, time system
- **State vector** — epoch + position [X, Y, Z] + velocity [X_DOT, Y_DOT, Z_DOT]
- **Keplerian elements** (optional) — semi-major axis, eccentricity, inclination, RAAN, argument of pericenter, true/mean anomaly, GM
- **Spacecraft parameters** (optional) — mass, drag area/coefficient, SRP area/coefficient
- **Covariance** (optional) — 6$\times$6 symmetric covariance matrix with reference frame
- **Maneuvers** (optional, multiple) — ignition epoch, duration, delta-mass, reference frame, delta-V components
- **User-defined parameters** (optional) — arbitrary key-value pairs

## Parsing

```python
from brahe.ccsds import OPM

# From file (KVN, XML, or JSON — auto-detected)
opm = OPM.from_file("state.opm")

# From string
opm = OPM.from_str(kvn_content)
```

## Accessing Data

### Metadata

```python
opm.format_version()   # 3.0
opm.object_name()      # "GODZILLA 5"
opm.object_id()        # "1998-999A"
opm.center_name()      # "EARTH"
opm.ref_frame()        # "ITRF2000"
opm.time_system()      # "UTC"
```

### State Vector

Position and velocity are returned in SI units (meters, m/s):

```python
pos = opm.position()   # [6503514.0, 1239647.0, -717490.0]  (meters)
vel = opm.velocity()   # [-873.16, 8740.42, -4191.076]      (m/s)
```

### Keplerian Elements

```python
opm.has_keplerian_elements()  # True/False

opm.semi_major_axis()         # 41399512.3 (meters, converted from km)
```

Full Keplerian element access is available through `to_dict()`:

```python
d = opm.to_dict()
kep = d["keplerian_elements"]
# kep = {
#     "semi_major_axis": 41399512.3,     # meters
#     "eccentricity": 0.020842611,
#     "inclination": 0.117746,           # degrees
#     "ra_of_asc_node": 17.604721,       # degrees
#     "arg_of_pericenter": 218.242943,   # degrees
#     "true_anomaly": 41.922339,         # degrees
#     "mean_anomaly": None,
#     "gm": 3.986004415e+14              # m³/s²
# }
```

### Spacecraft Parameters

```python
opm.mass()  # 3000.0 (kg), or None
```

### Maneuvers

```python
opm.num_maneuvers()    # 2

m = opm.maneuver(0)
# m = {
#     "epoch_ignition": "2000-06-03T09:00:34.1",
#     "duration": 132.6,           # seconds
#     "delta_mass": -18.418,       # kg (negative = mass decrease)
#     "ref_frame": "J2000",
#     "dv": [-23.257, 16.8316, -8.93444]  # m/s
# }
```

Each maneuver specifies its own reference frame for the delta-V components. Common frames include J2000/EME2000 (inertial) and RTN (orbit-relative radial-transverse-normal).

### Full Serialization

```python
d = opm.to_dict()
# d = {
#     "header": { ... },
#     "metadata": { "object_name": "...", "ref_frame": "...", ... },
#     "state_vector": { "epoch": "...", "position": [...], "velocity": [...] },
#     "keplerian_elements": { ... },       # if present
#     "spacecraft_parameters": { ... },    # if present
#     "maneuvers": [ { ... }, ... ],       # if present
#     "user_defined": { "KEY": "value" }   # if present
# }
```

## KVN Format Example

An OPM KVN file with a state vector, Keplerian elements, and a maneuver:

```text
CCSDS_OPM_VERS = 3.0
CREATION_DATE = 2024-01-15T00:00:00
ORIGINATOR = EXAMPLE

OBJECT_NAME = MY SATELLITE
OBJECT_ID = 2024-001A
CENTER_NAME = EARTH
REF_FRAME = EME2000
TIME_SYSTEM = UTC

EPOCH = 2024-01-15T00:00:00
X = 6878.137 [km]
Y = 0.000 [km]
Z = 0.000 [km]
X_DOT = 0.000 [km/s]
Y_DOT = 7.612 [km/s]
Z_DOT = 0.000 [km/s]

SEMI_MAJOR_AXIS = 6878.137 [km]
ECCENTRICITY = 0.001
INCLINATION = 0.0 [deg]
RA_OF_ASC_NODE = 0.0 [deg]
ARG_OF_PERICENTER = 0.0 [deg]
TRUE_ANOMALY = 0.0 [deg]
GM = 398600.4415 [km**3/s**2]

MAN_EPOCH_IGNITION = 2024-01-15T01:00:00
MAN_DURATION = 60.0 [s]
MAN_DELTA_MASS = -5.0 [kg]
MAN_REF_FRAME = RTN
MAN_DV_1 = 0.010 [km/s]
MAN_DV_2 = 0.000 [km/s]
MAN_DV_3 = 0.000 [km/s]
```

Note the optional unit annotations in square brackets (`[km]`, `[deg]`). Brahe strips these during parsing.

---

## See Also

- [API Reference — OPM](../../library_api/ccsds/opm.md)
- [CCSDS Data Formats](index.md) — Overview of all message types
- [Keplerian Elements](../orbits/properties.md) — Orbital element definitions
