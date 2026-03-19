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

## Parsing and Accessing Data

Parse from file or string, then access state vector, Keplerian elements, spacecraft parameters, and maneuvers:

=== "Python"
    ``` python
    --8<-- "./examples/ccsds/opm_parse_access.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/ccsds/opm_parse_access.rs:4"
    ```

## Initializing a Propagator

Extract position, velocity, epoch, and spacecraft parameters from an OPM to initialize a `NumericalOrbitPropagator`:

=== "Python"
    ``` python
    --8<-- "./examples/ccsds/opm_init_propagator.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/ccsds/opm_init_propagator.rs:4"
    ```

## Maneuver Propagation

Read OPM maneuvers and apply them as impulsive delta-V events during propagation:

=== "Python"
    ``` python
    --8<-- "./examples/ccsds/opm_maneuver_propagation.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/ccsds/opm_maneuver_propagation.rs:4"
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
