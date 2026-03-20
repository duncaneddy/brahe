# OPM — Orbit Parameter Message

An Orbit Parameter Message (OPM) carries a single spacecraft state at one epoch — position, velocity, and optionally Keplerian elements, spacecraft parameters, maneuvers, and covariance. It is the standard format for handing off initial conditions for propagation or documenting a maneuver plan.

## Parse and Initialize a Propagator

Extract position, velocity, epoch, and spacecraft parameters from an OPM to initialize a `NumericalOrbitPropagator`:

=== "Python"
    ``` python
    --8<-- "./examples/ccsds/opm_init_propagator.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/ccsds/opm_init_propagator.rs:4"
    ```

## Accessing OPM Data

Parse from file or string, then access the state vector, optional Keplerian elements, spacecraft parameters, covariance, and maneuvers:

=== "Python"
    ``` python
    --8<-- "./examples/ccsds/opm_parse_access.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/ccsds/opm_parse_access.rs:4"
    ```

## What an OPM Contains

Every OPM has a **header** (version, creation date, originator), **metadata** (object identity, center body, reference frame, time system), and a **state vector** (epoch plus position and velocity). Beyond these required parts, four optional sections can be present.

**Keplerian elements** duplicate the state vector information in orbital-element form — semi-major axis, eccentricity, inclination, RAAN, argument of pericenter, and true or mean anomaly, plus $GM$. The redundancy is intentional: elements are easier for humans to review at a glance, and some receiving systems prefer them as input.

**Spacecraft parameters** record physical properties relevant to force modeling — mass, drag area and coefficient ($C_D$), and solar radiation pressure area and coefficient ($C_R$). These feed directly into atmospheric drag and SRP force models during numerical propagation.

**Maneuvers** describe planned or executed burns. Each maneuver specifies an ignition epoch, duration, delta-mass, reference frame, and three delta-V components. Multiple maneuvers are allowed, and the reference frame can differ between them (e.g., RTN for in-plane burns, EME2000 for inertial targeting).

**Covariance** provides a 6$\times$6 symmetric position-velocity covariance matrix with an optional reference frame override relative to the state vector frame.

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
