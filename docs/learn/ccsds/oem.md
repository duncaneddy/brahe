# OEM — Orbit Ephemeris Message

An Orbit Ephemeris Message (OEM) carries time-ordered state vectors for spacecraft ephemeris exchange. The typical workflow is to parse an OEM file and convert it into an `OrbitTrajectory` for interpolation and analysis, or to generate an OEM from a propagator for distribution.

## Parse and Access

Parse from file or string, then access header properties, segment metadata, and state vectors:

=== "Python"
    ``` python
    --8<-- "./examples/ccsds/oem_parse_access.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/ccsds/oem_parse_access.rs:4"
    ```

## Converting to OrbitTrajectory

The primary interoperability point for OEM data is conversion to brahe's `OrbitTrajectory`. Each OEM segment maps to a trajectory object, giving you Hermite interpolation at arbitrary epochs within the covered time span:

=== "Python"
    ``` python
    --8<-- "./examples/ccsds/oem_to_trajectory.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/ccsds/oem_to_trajectory.rs:4"
    ```

## How OEM Messages Are Organized

An OEM message begins with a **header** that records the format version, creation date, and originator. The bulk of the data lives in one or more **segments**, each of which has its own metadata block and a sequence of state vectors.

Multiple segments exist because a single file may need to cover different trajectory arcs. A maneuver boundary, a change in reference frame, or a gap in tracking data each warrant a new segment. Within a segment, the metadata block records the object identity, center body, reference frame, time system, time span, and interpolation settings. The state vectors follow — each line provides an epoch plus position and velocity (and optionally acceleration). If covariance data is available, it appears as one or more 6$\times$6 symmetric matrices attached to the segment, each with its own epoch and optional reference frame override.

## Creating and Writing OEMs

Build an OEM programmatically by defining a header, adding segments with metadata, and populating state vectors. The resulting message can be serialized to KVN, XML, or JSON:

=== "Python"
    ``` python
    --8<-- "./examples/ccsds/oem_create_write.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/ccsds/oem_create_write.rs:4"
    ```

!!! info "Round-Trip Fidelity"
    Writing and re-parsing an OEM preserves all metadata, state vectors, and covariance data. Numeric precision may vary slightly due to floating-point formatting, but values are preserved within the precision of the output format.

## Generating from a Propagator

Propagate an orbit numerically, extract the trajectory, and build an OEM for distribution:

=== "Python"
    ``` python
    --8<-- "./examples/ccsds/oem_from_propagator.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/ccsds/oem_from_propagator.rs:4"
    ```

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
