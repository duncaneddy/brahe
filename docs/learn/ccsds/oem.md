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

## Parsing and Accessing Data

Parse from file or string, then access header properties, segment metadata, and state vectors:

=== "Python"
    ``` python
    --8<-- "./examples/ccsds/oem_parse_access.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/ccsds/oem_parse_access.rs:4"
    ```

## Creating from Scratch

Build an OEM programmatically — define header, add segments with metadata, and populate state vectors:

=== "Python"
    ``` python
    --8<-- "./examples/ccsds/oem_create_write.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/ccsds/oem_create_write.rs:4"
    ```

## Append, Extend, and Delete (Python)

Python supports collection-style mutation on OEM segments and state vectors:

``` python
--8<-- "./examples/ccsds/oem_append_extend.py:8"
```

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

## Converting to OrbitTrajectory

Convert OEM segments to brahe `OrbitTrajectory` objects for interpolation and analysis:

=== "Python"
    ``` python
    --8<-- "./examples/ccsds/oem_to_trajectory.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/ccsds/oem_to_trajectory.rs:4"
    ```

## Writing

```python
# Write to KVN string
kvn = oem.to_string("KVN")

# Write to file (KVN, XML, or JSON)
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
