# OMM — Orbit Mean-elements Message

An Orbit Mean-elements Message (OMM) is the CCSDS-standardized representation of TLE/GP data — the same orbital elements traditionally distributed as Two-Line Element sets, in a structured, self-describing format. Data sources like CelesTrak and Space-Track distribute GP data as OMM. The typical workflow is to parse an OMM and initialize an SGP4 propagator.

## Parse and Propagate with SGP4

Extract OMM mean elements and TLE parameters to create an `SGPPropagator`:

=== "Python"
    ``` python
    --8<-- "./examples/ccsds/omm_init_sgp.py:10"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/ccsds/omm_init_sgp.rs:5"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/ccsds/omm_init_sgp.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/ccsds/omm_init_sgp.rs.txt"
        ```

## Accessing Mean Elements and TLE Parameters

Parse from file or string, then access metadata, mean elements, and TLE parameters. The message carries two main data sections: **mean elements** (epoch, mean motion, eccentricity, inclination, RAAN, argument of pericenter, mean anomaly) and **TLE parameters** (NORAD catalog ID, classification, element set number, revolution count, $B^*$ drag term, mean motion derivatives):

=== "Python"
    ``` python
    --8<-- "./examples/ccsds/omm_parse_access.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/ccsds/omm_parse_access.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/ccsds/omm_parse_access.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/ccsds/omm_parse_access.rs.txt"
        ```

!!! info "Unit Convention for OMM"
    Mean motion, angles, and TLE drag terms are kept in their CCSDS/TLE-native units (rev/day, degrees, etc.) because these values are needed as-is for TLE generation and SGP4 initialization. Only GM is converted to SI (m$^3$/s$^2$).

## OMM and GPRecord

Brahe's `GPRecord` type — returned by both `CelestrakClient` and `SpaceTrackClient` when querying GP data — has a bidirectional relationship with OMM. A `GPRecord` can be converted to an OMM via `to_omm()` for CCSDS-compliant export, and an OMM can be converted to a `GPRecord` via `to_gp_record()` for use with brahe's ephemeris infrastructure.

This means you can move freely between the two representations: query CelesTrak for a satellite, get a `GPRecord`, and export it as a standards-compliant OMM file for distribution. Or parse an OMM file received from an external system and convert it to a `GPRecord` to use the same downstream code you would with a CelesTrak or Space-Track query. Both conversions preserve all shared fields, so switching between formats introduces no data loss.

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
- [Ephemeris Data Sources](../ephemeris/index.md) — CelesTrak and Space-Track clients
- [SGP Propagation](../orbit_propagation/sgp_propagation.md) — SGP4/SDP4 propagation theory and usage
