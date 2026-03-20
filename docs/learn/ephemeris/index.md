# Ephemeris Data Sources

Brahe queries satellite orbital data from two public sources -- CelesTrak and Space-Track --
and returns it as `GPRecord` objects that convert directly into SGP4 propagators. For most
operational satellite tracking workflows, this is the primary entry point: fetch current
elements for a satellite, build a propagator, and compute states.

## The Core Workflow

The most common usage pattern is three conceptual steps: **query** a data source for a
satellite by catalog number or name, **receive** a `GPRecord` containing the latest orbital
elements, and **propagate** to compute position and velocity at any future time. Brahe
provides a convenience method that collapses these steps into a single call:

=== "Python"
    ``` python
    --8<-- "./examples/datasets/celestrak_as_propagator.py:11"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/datasets/celestrak_as_propagator.rs:9"
    ```

Behind the scenes, `get_sgp_propagator` queries the data source, deserializes the response
into a `GPRecord`, extracts the TLE lines, and initializes an `SGPPropagator`. The returned
propagator is ready for immediate use with `propagate_to()` and `current_state()`.

## What is a GPRecord?

A General Perturbations (GP) record is the standard data format for satellite orbital
elements distributed by the US Space Surveillance Network. It is formally defined as an
Orbit Mean-Elements Message (OMM) under the CCSDS standard. Each record contains
approximately 40 fields spanning three categories: **identifiers** (NORAD catalog number,
international designator, object name), **orbital elements** (mean motion, eccentricity,
inclination, RAAN, argument of perigee, mean anomaly, $B^*$ drag term), and **metadata**
(epoch, classification, originator, reference frame). When TLE lines are available, they
are included as well.

Brahe's `GPRecord` struct handles a practical complication transparently: SpaceTrack returns
all JSON values as strings (e.g., `"NORAD_CAT_ID": "25544"`), while CelesTrak returns
numeric fields as native JSON numbers (e.g., `"NORAD_CAT_ID": 25544`). Custom deserializers
accept both formats, so downstream code works identically regardless of which source
produced the data. For the complete field listing, see the
[GPRecord API Reference](../../library_api/ephemeris/shared_types.md).

## How the Pieces Connect

`GPRecord` serves as the bridge type in Brahe's ephemeris architecture. Both
`CelestrakClient` and `SpaceTrackClient` return `GPRecord` from their GP query methods,
making it possible to write source-agnostic code. From a `GPRecord`, you can:

- **Build an SGP4 propagator** using the embedded TLE data, which is the primary use case
  for operational orbit prediction.
- **Export to CCSDS OMM format** for standards-compliant data exchange.
- **Access individual fields** for filtering, cataloging, or display purposes.

This design means that switching between CelesTrak and SpaceTrack requires changing only
the client instantiation and query call -- all downstream propagation and analysis code
remains unchanged.

## Choosing a Data Source

**CelesTrak** is the simplest starting point. It requires no account or authentication,
provides pre-built satellite groups (e.g., active satellites, GPS, Starlink), and supports
lookup by catalog number, name, or international designator. It is well-suited for quick
prototyping, educational use, and applications that need common satellite constellations.

**Space-Track** is the authoritative source for the US satellite catalog. It requires a
free account and provides full server-side query filtering on any GP field, access to the
complete historical catalog, supplemental data products (SP ephemeris, file shares), and
SATCAT metadata. It is the appropriate choice when you need comprehensive catalog access
or precise control over query results.

---

## See Also

- [CelesTrak](celestrak.md) -- Using the CelesTrak client
- [Space-Track](spacetrack/index.md) -- Using the Space-Track client
- [Ephemeris API Reference](../../library_api/ephemeris/index.md) -- Complete function and type documentation
- [Two-Line Elements](../orbits/two_line_elements.md) -- TLE and 3LE format details
- [SGP Propagation](../orbit_propagation/sgp_propagation.md) -- SGP4/SDP4 propagation theory and usage
