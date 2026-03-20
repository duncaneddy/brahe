# OEM-Based Access Prediction

This example demonstrates an end-to-end workflow for computing ground station access windows using a CCSDS Orbit Ephemeris Message (OEM). We'll generate an artificial OEM from an ISS-like orbit, save and reload it from disk, convert it to an `OrbitTrajectory`, and compute access windows against three ground stations. Finally, we compare the OEM-based results against direct propagator results to validate fidelity.

---

## Setup

Import the necessary libraries and initialize Earth orientation parameters.

``` python
--8<-- "./examples/examples/oem_access_prediction.py:preamble"
```

## Generate OEM

Define an ISS-like orbit (~408 km altitude, 51.6° inclination), propagate it for 24 hours using a Keplerian propagator, and build an OEM with the Cartesian ECI states. Since OEM files store position and velocity data, we sample the propagator's `state_eci()` method to get Cartesian states regardless of its internal representation.

``` python
--8<-- "./examples/examples/oem_access_prediction.py:generate_oem"
```

## Save and Load OEM

Write the OEM to disk in CCSDS KVN format, then load it back to demonstrate the round-trip file I/O workflow.

``` python
--8<-- "./examples/examples/oem_access_prediction.py:write_read_oem"
```

## Create OrbitTrajectory

Convert the loaded OEM segment into an `OrbitTrajectory` object, which can be used directly with Brahe's access computation pipeline.

``` python
--8<-- "./examples/examples/oem_access_prediction.py:create_trajectory"
```

## Define Ground Stations

Create `PointLocation` objects for San Francisco, New York, and London.

``` python
--8<-- "./examples/examples/oem_access_prediction.py:ground_stations"
```

## Compute Access Windows from OEM

Pass the OEM-loaded trajectory directly to `location_accesses()` with a 10° minimum elevation constraint. This is the key step — `OrbitTrajectory` objects work as state providers for access computation, just like propagators.

``` python
--8<-- "./examples/examples/oem_access_prediction.py:compute_accesses_oem"
```

## Compute Reference Windows

For validation, compute access windows using the original Keplerian propagator over the same time window and with the same constraint.

``` python
--8<-- "./examples/examples/oem_access_prediction.py:compute_accesses_direct"
```

## Compare Results

Compare each OEM-based access window against the corresponding direct propagator window. With 60-second OEM spacing and Lagrange interpolation, we expect sub-5-second timing agreement and sub-0.5° elevation agreement.

``` python
--8<-- "./examples/examples/oem_access_prediction.py:compare_results"
```

## Display Results

Print all OEM-based access windows grouped by station with summary statistics.

``` python
--8<-- "./examples/examples/oem_access_prediction.py:display_results"
```

## Full Code Example

```python title="oem_access_prediction.py"
--8<-- "./examples/examples/oem_access_prediction.py:all"
```

## See Also

- [Access Computation](../learn/access_computation/index.md)
- [CCSDS OEM](../learn/ccsds/oem.md)
- [Predicting Ground Contacts](ground_contacts.md)
