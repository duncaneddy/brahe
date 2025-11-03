# SGP Propagation

The `SGPPropagator` implements the SGP4/SDP4 propagation models for orbital prediction. SGP4 is a standard method for satellite tracking and includes simplified perturbations from Earth oblateness and atmospheric drag, making it suitable for operational satellite tracking and near-Earth orbit propagation. It is widely used with Two-Line Element (TLE) data provided by NORAD and other space tracking organizations.

For complete API documentation, see the [SGPPropagator API Reference](../../library_api/propagators/sgp_propagator.md).

## TLE Format Support

SGP4 propagation is based on Two-Line Element (TLE) sets, a compact data format for orbital elements. Brahe supports both traditional and modern TLE formats:

- **Classic Format**: Traditional numeric NORAD catalog numbers (5 digits, up to 99999)
- **Alpha-5 Format**: Extended alphanumeric catalog numbers for satellites beyond 99999

The initialization automatically detects and handles both formats.

## Initialization

The `SGPPropagator` is initialized from TLE data. The TLE lines contain all orbital parameters needed for propagation.

### From Two Line Elements (TLE)

The most common initialization uses two lines of TLE data.

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/sgp_propagation/from_tle.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/sgp_propagation/from_tle.rs:4"
    ```

### From 3-Line Elements (3LE)

Three-line TLE format includes an optional satellite name on the first line.

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/sgp_propagation/from_3le.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/sgp_propagation/from_3le.rs:4"
    ```

### Configuring Output Format

By default, SGP4 outputs states in ECI Cartesian coordinates. Use `with_output_format()` to configure the output frame and representation.

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/sgp_propagation/output_format.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/sgp_propagation/output_format.rs:4"
    ```

## Stepping Through Time

The SGP propagator uses the same stepping interface as other propagators through the `OrbitPropagator` trait.

### Single and Multiple Steps

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/sgp_propagation/stepping.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/sgp_propagation/stepping.rs:4"
    ```

### Propagate to Target Epoch

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/sgp_propagation/propagate_to_epoch.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/sgp_propagation/propagate_to_epoch.rs:4"
    ```

## Direct State Queries

The SGP propagator implements the `StateProvider` trait, allowing direct state computation at arbitrary epochs without stepping. Because SGP4 uses closed-form solutions, state queries are efficient and do not require building a trajectory.

### Single Epoch Queries

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/sgp_propagation/single_epoch_query.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/sgp_propagation/single_epoch_query.rs:4"
    ```

### Batch Queries

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/sgp_propagation/batch_queries.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/sgp_propagation/batch_queries.rs:4"
    ```

### Special: PEF Frame

SGP4 natively outputs states in the TEME (True Equator Mean Equinox) frame. For specialized applications, you can access states in the intermediate PEF (Pseudo-Earth-Fixed) frame:

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/sgp_propagation/pef_frame.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/sgp_propagation/pef_frame.rs:4"
    ```

## Extracting Orbital Elements from TLE

The propagator can extract Keplerian orbital elements directly from the TLE data:

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/sgp_propagation/get_elements.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/sgp_propagation/get_elements.rs:4"
    ```

## Trajectory Management

SGP propagators support the same trajectory management as Keplerian propagators, including frame conversions and memory management.

### Memory Management

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/sgp_propagation/memory_management.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/sgp_propagation/memory_management.rs:4"
    ```

## Limitations and Considerations

### Immutable Initial Conditions

Unlike the Keplerian propagator, SGP4 initial conditions are derived from the TLE and **cannot be changed**. Attempting to call `set_initial_conditions()` will result in a panic:

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    prop = bh.SGPPropagator.from_tle(line1, line2, 60.0)

    # This will raise an error - SGP initial conditions come from TLE
    # prop.set_initial_conditions(...)  # Don't do this!

    # To use different orbital elements, create a KeplerianPropagator instead
    ```

=== "Rust"

    ```rust
    // This will panic - SGP initial conditions come from TLE
    // prop.set_initial_conditions(...);  // Don't do this!

    // To use different orbital elements, create a KeplerianPropagator instead
    ```

## Identity Tracking

Like Keplerian propagators, SGP propagators support identity tracking:

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/sgp_propagation/identity_tracking.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/sgp_propagation/identity_tracking.rs:4"
    ```

## See Also

- [Orbit Propagation Overview](index.md) - Propagation concepts and trait hierarchy
- [Keplerian Propagation](keplerian_propagation.md) - Analytical two-body propagator
- [Trajectories](../trajectories/index.md) - Trajectory storage and operations
- [Two-Line Elements](../orbits/two_line_elements.md) - Working with TLE data
- [SGPPropagator API Reference](../../library_api/propagators/sgp_propagator.md)
