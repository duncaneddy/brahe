# Keplerian Propagation

The `KeplerianPropagator` provides fast, analytical two-body orbital propagation using Kepler's equations. It assumes only gravitational attraction from a central body (Earth) with no perturbations, making it ideal for rapid trajectory generation, high-altitude orbits, or when perturbations are negligible.

For complete API documentation, see the [KeplerianPropagator API Reference](../../library_api/propagators/keplerian_propagator.md).

## Initialization

The `KeplerianPropagator` can be initialized from several state representations.

### From Keplerian Elements

The most direct initialization method uses classical Keplerian orbital elements.

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/keplerian_propagation/init_from_elements.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/keplerian_propagation/init_from_elements.rs:4"
    ```

### From ECI Cartesian State

Initialize from position and velocity vectors in the Earth-Centered Inertial (ECI) frame.

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/keplerian_propagation/init_from_eci.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/keplerian_propagation/init_from_eci.rs:4"
    ```

### From ECEF Cartesian State

Initialize from position and velocity vectors in the Earth-Centered Earth-Fixed (ECEF) frame. The propagator will automatically convert to ECI internally.

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/keplerian_propagation/init_from_ecef.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/keplerian_propagation/init_from_ecef.rs:4"
    ```

## Stepping Through Time

One of the primary functions of propagators is to step forward in time, generating new states at regular intervals. There are several methods to advance the propagator's internal state. Each stepping operation adds new state(s) to the internal trajectory.

### Single Steps

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/keplerian_propagation/single_steps.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/keplerian_propagation/single_steps.rs:4"
    ```

### Multiple Steps

The `propagate_steps()` method allows taking multiple fixed-size steps in one call.

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/keplerian_propagation/multiple_steps.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/keplerian_propagation/multiple_steps.rs:4"
    ```

### Propagate to Target Epoch

For precise time targeting, use `propagate_to()` which adjusts the final step size to exactly reach the target epoch.

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/keplerian_propagation/propagate_to_epoch.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/keplerian_propagation/propagate_to_epoch.rs:4"
    ```

## Direct State Queries

The `StateProvider` trait allows computing states at arbitrary epochs without building a trajectory. This is useful for sparse sampling or parallel batch computation.

### Single Epoch Queries

Single epoch queries like `state()`, `state_eci()`, and `state_ecef()` compute the state at a specific epoch on demand.

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/keplerian_propagation/state_query_single.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/keplerian_propagation/state_query_single.rs:4"
    ```

### Batch Queries

Batch queries like `states()` and `states_eci()` compute states at each epoch in a provided list.

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/keplerian_propagation/state_query_batch.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/keplerian_propagation/state_query_batch.rs:4"
    ```

## Trajectory Management

The propagator stores all stepped states in an internal `OrbitTrajectory`. This trajectory can be accessed, converted, and managed.

### Accessing the Trajectory

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/keplerian_propagation/trajectory_access.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/keplerian_propagation/trajectory_access.rs:4"
    ```

### Frame Conversions

You can use the OrbitTrajectory's frame conversion methods to get the trajectory in different reference frames.

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/keplerian_propagation/trajectory_conversions.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/keplerian_propagation/trajectory_conversions.rs:4"
    ```

### Memory Management

Propagators support trajectory memory management via eviction policies to limit memory usage for long-running applications.

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/keplerian_propagation/trajectory_memory.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/keplerian_propagation/trajectory_memory.rs:4"
    ```

## Configuration and Control

There are several methods to manage and configure the propagator during its lifecycle.

### Resetting the Propagator

You can reset the propagator to its initial conditions using the `reset()` method.

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/keplerian_propagation/reset_propagator.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/keplerian_propagation/reset_propagator.rs:4"
    ```

### Changing Step Size

If you need to adjust the default step size during propagation, use the `set_step_size()` method.

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/keplerian_propagation/change_step_size.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/keplerian_propagation/change_step_size.rs:4"
    ```

## Identity Tracking

Finally, the `IdentifiableStateProvider` trait allows you to set and get identity information for the propagator. This can be useful when managing multiple propagators in an application.

Track propagators with names, IDs, or UUIDs for multi-satellite scenarios.

=== "Python"

    ``` python
    --8<-- "./examples/orbit_propagation/keplerian_propagation/identity_tracking.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbit_propagation/keplerian_propagation/identity_tracking.rs:4"
    ```

---

## See Also

- [Orbit Propagation Overview](index.md) - Propagation concepts and trait hierarchy
- [SGP Propagation](sgp_propagation.md) - TLE-based SGP4/SDP4 propagator
- [Trajectories](../trajectories/index.md) - Trajectory storage and operations
- [KeplerianPropagator API Reference](../../library_api/propagators/keplerian_propagator.md)
