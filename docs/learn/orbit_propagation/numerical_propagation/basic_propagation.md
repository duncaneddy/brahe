# Numerical Orbit Propagator

This guide covers the fundamental operations of the `NumericalOrbitPropagator`: creating propagators, stepping through time, accessing states, and managing trajectories.

For API details, see the [NumericalOrbitPropagator API Reference](../../../library_api/propagators/numerical_orbit_propagator.md).

## Creating a Propagator

The `NumericalOrbitPropagator` requires an initial epoch, state, propagation configuration, force model configuration, and optional parameters.

### Minimal Setup

The simplest setup uses default configurations:

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/basic_propagation.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/basic_propagation.rs:4"
    ```

### With Trajectory Storage

To store intermediate states during propagation, enable trajectory recording:

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/propagation_with_trajectory.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/propagation_with_trajectory.rs:4"
    ```

## Stepping Through Time

The propagator provides several methods for advancing through time, following the same interface as analytical propagators.

### Single Steps

- `step()` - Advance by the integrator's current step size
- `step_by(dt)` - Advance by a specific duration (seconds)
- `step_past(epoch)` - Step until past a target epoch

### Multiple Steps

- `propagate_steps(n)` - Take N steps
- `propagate_to(epoch)` - Propagate exactly to a target epoch

The `propagate_to()` method is the most commonly used, as it handles step-size adjustment to reach the exact target epoch.

## Accessing State

### Current State

After propagation, access the current state using:

- `current_epoch()` - Returns the propagator's current epoch
- `current_state()` - Returns the current state vector (Cartesian ECI)

### State at Arbitrary Epochs

The `StateProvider` trait enables state queries at any epoch:

- `state(epoch)` - State in the propagator's native format
- `state_eci(epoch)` - Cartesian state in ECI frame
- `state_ecef(epoch)` - Cartesian state in ECEF frame
- `state_koe(epoch, angle_format)` - Keplerian orbital elements

For epochs within the propagated trajectory, interpolation is used. For epochs beyond the trajectory, the propagator advances to that epoch.

### Batch Queries

For multiple epochs, use the batch query methods:

- `states(epochs)` - States at multiple epochs
- `states_eci(epochs)` - ECI states at multiple epochs
- `states_koe(epochs, angle_format)` - Keplerian elements at multiple epochs

## Trajectory Management

The propagator maintains an internal `OrbitTrajectory` containing all propagated states.

### Accessing the Trajectory

Access the trajectory directly via the `trajectory` property. The trajectory provides:

- `len()` - Number of stored states
- `epochs()` - List of all epoch times
- `states()` - Array of all state vectors
- `state_at_epoch(epoch)` - Interpolated state at any epoch within the trajectory

### Memory Management

For long propagations, use eviction policies to limit memory:

- `set_eviction_policy_max_size(n)` - Keep only the N most recent states
- `set_eviction_policy_max_age(duration)` - Keep only states within a time window

### Resetting

Use `reset()` to return the propagator to its initial conditions, clearing the trajectory.

## Propagator Parameters

Some force models require additional parameters. These are provided as a parameter vector during construction:

| Index | Parameter | Units | Description |
|-------|-----------|-------|-------------|
| 0 | mass | kg | Spacecraft mass |
| 1 | drag_area | m$^2$ | Cross-sectional area for drag |
| 2 | Cd | - | Drag coefficient |
| 3 | srp_area | m$^2$ | Cross-sectional area for SRP |
| 4 | Cr | - | Reflectivity coefficient |

The `ForceModelConfig.requires_params()` method indicates whether parameters are needed.

## Identity Tracking

For multi-satellite applications, propagators can be identified by name, ID, or UUID:

```python
prop = bh.NumericalOrbitPropagator(...)
prop = prop.with_name("ISS")
prop = prop.with_id(25544)
```

This enables tracking propagators in access computation, conjunction analysis, and other multi-object scenarios.

---

## See Also

- [Numerical Propagation Overview](index.md) - Architecture and concepts
- [Force Models](force_models.md) - Configuring force models
- [Integrator Configuration](integrator_configuration.md) - Integration method selection
- [NumericalOrbitPropagator API Reference](../../../library_api/propagators/numerical_orbit_propagator.md)
