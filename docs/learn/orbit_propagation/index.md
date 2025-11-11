# Orbit Propagation

Orbit propagation is the process of computing a satellite's trajectory over time from an initial state. Brahe's propagation system is built on a set of [Rust traits](https://doc.rust-lang.org/book/ch10-02-traits.html) that define common functionality. This design allows for common usage patterns making it easy to switch between different propagator implementations.

All propagators in Brahe implement the `OrbitPropagator` trait, which provides methods for stepping through time, managing trajectory history, and accessing the current state. All propagators store their state history in an `OrbitTrajectory`, which implements the `Trajectory` and `Interpolatable` traits for state storage and interpolation. See the [Trajectory](../trajectories/index.md) documentation for more details on these traits and the methods they provide.

There is also the `StateProvider` trait, which extends propagators with the ability to compute states directly at arbitrary epochs. How this is implemented depends on the specific propagator. Analytic propagators like `KeplerianPropagator` and `SGPPropagator` can compute states at any time using closed-form solutions, while numerical propagators typically require time-stepping to the desired epoch, then interpolating the result.

The `IdentifiableStateProvider` trait combines `StateProvider` with `Identifiable`, enabling identification of Propagators by name, ID, or UUID. This is useful for tracking multiple satellites in applications like ground station access computation or conjunction analysis.

### OrbitPropagator Trait

The `OrbitPropagator` trait is the foundation for all propagator implementations. It defines the core interface for stepping through time, managing state, and controlling trajectory accumulation.

**Stepping Operations**:

- `step()` - Advance by the default step size
- `step_by(step_size)` - Advance by a specified duration (seconds)
- `step_past(target_epoch)` - Step until the given Epoch is passed
- `propagate_steps(n)` - Take N steps of default step size
- `propagate_to(target_epoch)` - Propagate precisely to a target epoch

**State Access**:

- `current_epoch()` - Get the most recent propagated epoch
- `current_state()` - Get the most recent propagated state
- `initial_epoch()` - Get the initial epoch
- `initial_state()` - Get the initial state

**Configuration**:

- `step_size()` - Get the default step size (seconds)
- `set_step_size(step_size)` - Set the default step size (seconds)
- `reset()` - Reset propagator to initial conditions
- `set_initial_conditions(epoch, state, frame, representation, angle_format)` - Update initial conditions

**Trajectory Management**:

- `propagate_trajectory(epochs)` - Propagate to multiple epochs with `propagate_to()` calls for each provided epoch
- `set_eviction_policy_max_size(n)` - Keep only N most recent states
- `set_eviction_policy_max_age(duration)` - Keep only states within time window (seconds)

### StateProvider Trait

The `StateProvider` trait extends propagators with methods to get the state at arbitrary epochs. For analytic propagators, this is done using closed-form solutions which immediately compute the state without time-stepping. Numerical propagators typically step to the desired epoch and interpolate the result. This trait provides both single-epoch and multi-epoch (batch) query methods.

**Single Epoch Queries**:

- `state(epoch)` - Get state in propagator's native format
- `state_eci(epoch)` - Get Cartesian state in ECI frame
- `state_ecef(epoch)` - Get Cartesian state in ECEF frame
- `state_as_osculating_elements(epoch, angle_format)` - Get Keplerian elements

**Multi-Epoch Queries** (Batch Operations):

- `states(epochs)` - Get states at multiple epochs in native format
- `states_eci(epochs)` - Get ECI Cartesian states at multiple epochs
- `states_ecef(epochs)` - Get ECEF Cartesian states at multiple epochs
- `states_as_osculating_elements(epochs, angle_format)` - Get Keplerian elements at multiple epochs

### IdentifiableStateProvider Trait

The `IdentifiableStateProvider` trait combines `StateProvider` with `Identifiable`, identifying propagator objects by name, ID, or UUID. This trait inherits all methods from:
- `StateProvider`: All state query methods
- `Identifiable`: `with_name()`, `with_id()`, `with_uuid()`, `get_name()`, `get_id()`, `get_uuid()`

## Choosing a Propagator

Brahe currently provides two propagator implementations:
- `KeplerianPropagator`: An analytic two-body propagator using Keplerian orbital elements. Suitable for high-level mission design and long-term propagation where perturbations are negligible.
- `SGPPropagator`: An analytic propagator based on the SGP4/SDP4 models using TLE data. Suitable for tracking Earth-orbiting satellites with moderate accuracy.

---

## See Also

- [Keplerian Propagation](keplerian_propagation.md) - Analytical two-body propagator
- [SGP Propagation](sgp_propagation.md) - TLE-based SGP4/SDP4 propagator
- [Trajectories](../trajectories/index.md) - Trajectory storage and management
- [Frame Transformations](../frames/index.md) - ECI/ECEF conversions
- [API Reference](../../library_api/propagators/index.md) - Complete API documentation
