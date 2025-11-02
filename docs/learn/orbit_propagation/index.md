# Orbit Propagation

Orbit propagation is the process of computing a satellite's position and velocity over time from an initial state. Brahe provides a flexible propagation system built on a hierarchy of traits that define common functionality, allowing you to choose the right propagator for your application while maintaining a consistent interface.

## Propagation Traits

Brahe's propagation system is built on traits that define core functionality. This design allows different propagator implementations to share a common interface while providing specialized capabilities.

### OrbitPropagator Trait

The `OrbitPropagator` trait is the foundation for all propagator implementations. It defines the core interface for stepping through time, managing state, and controlling trajectory accumulation.

**Purpose**: Provides stepping operations, trajectory management, and state access for time-series propagation.

**Key Methods**:

**Stepping Operations**:

- `step()` - Advance by the default step size
- `step_by(step_size)` - Advance by a specified duration (seconds)
- `step_past(target_epoch)` - Step until past a target epoch
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

- `propagate_trajectory(epochs)` - Propagate to multiple epochs and store states
- `set_eviction_policy_max_size(n)` - Keep only N most recent states
- `set_eviction_policy_max_age(duration)` - Keep only states within time window (seconds)

**When to Use**: All propagators implement this trait. Use these methods for time-stepping through orbits, accumulating trajectory history, and managing memory for long-running applications.

### StateProvider Trait

The `StateProvider` trait extends propagators with the ability to compute states directly at arbitrary epochs without requiring time-stepping. This trait is designed for analytic propagators with closed-form solutions (like Keplerian and SGP4).

**Purpose**: Enables direct state computation at any epoch, in any reference frame, without building a trajectory.

**Key Methods**:

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

**When to Use**: Use this trait when you need to query satellite states at specific times without building a full trajectory, or when you need states in different reference frames. Particularly useful for:
- Computing states at irregular time intervals
- Ground track generation
- Access window computation
- Parallel batch state computation

**Requires**: Only available for analytic propagators (`KeplerianPropagator`, `SGPPropagator`).

### IdentifiableStateProvider Trait

The `IdentifiableStateProvider` trait combines `StateProvider` with `Identifiable`, enabling satellite tracking with both orbital state computation and identity management.

**Purpose**: Provides unified interface for access computation and multi-satellite scenarios where satellite identity must be tracked alongside state.

**Key Methods**:

Inherits all methods from:
- `StateProvider`: All state query methods
- `Identifiable`: `with_name()`, `with_id()`, `with_uuid()`, `get_name()`, `get_id()`, `get_uuid()`

**When to Use**: This trait is automatically implemented for any type that implements both `StateProvider` and `Identifiable`. Use in applications that need to track multiple satellites by name, ID, or UUID while computing their states:
- Ground station access computation
- Conjunction analysis
- Multi-satellite visualization
- Satellite catalog management

**Automatic Implementation**: You don't need to implement this trait explicitly—any propagator that implements `StateProvider` and `Identifiable` automatically gains this trait through a blanket implementation.

## Choosing a Propagator

Brahe provides two propagator implementations, each optimized for different use cases:

### KeplerianPropagator - Analytical Two-Body Propagation

**Use when**: You need fast, analytical propagation assuming only two-body dynamics (no perturbations).

**Features**:
- Analytical solution to Kepler's equations
- No perturbations (Earth oblateness, atmospheric drag, solar pressure, etc.)
- Extremely fast computation
- Perfect for short-term propagation or when perturbations are negligible
- Implements: `OrbitPropagator`, `StateProvider`, `Identifiable`

**Best for**:
- High-altitude orbits (GEO, MEO) where perturbations are small
- Short-term propagation (hours to days)
- Rapid trajectory generation for visualization
- Educational purposes and initial mission analysis
- Batch processing requiring maximum speed

**Initialization**: From Keplerian elements, ECI Cartesian state, or ECEF Cartesian state

### SGPPropagator - TLE-Based Propagation

**Use when**: Working with Two-Line Element (TLE) data or need simplified perturbation modeling.

**Features**:
- SGP4/SDP4 propagation model
- Includes simplified perturbations (Earth oblateness, atmospheric drag)
- Standard for operational satellite tracking
- Parses TLE data automatically
- Supports Classic and Alpha-5 TLE formats
- Implements: `OrbitPropagator`, `StateProvider`, `Identifiable`

**Best for**:
- Near-Earth satellites (LEO, MEO) where TLE data is available
- Operational satellite tracking and catalog maintenance
- Ground station pass prediction
- Short to medium-term propagation (days to weeks)
- When TLE data is the primary orbit source

**Limitations**:
- Accuracy degrades for highly eccentric or deep-space orbits
- Not suitable for high-precision applications
- Initial conditions cannot be changed (derived from TLE)

**Initialization**: From 2-line or 3-line TLE format

### Quick Decision Guide

```
Have TLE data?
├─ Yes → SGPPropagator
└─ No
   ├─ Need perturbations (LEO)?
   │  └─ Use SGPPropagator with generated TLE or consider external integrator
   └─ Two-body dynamics sufficient?
      └─ Yes → KeplerianPropagator (fastest)
```

## See Also

- [Keplerian Propagation](keplerian_propagation.md) - Analytical two-body propagator
- [SGP Propagation](sgp_propagation.md) - TLE-based SGP4/SDP4 propagator
- [Trajectories](../trajectories/index.md) - Trajectory storage and management
- [Frame Transformations](../frame_transformations.md) - ECI/ECEF conversions
- [API Reference](../../library_api/propagators/index.md) - Complete API documentation
