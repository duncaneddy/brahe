# Numerical Propagation

Numerical propagation solves equations of motion through numerical integration, enabling high-fidelity dynamics modeling with arbitrary force models. Unlike analytical propagators that use closed-form solutions, numerical propagators step through time, computing accelerations at each step and integrating to get the next state. The 

Brahe provides two numerical propagator types:

- **`NumericalOrbitPropagator`** - Specialized for orbital mechanics with built-in force models (gravity, drag, SRP, third-body)

Both propagators implement the same trait hierarchy as analytical propagators (`OrbitPropagator`, `StateProvider`, `IdentifiableStateProvider`), providing consistent APIs for stepping, state queries, and trajectory management.

## Architecture Overview

The numerical propagation system consists of several configurable components:

### Force Model Configuration

The `ForceModelConfig` (Python) / `ForceModelConfiguration` (Rust) specifies which physical forces to include:

- **Gravity**: Point-mass or spherical harmonics (EGM2008 up to degree/order 100)
- **Atmospheric Drag**: Harris-Priester or exponential models
- **Solar Radiation Pressure**: Cannonball model with conical or cylindrical eclipse
- **Third-Body**: Sun and Moon gravitational perturbations
- **Relativistic Effects**: General relativistic corrections

### Integrator Configuration

The `NumericalPropagationConfig` specifies the integration method and tolerances:

- **Fixed-Step Methods**: RK4 (4th-order Runge-Kutta)
- **Adaptive Methods**: DP54, RKF45, RKN1210 with automatic step-size control
- **Tolerances**: Absolute and relative error tolerances for adaptive methods

### Event Detection

The propagator supports event detection during integration:

- **Time Events**: Trigger at specific epochs
- **Value Events**: Trigger when a computed quantity crosses a threshold
- **Binary Events**: Trigger on boolean state changes
- **Altitude Events**: Trigger at specific altitudes

Events can be configured to execute callbacks for impulsive maneuvers.

### Trajectory Storage

All propagators maintain an internal `OrbitTrajectory` storing propagated states. The trajectory supports:

- Hermite interpolation for smooth state queries between steps
- Eviction policies to limit memory usage
- Frame and representation conversions

## Quick Start

The simplest way to create a numerical propagator uses default configurations:

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/basic_propagation.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/basic_propagation.rs:4"
    ```

## When to Use Numerical Propagation

Numerical propagation is appropriate when:

- **Perturbations matter**: LEO satellites experience significant drag and J2 effects
- **High precision is required**: Conjunction analysis, maneuver planning, state estimation
- **Custom dynamics are needed**: Continuous thrust, attitude-dependent forces, relative motion
- **Event detection is required**: Finding specific orbital events like eclipse entry/exit

For rapid trajectory generation where perturbations are negligible, analytical propagators ([Keplerian](../keplerian_propagation.md) or [SGP4](../sgp_propagation.md)) are faster.

---

## Section Contents

- [Numerical Orbit Propagator](basic_propagation.md) - Getting started with numerical propagation
- [Force Models](force_models.md) - Configuring physical force models
- [Integrator Configuration](integrator_configuration.md) - Choosing integration methods
- [Event Detection](event_detection.md) - Detecting orbital events
- [Premade Events](premade_events.md) - Built-in event types
- [Event Callbacks](event_callbacks.md) - Responding to detected events
- [Maneuvers](maneuvers.md) - Impulsive and continuous thrust
- [Covariance and Sensitivity](covariance_sensitivity.md) - Uncertainty propagation
- [Extending Spacecraft State](extending_state.md) - Additional state variables
- [General Dynamics Propagation](../../generic_dynamics.md) - Propagating arbitrary ODEs

## See Also

- [NumericalOrbitPropagator API Reference](../../../library_api/propagators/numerical_orbit_propagator.md)
- [NumericalPropagator API Reference](../../../library_api/propagators/numerical_propagator.md)
