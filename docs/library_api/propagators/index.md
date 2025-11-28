# Propagators

**Module**: `brahe.propagators`

Orbit propagators for predicting satellite positions over time.

## Analytical Propagators

- **[KeplerianPropagator](keplerian_propagator.md)** - Analytical two-body orbit propagator
- **[SGPPropagator](sgp_propagator.md)** - SGP4/SDP4 orbit propagator for TLE data

## Numerical Propagators

- **[NumericalOrbitPropagator](numerical_orbit_propagator.md)** - High-fidelity numerical orbit propagator with force models
- **[NumericalPropagator](numerical_propagator.md)** - Generic numerical propagator for arbitrary dynamics

## Configuration

- **[NumericalPropagationConfig](numerical_propagation_config.md)** - Integration method and tolerance configuration
- **[ForceModelConfig](force_model_config.md)** - Force model configuration for numerical propagation

## See Also

- **[Event Detection](../events/index.md)** - Event detection system for numerical propagators
