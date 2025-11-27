# NumericalPropagator

Generic numerical propagator for arbitrary dynamical systems. Unlike `NumericalOrbitPropagator` which has built-in orbital force models, `NumericalPropagator` accepts user-defined dynamics functions, making it suitable for attitude propagation, chemical kinetics, population models, or any ODE system.

!!! note
    For conceptual explanations and usage examples, see [Generic Dynamics](../../learn/generic_dynamics.md) in the User Guide.

## Class Reference

::: brahe.NumericalPropagator
    options:
      show_root_heading: true
      show_root_full_path: false

---

## See Also

- [NumericalOrbitPropagator](numerical_orbit_propagator.md) - Orbit propagator with built-in force models
- [Event Detection](../events/index.md) - Event detection system
- [Numerical Propagation Guide](../../learn/orbit_propagation/numerical_propagation/index.md) - User guide documentation
