# NumericalOrbitPropagatorBuilder

Builder for `NumericalOrbitPropagator`. `builder()` takes the three required inputs (`epoch`, `state`, `force_config`) directly as arguments; optional inputs are set through chained setters and default to `None` (`NumericalPropagationConfig.default()` for the propagation configuration).

!!! note
    For conceptual explanations and usage examples, see [Numerical Orbit Propagator](../../learn/orbit_propagation/numerical_propagation/numerical_orbit_propagator.md) in the User Guide.

::: brahe.NumericalOrbitPropagatorBuilder
    options:
      show_root_heading: true
      show_root_full_path: false

---

## See Also

- [NumericalOrbitPropagator](numerical_orbit_propagator.md) - High-fidelity numerical orbit propagator with built-in force models
- [Numerical Orbit Propagator Guide](../../learn/orbit_propagation/numerical_propagation/numerical_orbit_propagator.md) - User guide documentation
