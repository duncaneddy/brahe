# NumericalPropagatorBuilder

Builder for `NumericalPropagator`. `builder()` takes the three required inputs (`epoch`, `state`, `dynamics_fn`) directly as arguments; optional inputs are set through chained setters and default to `None` (`NumericalPropagationConfig.default()` for the propagation configuration).

!!! note
    For conceptual explanations and usage examples, see [General Dynamics Propagation](../../learn/orbit_propagation/numerical_propagation/generic_dynamics.md) in the User Guide.

::: brahe.NumericalPropagatorBuilder
    options:
      show_root_heading: true
      show_root_full_path: false

---

## See Also

- [NumericalPropagator](numerical_propagator.md) - Generic numerical propagator for arbitrary dynamics
- [General Dynamics Propagation Guide](../../learn/orbit_propagation/numerical_propagation/generic_dynamics.md) - User guide documentation
