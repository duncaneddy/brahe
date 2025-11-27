# Numerical Propagation Configuration

Configuration classes for numerical integration settings. `NumericalPropagationConfig` combines the integration method, integrator tolerances, and variational equation settings into a single configuration object.

!!! note
    For conceptual explanations and usage examples, see [Integrator Configuration](../../learn/orbit_propagation/numerical_propagation/integrator_configuration.md) in the User Guide.

## NumericalPropagationConfig

::: brahe.NumericalPropagationConfig
    options:
      show_root_heading: true
      show_root_full_path: false

## VariationalConfig

::: brahe.VariationalConfig
    options:
      show_root_heading: true
      show_root_full_path: false

## Enumerations

::: brahe.IntegrationMethod
    options:
      show_root_heading: true
      show_root_full_path: false

---

## See Also

- [IntegratorConfig](../integrators/config.md) - Integrator configuration
- [NumericalOrbitPropagator](numerical_orbit_propagator.md) - Orbit propagator using this configuration
- [NumericalPropagator](numerical_propagator.md) - Generic propagator using this configuration
- [ForceModelConfig](force_model_config.md) - Force model configuration
- [Integrator Configuration Guide](../../learn/orbit_propagation/numerical_propagation/integrator_configuration.md) - User guide
- [Covariance and Sensitivity](../../learn/orbit_propagation/numerical_propagation/covariance_sensitivity.md) - Variational equations guide
