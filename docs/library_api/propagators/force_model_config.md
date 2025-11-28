# Force Model Configuration

Configuration classes for numerical orbit propagation force models. `ForceModelConfig` provides factory methods for common configurations and allows customization of gravity, atmospheric drag, solar radiation pressure, and third-body perturbations.

!!! note
    For conceptual explanations and usage examples, see [Force Models](../../learn/orbit_propagation/numerical_propagation/force_models.md) in the User Guide.

## ForceModelConfig

::: brahe.ForceModelConfig

## Configuration Components

::: brahe.GravityConfiguration

::: brahe.DragConfiguration

::: brahe.SolarRadiationPressureConfiguration

::: brahe.ThirdBodyConfiguration

## Enumerations

::: brahe.AtmosphericModel

::: brahe.EclipseModel

::: brahe.ThirdBody

::: brahe.ParameterSource

---

## See Also

- [NumericalOrbitPropagator](numerical_orbit_propagator.md) - Propagator using force models
- [Orbital Dynamics](../orbit_dynamics/index.md) - Detailed force model documentation
- [Force Models Guide](../../learn/orbit_propagation/numerical_propagation/force_models.md) - User guide
