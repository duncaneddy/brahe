# Force Model Configuration

Configuration classes for numerical orbit propagation force models. `ForceModelConfig` provides factory methods for common configurations and allows customization of gravity, atmospheric drag, solar radiation pressure, and third-body perturbations.

!!! note
    For conceptual explanations and usage examples, see [Force Models](../../learn/orbit_propagation/numerical_propagation/force_models.md) in the User Guide. For `CentralBody`, `lunar_default()`/`mars_default()`, and `state_in_frame`, see [Frame Router & Multibody Propagation](../../learn/frames/frame_transformations.md).

## ForceModelConfig

::: brahe.ForceModelConfig

## Central Body

::: brahe.CentralBody

## Configuration Components

::: brahe.GravityConfiguration

::: brahe.DragConfiguration

::: brahe.SolarRadiationPressureConfiguration

::: brahe.ThirdBodyConfiguration

## Enumerations

::: brahe.AtmosphericModel

::: brahe.EclipseModel

::: brahe.ThirdBody

::: brahe.OccultingBody

::: brahe.ParameterSource

---

## See Also

- [NumericalOrbitPropagator](numerical_orbit_propagator.md) - Propagator using force models
- [Orbital Dynamics](../orbit_dynamics/index.md) - Detailed force model documentation
- [Force Models Guide](../../learn/orbit_propagation/numerical_propagation/force_models.md) - User guide
- [Frame Router & Multibody Propagation](../../learn/frames/frame_transformations.md) - `CentralBody`, propagation defaults, and `state_in_frame`
