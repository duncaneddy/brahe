# Force Model Configuration

Configuration classes for numerical orbit propagation force models. `ForceModelConfig` provides factory methods for common configurations and allows customization of gravity, atmospheric drag, solar radiation pressure, third-body perturbations, and solid Earth tides.

!!! note
    For conceptual explanations and usage examples, see [Force Models](../../learn/orbit_propagation/numerical_propagation/force_models.md) in the User Guide. For `CentralBody`, `lunar_default()`/`cislunar_default()`/`mars_default()`, and `state_in_frame`, see [Cislunar and Lunar Propagation](../../learn/orbit_propagation/numerical_propagation/cislunar_lunar_propagation.md) and [Propagation Around Other Central Bodies](../../learn/orbit_propagation/numerical_propagation/other_central_bodies.md).

## ForceModelConfig

::: brahe.ForceModelConfig

## Central Body

::: brahe.CentralBody

## Configuration Components

::: brahe.GravityConfiguration

::: brahe.DragConfiguration

::: brahe.SolarRadiationPressureConfiguration

::: brahe.ThirdBodyConfiguration

::: brahe.TidesConfiguration

::: brahe.SolidTideConfig

::: brahe.PermanentTideConfig

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
- [Cislunar and Lunar Propagation](../../learn/orbit_propagation/numerical_propagation/cislunar_lunar_propagation.md) - `CentralBody`, `lunar_default()`/`cislunar_default()`, and `state_in_frame`
- [Propagation Around Other Central Bodies](../../learn/orbit_propagation/numerical_propagation/other_central_bodies.md) - Mars, custom bodies, and user-defined body-fixed frames
