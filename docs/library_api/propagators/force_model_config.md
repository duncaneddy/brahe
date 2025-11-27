# Force Model Configuration

Configuration classes for numerical orbit propagation force models. `ForceModelConfig` provides factory methods for common configurations and allows customization of gravity, atmospheric drag, solar radiation pressure, and third-body perturbations.

!!! note
    For conceptual explanations and usage examples, see [Force Models](../../learn/orbit_propagation/numerical_propagation/force_models.md) in the User Guide.

## ForceModelConfig

::: brahe.ForceModelConfig
    options:
      show_root_heading: true
      show_root_full_path: false

## Configuration Components

### GravityConfiguration

::: brahe.GravityConfiguration
    options:
      show_root_heading: true
      show_root_full_path: false

### DragConfiguration

::: brahe.DragConfiguration
    options:
      show_root_heading: true
      show_root_full_path: false

### SolarRadiationPressureConfiguration

::: brahe.SolarRadiationPressureConfiguration
    options:
      show_root_heading: true
      show_root_full_path: false

### ThirdBodyConfiguration

::: brahe.ThirdBodyConfiguration
    options:
      show_root_heading: true
      show_root_full_path: false

## Enumerations

### AtmosphericModel

::: brahe.AtmosphericModel
    options:
      show_root_heading: true
      show_root_full_path: false

### EclipseModel

::: brahe.EclipseModel
    options:
      show_root_heading: true
      show_root_full_path: false

### ThirdBody

::: brahe.ThirdBody
    options:
      show_root_heading: true
      show_root_full_path: false

### ParameterSource

::: brahe.ParameterSource
    options:
      show_root_heading: true
      show_root_full_path: false

---

## See Also

- [NumericalOrbitPropagator](numerical_orbit_propagator.md) - Propagator using force models
- [Orbital Dynamics](../orbit_dynamics/index.md) - Detailed force model documentation
- [Force Models Guide](../../learn/orbit_propagation/numerical_propagation/force_models.md) - User guide
