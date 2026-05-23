# Gravity Models

Gravity acceleration functions including point-mass and spherical harmonic models.

!!! note
    For conceptual explanations and examples, see [Gravity Models](../../learn/orbital_dynamics/gravity.md) in the Learn section.

## Point-Mass Gravity

::: brahe.accel_point_mass_gravity

## Spherical Harmonic Gravity

::: brahe.accel_gravity_spherical_harmonics

## Gravity Model Class

::: brahe.GravityModel

## Gravity Model Type

`GravityModelType` selects which spherical harmonic field is loaded into a
[`GravityModel`](#gravity-model-class) (or wired into a
[`GravityConfiguration`](../propagators/force_model_config.md)). In addition to
the packaged constants `EGM2008_360`, `GGM05S`, and `JGM3`, two constructors
load external models:

- `GravityModelType.from_file(path)` — load any `.gfc` file from disk.
- `GravityModelType.icgem(body, name)` — reference any model from the
  [ICGEM catalog](../../learn/datasets/icgem.md); the matching `.gfc` file is
  downloaded into `$BRAHE_CACHE/icgem/models/<body>/` on first use and cached
  permanently.

The same `GravityModelType` can be passed to
[`GravityConfiguration.spherical_harmonic(model_type=...)`](../propagators/force_model_config.md)
to use the model as the central-body field in a `NumericalOrbitPropagator`.
For end-to-end usage examples see the
[ICGEM dataset guide](../../learn/datasets/icgem.md) and the
[Gravity Models user guide](../../learn/orbital_dynamics/gravity.md#loading-models-from-icgem).

::: brahe.GravityModelType

## See Also

- [Gravity Models (Learn)](../../learn/orbital_dynamics/gravity.md) - Conceptual explanation and examples
- [ICGEM Dataset Interface (Learn)](../../learn/datasets/icgem.md) - Discovering and downloading ICGEM models
- [ICGEM Functions API](../datasets/icgem.md) - `brahe.datasets.icgem` API reference
- [Force Models (Learn)](../../learn/orbit_propagation/numerical_propagation/force_models.md) - Using a gravity model in a numerical propagator
- [Orbital Dynamics Module](index.md) - Complete orbit dynamics API reference
