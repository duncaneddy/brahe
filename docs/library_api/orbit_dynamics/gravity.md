# Gravity Models

Gravity acceleration functions including point-mass and spherical harmonic models.

!!! note
    For conceptual explanations and examples, see [Gravity Models](../../learn/orbital_dynamics/gravity.md) in the Learn section.

## Point-Mass Gravity

::: brahe.accel_point_mass_gravity

## Spherical Harmonic Gravity

`accel_gravity_spherical_harmonics` dispatches to whichever kernel the
[`GravityModel`](#gravity-model-class)'s [`GravityModelCoefficients`](#gravity-model-coefficients) has
precomputed (Clenshaw by default). `accel_gravity_spherical_harmonics_clenshaw`
and `accel_gravity_spherical_harmonics_cunningham` force evaluation through a
specific kernel instead of dispatching automatically.

::: brahe.accel_gravity_spherical_harmonics

::: brahe.accel_gravity_spherical_harmonics_clenshaw

::: brahe.accel_gravity_spherical_harmonics_cunningham

## Gravity Model Class

::: brahe.GravityModel

## Gravity Model Coefficients

`GravityModelCoefficients` selects which kernel's precomputed coefficient set(s) a
[`GravityModel`](#gravity-model-class) builds when it is loaded
(`GravityModel.from_model_type` / `from_file` default to `GravityModelCoefficients.Clenshaw`;
`from_model_type_with_coefficients` / `from_file_with_coefficients` accept an explicit
value): `GravityModelCoefficients.Clenshaw` (default), `.Cunningham`, or `.Both`. A model
can only evaluate through a kernel whose coefficient set is present;
`GravityModel.precompute_clenshaw_coefficients()` / `precompute_cunningham_coefficients()`
add a coefficient set to an already-loaded model, and `drop_clenshaw_coefficients()` /
`drop_cunningham_coefficients()` free one to reduce memory use.

::: brahe.GravityModelCoefficients

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
