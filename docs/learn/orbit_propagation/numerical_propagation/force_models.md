# Force Models

The `ForceModelConfig` (Python) / `ForceModelConfiguration` (Rust) defines which physical forces affect the spacecraft during propagation. Brahe provides preset configurations for common scenarios and allows custom configurations for specific requirements.

For API details, see the [ForceModelConfig API Reference](../../../library_api/propagators/force_model_config.md).

## Force Model Components

### Gravity Configuration

Gravity is the primary force in orbital mechanics. Brahe supports two gravity models:

**Point Mass**: Simple two-body central gravity ($\mu/r^2$). Fast but ignores Earth's non-spherical shape.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/force_model_gravity_pointmass.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/force_model_gravity_pointmass.rs:4"
    ```

**Spherical Harmonics**: High-fidelity gravity using EGM2008 model. Degree and order control accuracy vs computation time.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/force_model_gravity_spherical.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/force_model_gravity_spherical.rs:4"
    ```

### Atmospheric Drag

Atmospheric drag is significant for LEO satellites. Three atmospheric models are available:

**Harris-Priester**: Fast model with diurnal density variations. Valid 100-1000 km altitude. No space weather data required.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/force_model_drag_harris_priester.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/force_model_drag_harris_priester.rs:4"
    ```

**NRLMSISE-00**: High-fidelity empirical model using space weather data. Valid from ground to thermosphere.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/force_model_drag_nrlmsise.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/force_model_drag_nrlmsise.rs:4"
    ```

**Exponential**: Simple analytical model: $\rho(h) = \rho_0 \exp(-(h-h_0)/H)$. Fast for rough estimates.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/force_model_drag_exponential.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/force_model_drag_exponential.rs:4"
    ```

### Solar Radiation Pressure

SRP is significant for high-altitude orbits and high area-to-mass ratio spacecraft. Eclipse models determine shadow effects:

- **None**: Always illuminated (fast, inaccurate in shadow)
- **Cylindrical**: Sharp shadow boundary (simple, fast)
- **Conical**: Penumbra and umbra regions (most accurate)

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/force_model_srp.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/force_model_srp.rs:4"
    ```

### Third-Body Perturbations

Gravitational attraction from Sun, Moon, and planets causes long-period variations in orbital elements.

Ephemeris sources:

- **LowPrecision**: Fast analytical (~km accuracy), Sun/Moon only
- **DE440s**: JPL high precision (~m accuracy), all planets, 1550-2650 CE
- **DE440**: JPL highest precision (~mm accuracy), all planets, 13200 BCE-17191 CE

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/force_model_third_body.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/force_model_third_body.rs:4"
    ```

### Relativistic Effects

General relativistic corrections can be enabled via the `relativity` boolean flag. These effects are typically small but can be significant for precision orbit determination.

## Parameter Configuration

Force model parameters (mass, drag area, Cd, etc.) can be specified either as fixed values or as indices into a parameter vector.

### Using Fixed Values

Use `ParameterSource.value()` (Python) / `ParameterSource::Value` (Rust) for parameters that don't change:

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/force_model_parameter_value.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/force_model_parameter_value.rs:4"
    ```

### Using Parameter Indices

Use `ParameterSource.from_index()` (Python) / `ParameterSource::ParameterIndex` (Rust) for parameters that may be varied or estimated:

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/force_model_parameter_index.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/force_model_parameter_index.rs:4"
    ```

### Default Parameter Layout

When using parameter indices, the default layout is:

| Index | Parameter | Units | Typical Value |
|-------|-----------|-------|---------------|
| 0 | mass | kg | 1000.0 |
| 1 | drag_area | m² | 10.0 |
| 2 | Cd | - | 2.2 |
| 3 | srp_area | m² | 10.0 |
| 4 | Cr | - | 1.3 |

Custom indices can be used by specifying different values in the configuration.

### Building Custom Configurations

Combine components for specific mission requirements:

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/force_model_custom.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/force_model_custom.rs:4"
    ```

## Preset Configurations

Brahe provides preset configurations for common scenarios:

| Preset | Gravity | Drag | SRP | Third-Body | Relativity | Requires Params |
|--------|---------|------|-----|------------|------------|-----------------|
| `two_body()` | PointMass | None | None | None | No | No |
| `earth_gravity()` | 20×20 | None | None | None | No | No |
| `conservative_forces()` | 80×80 | None | None | Sun/Moon (DE440s) | Yes | No |
| `default()` | 20×20 | Harris-Priester | Conical | Sun/Moon (LP) | No | Yes |
| `leo_default()` | 30×30 | NRLMSISE-00 | Conical | Sun/Moon (DE440s) | No | Yes |
| `geo_default()` | 8×8 | None | Conical | Sun/Moon (DE440s) | No | Yes |
| `high_fidelity()` | 120×120 | NRLMSISE-00 | Conical | All planets (DE440s) | Yes | Yes |

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/force_model_presets.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/force_model_presets.rs:4"
    ```

## Performance Considerations

Force model complexity directly affects propagation speed:

1. **Gravity harmonics**: Computation scales with $(n+1)^2$ for degree $n$
2. **Third-body**: Requires ephemeris lookups for Sun/Moon positions
3. **Drag**: Requires density model evaluation and velocity-relative calculations
4. **SRP**: Requires eclipse geometry computation

For time-critical applications, use the minimum fidelity that meets accuracy requirements.

---

## See Also

- [Numerical Propagation Overview](index.md) - Architecture and concepts
- [Integrator Configuration](integrator_configuration.md) - Integration methods
- [ForceModelConfig API Reference](../../../library_api/propagators/force_model_config.md)
