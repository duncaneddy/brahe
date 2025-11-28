# Force Models

The `ForceModelConfig` (Python) / `ForceModelConfig` (Rust) defines which physical forces affect the spacecraft during propagation. Brahe provides preset configurations for common scenarios and allows custom configurations for specific requirements.

For API details, see the [ForceModelConfig API Reference](../../../library_api/propagators/force_model_config.md).

## Full Example

here is a complete example creating a `ForceModelConfig` exercising all available configuration options:

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/force_model_overview.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/force_model_overview.rs:4"
    ```

## Architecture Overview

### Configuration Hierarchy

`ForceModelConfig` is the top-level container that aggregates all force model settings. Each force type has its own configuration struct:

``` .no-linenums
ForceModelConfig
├── gravity: GravityConfiguration
│   ├── PointMass
│   └── SphericalHarmonic { source, degree, order }
├── drag: DragConfiguration
│   ├── model: AtmosphericModel
│   ├── area: ParameterSource
│   └── cd: ParameterSource
├── srp: SolarRadiationPressureConfiguration
│   ├── area: ParameterSource
│   ├── cr: ParameterSource
│   └── eclipse_model: EclipseModel
├── third_body: ThirdBodyConfiguration
│   ├── ephemeris_source: EphemerisSource
│   └── bodies: Vec<ThirdBody>
├── relativity: bool
└── mass: ParameterSource
```

Each sub-configuration is optional (`None` disables that force). The configuration is captured at propagator construction time and remains immutable during propagation.

### Parameter Sources

Spacecraft parameters (mass $m$, drag area $A_d$, coefficient of drag $C_d$, SRP area $A_{SRP}$, coefficient of reflectivity $C_r$) can be specified in two ways via `ParameterSource`:

- **`Value(f64)`** - Fixed constant embedded at construction. The value is baked into the dynamics function and cannot change during propagation.

- **`ParameterIndex(usize)`** - Index into an parameter vector. This allows parameters to be varied or estimated as part of orbit determination or sensitivity analysis.

The [Parameter Configuration](#parameter-configuration) section below provides detailed examples of both approaches.

## Force Model Components

### Gravity Configuration

Gravity is the primary force in orbital mechanics. Brahe supports two gravity models:

**Point Mass**: Simple two-body central gravity. Fast but ignores Earth's non-spherical shape.

$$
\mathbf{a} = -\frac{GM}{r^3} \mathbf{r}
$$

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/force_model_gravity_pointmass.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/force_model_gravity_pointmass.rs:4"
    ```

**Spherical Harmonics**: High-fidelity gravity using EGM2008, GGM05S, or user-defined `.gfc` model. Degree and order control accuracy vs computation time.

$$
\mathbf{a} = -\nabla V, \quad V(r, \phi, \lambda) = \frac{GM}{r} \sum_{n=0}^{N} \sum_{m=0}^{n} \left(\frac{R_E}{r}\right)^n \bar{P}_{nm}(\sin\phi) \left(\bar{C}_{nm}\cos(m\lambda) + \bar{S}_{nm}\sin(m\lambda)\right)
$$

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/force_model_gravity_spherical.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/force_model_gravity_spherical.rs:4"
    ```

### Atmospheric Drag

Atmospheric drag is significant for LEO satellites.

$$
\mathbf{a}_D = -\frac{1}{2} C_D \frac{A}{m} \rho v_{rel}^2 \mathbf{\hat{v}}_{rel}
$$

where $\rho$ is atmospheric density, $v_{rel}$ is velocity relative to the atmosphere, $C_D$ is drag coefficient, and $A/m$ is area-to-mass ratio.

Three atmospheric models are available:

**Harris-Priester**: Fast model with diurnal density variations. Valid 100-1000 km altitude. No space weather data required.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/force_model_drag_harris_priester.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/force_model_drag_harris_priester.rs:4"
    ```

**NRLMSISE-00**: High-fidelity empirical model using space weather data. Valid from ground to thermosphere (~1000 km). More computationally intensive.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/force_model_drag_nrlmsise.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/force_model_drag_nrlmsise.rs:4"
    ```

**Exponential**: An expontential atmospheric density model defined by which provides a simple approximation that is fast for rough calculations:

$$
\rho(h) = \rho_0 e^{-\frac{h-h_0}{H}}
$$

$\rho_0$ is reference density at altitude $h_0$ and $H$ is scale height.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/force_model_drag_exponential.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/force_model_drag_exponential.rs:4"
    ```

### Solar Radiation Pressure

SRP is significant for high-altitude orbits and high area-to-mass ratio spacecraft.

$$
\mathbf{a}_{SRP} = -P_{\odot} C_R \frac{A}{m} \nu \frac{\mathbf{r}_{\odot}}{|\mathbf{r}_{\odot}|}
$$

where $P_{\odot} \approx 4.56 \times 10^{-6}$ N/m² is solar pressure at 1 AU, $C_R$ is reflectivity coefficient, $\nu$ is shadow function (0-1), and $\mathbf{r}_{\odot}$ is the Sun position vector.

Eclipse models determine shadow effects:

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

$$
\mathbf{a}_{TB} = GM_{b} \left(\frac{\mathbf{r}_b - \mathbf{r}}{|\mathbf{r}_b - \mathbf{r}|^3} - \frac{\mathbf{r}_b}{|\mathbf{r}_b|^3}\right)
$$

where $GM_b$ is the gravitational parameter of the third body, $\mathbf{r}_b$ is its position, and $\mathbf{r}$ is the satellite position.

Ephemeris sources:

- **LowPrecision**: Fast analytical, Sun/Moon only
- **DE440s**: JPL high precision, all planets, 1550-2650 CE
- **DE440**: JPL high precision, all planets, 13200 BCE-17191 CE

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

$$
\mathbf{a} = -\frac{GM}{r^2} \left( \left( 4\frac{GM}{c^2r} - \frac{v^2}{c^2} \right)\mathbf{e}_r + 4\frac{v^2}{c^2}\left(\mathbf{e}_r \cdot \mathbf{e}_v\right)\mathbf{e}_v\right)
$$

where $c$ is the speed of light, $\mathbf{e}_r$ is the radial unit vector, and $\mathbf{e}_v$ is the velocity unit vector.

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

<div class="center-table" markdown="1">
| Index | Parameter | Units | Typical Value |
|-----|---------|-----|-------------|
| 0 | mass | kg | 1000.0 |
| 1 | drag_area | m² | 10.0 |
| 2 | Cd | - | 2.2 |
| 3 | srp_area | m² | 10.0 |
| 4 | Cr | - | 1.3 |
</div>

## Preset Configurations

Brahe provides preset configurations for common scenarios:

<div class="center-table" markdown="1">
| Preset | Gravity | Drag | SRP | Third-Body | Relativity | Requires Params |
|------|-------|----|---|----------|----------|---------------|
| `two_body()` | PointMass | None | None | None | No | No |
| `earth_gravity()` | 20×20 | None | None | None | No | No |
| `conservative_forces()` | 80×80 | None | None | Sun/Moon (DE440s) | Yes | No |
| `default()` | 20×20 | Harris-Priester | Conical | Sun/Moon (LP) | No | Yes |
| `leo_default()` | 30×30 | NRLMSISE-00 | Conical | Sun/Moon (DE440s) | No | Yes |
| `geo_default()` | 8×8 | None | Conical | Sun/Moon (DE440s) | No | Yes |
| `high_fidelity()` | 120×120 | NRLMSISE-00 | Conical | All planets (DE440s) | Yes | Yes |
</div>

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/force_model_presets.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/force_model_presets.rs:4"
    ```

---

## See Also

- [Numerical Propagation Overview](index.md) - Architecture and concepts
- [Integrator Configuration](integrator_configuration.md) - Integration methods
- [ForceModelConfig API Reference](../../../library_api/propagators/force_model_config.md)
