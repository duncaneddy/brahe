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

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_overview.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_overview.rs.txt"
        ```

## Architecture Overview

### Configuration Hierarchy

`ForceModelConfig` is the top-level container that aggregates all force model settings. Each force type has its own configuration struct:

``` .no-linenums
ForceModelConfig
├── gravity: GravityConfiguration
│   ├── Zero
│   ├── PointMass
│   ├── SphericalHarmonic { source, degree, order }
│   └── EarthZonal { degree }
├── drag: DragConfiguration
│   ├── model: AtmosphericModel
│   ├── area: ParameterSource
│   ├── cd: ParameterSource
│   └── body: Option<CentralBody>
├── srp: SolarRadiationPressureConfiguration
│   ├── area: ParameterSource
│   ├── cr: ParameterSource
│   └── eclipse_model: EclipseModel
├── third_body: Vec<ThirdBodyConfiguration>
│   └── per entry:
│       ├── body: ThirdBody
│       ├── ephemeris_source: EphemerisSource
│       └── gravity: GravityConfiguration
├── relativity: bool
└── mass: ParameterSource
```

Each sub-configuration is optional (`None` disables that force). The configuration is captured at propagator construction time and remains immutable during propagation.

Each third-body entry carries its own ephemeris source and gravity model (point-mass by default), and drag can name a `body` other than the central body — see [Body Attribution](#body-attribution) below. In Python, `third_body` accepts a single `ThirdBody` or `ThirdBodyConfiguration`, or a list mixing both; bare bodies become point-mass entries with DE440s ephemerides. Serialized configurations deserialize the same shapes.

### Parameter Sources

Spacecraft parameters (mass $m$, drag area $A_d$, coefficient of drag $C_d$, SRP area $A_{SRP}$, coefficient of reflectivity $C_r$) can be specified in two ways via `ParameterSource`:

- **`Value(f64)`** - Fixed constant embedded at construction. The value is baked into the dynamics function and cannot change during propagation.

- **`ParameterIndex(usize)`** - Index into an parameter vector. This allows parameters to be varied or estimated as part of orbit determination or sensitivity analysis.

The [Parameter Configuration](#parameter-configuration) section below provides detailed examples of both approaches.

## Force Model Components

### Gravity Configuration

Gravity is the primary force in orbital mechanics. Brahe supports the following central gravity models (plus `GravityConfiguration.zero()` for barycentric propagation centers, which have no mass of their own and take all gravitational forces from third-body entries):

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

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_gravity_pointmass.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_gravity_pointmass.rs.txt"
        ```

**Spherical Harmonics**: High-fidelity gravity using a packaged model (EGM2008, GGM05S, JGM3), a user-supplied `.gfc` file, or any model fetched from the [ICGEM catalog](../../datasets/icgem.md). Degree and order control accuracy vs computation time.

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

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_gravity_spherical.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_gravity_spherical.rs.txt"
        ```

#### Selecting a Gravity Model

`GravityConfiguration.spherical_harmonic(...)` accepts a `model_type` argument
that selects which spherical harmonic field is loaded. Four sources are
available:

<div class="center-table" markdown="1">
| Constructor                                  | Source                                                  |
|----------------------------------------------|---------------------------------------------------------|
| `GravityModelType.EGM2008_360`               | Packaged with Brahe (Earth, 360×360)                    |
| `GravityModelType.GGM05S`                    | Packaged with Brahe (Earth, 180×180)                    |
| `GravityModelType.JGM3`                      | Packaged with Brahe (Earth, 70×70)                      |
| `GravityModelType.from_file("path.gfc")`     | Any `.gfc` file on disk                                 |
| `GravityModelType.icgem(body, name)`         | Any model from the [ICGEM catalog](../../datasets/icgem.md) (downloaded on first use) |
</div>

All four are interchangeable at the `GravityConfiguration` boundary. The same
`degree` / `order` truncation rules apply: they must satisfy `degree ≤ model.n_max`
and `order ≤ min(degree, model.m_max)`.

#### Using an ICGEM Gravity Model

`GravityModelType.icgem(body, name)` lets you reference any model from the
[ICGEM catalog](../../datasets/icgem.md) without manually managing the
download. The first time a propagator backed by this configuration is built,
brahe downloads the matching `.gfc` file into
`$BRAHE_CACHE/icgem/models/<body>/` and caches the parsed `GravityModel` in
memory. Subsequent propagators referencing the same model reuse both caches.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/force_model_gravity_icgem.py:13"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/force_model_gravity_icgem.rs:10"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_gravity_icgem.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_gravity_icgem.rs.txt"
        ```

!!! tip "Pre-fetching models for offline runs"
    Call `bh.datasets.icgem.download_model(body, name)` once during setup to
    warm the cache. Subsequent `GravityModelType.icgem(body, name)` references
    then resolve entirely from disk, with no network dependency at propagator
    construction time.

!!! note "Equality of `ICGEMModel` types"
    Two `GravityModelType.icgem(body, name)` instances compare equal only when
    both `body` and `name` match. This is what lets the in-process
    `GravityModel` cache de-duplicate across configurations — multiple
    propagators that request the same ICGEM model share a single parsed
    coefficient set in memory.

For the discovery, refresh, and cache mechanics around ICGEM downloads, see
the [ICGEM dataset guide](../../datasets/icgem.md).

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

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_drag_harris_priester.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_drag_harris_priester.rs.txt"
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

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_drag_nrlmsise.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_drag_nrlmsise.rs.txt"
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

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_drag_exponential.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_drag_exponential.rs.txt"
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

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_srp.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_srp.rs.txt"
        ```

### Third-Body Perturbations

Gravitational attraction from Sun, Moon, and planets causes long-period variations in orbital elements. Each perturbing body is configured with its own `ThirdBodyConfiguration` entry pairing the body with an ephemeris source and a gravity model.

For the default point-mass model the acceleration is the classical tidal form:

$$
\mathbf{a}_{TB} = GM_{b} \left(\frac{\mathbf{r}_b - \mathbf{r}}{|\mathbf{r}_b - \mathbf{r}|^3} - \frac{\mathbf{r}_b}{|\mathbf{r}_b|^3}\right)
$$

where $GM_b$ is the gravitational parameter of the third body, $\mathbf{r}_b$ is its position, and $\mathbf{r}$ is the satellite position. For barycentric central bodies the indirect term ($-GM_b \mathbf{r}_b/|\mathbf{r}_b|^3$) is omitted for bodies whose motion already defines the barycenter (everything for SSB; Earth and the Moon for EMB).

Ephemeris sources (set per entry):

- **LowPrecision**: Fast analytical, Sun/Moon only
- **DE440s**: JPL high precision, all planets, 1550-2650 CE
- **DE440**: JPL high precision, all planets, 13200 BCE-17191 CE

Planet perturbers come in two flavors: the `*Barycenter` variants (`MarsBarycenter` .. `NeptuneBarycenter`, NAIF IDs 4-8) use the planetary-system barycenter position with the system GM — the classical formulation, resolvable from the DE kernel alone and used by the default Earth force models — while the unqualified variants (`Mars` .. `Neptune`, NAIF IDs 499-899) are planet centers with planet-only GMs, resolved through their satellite-system ephemeris kernels.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/force_model_third_body.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/force_model_third_body.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_third_body.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_third_body.rs.txt"
        ```

### Body Attribution

Force models can be attributed to a body other than the propagation's central body. This supports configurations such as an Earth-Moon-barycenter-centered cislunar trajectory that still needs Earth-fidelity forces when passing through low altitudes.

**Extended third-body gravity.** A third-body entry's `gravity` can be `SphericalHarmonic` or `EarthZonal` instead of the default `PointMass`. The field is evaluated at the object's position relative to the perturbing body, oriented by that body's body-fixed frame (ITRF for Earth, LFPA for the Moon, MCMF for Mars, IAU frames for the other planet centers); the indirect term stays point-mass with the field's own GM, so the far-field limit reduces exactly to the tidal form above. `EarthZonal` requires `ThirdBody.EARTH`; `SphericalHarmonic` requires a body with a known body-fixed frame, which excludes the `*Barycenter` variants and `Custom` bodies.

**Attributed drag.** `DragConfiguration.body` names the body whose atmosphere produces the drag (default: the central body). Density and relative wind are evaluated at the object's state relative to that body, and the acceleration applies directly in the propagation frame. The Earth-atmosphere models (`NRLMSISE00`, `HarrisPriester`) require the attributed body to be Earth; the attributed body must have a known radius and spin rate. Drag about a barycentric central body is therefore valid only with an explicit attributed body.

The following example propagates an EMB-centered state through LEO altitudes with an Earth spherical-harmonic field and Earth-attributed NRLMSISE-00 drag:

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/cislunar_earth_forces.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/cislunar_earth_forces.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/numerical_propagation/cislunar_earth_forces.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/numerical_propagation/cislunar_earth_forces.rs.txt"
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

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_parameter_value.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_parameter_value.rs.txt"
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

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_parameter_index.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_parameter_index.rs.txt"
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
| Preset | Gravity | Drag | SRP | Third-Body | Relativity | Solid Tides | Requires Params |
|------|-------|----|---|----------|----------|----------|---------------|
| `two_body()` | PointMass | None | None | None | No | No | No |
| `earth_gravity()` | 20×20 | None | None | None | No | No | No |
| `conservative_forces()` | 80×80 | None | None | Sun/Moon (DE440s) | Yes | No | No |
| `default()` | 20×20 | Harris-Priester | Conical | Sun/Moon (LP) | No | No | Yes |
| `leo_default()` | 30×30 | NRLMSISE-00 | Conical | Sun/Moon (DE440s) | No | No | Yes |
| `geo_default()` | 8×8 | None | Conical | Sun/Moon (DE440s) | No | No | Yes |
| `high_fidelity()` | 120×120 | NRLMSISE-00 | Conical | All planets (DE440s) | Yes | Yes | Yes |
</div>

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/force_model_presets.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/force_model_presets.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_presets.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_presets.rs.txt"
        ```

---

## See Also

- [Numerical Propagation Overview](index.md) - Architecture and concepts
- [Integrator Configuration](integrator_configuration.md) - Integration methods
- [ForceModelConfig API Reference](../../../library_api/propagators/force_model_config.md)
