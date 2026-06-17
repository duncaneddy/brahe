# Tidal Corrections

Earth's oceans and solid body are elastically deformed by the gravitational
attraction of the Sun and Moon. This deformation causes time-varying changes
to the geopotential, producing small but measurable accelerations on low-Earth
orbit satellites. For a 500 km LEO satellite propagated over one orbital period,
solid Earth tides shift the position by roughly 1–2 m relative to a tide-free
model.

Brahe implements the solid Earth tide model from Chapter 6 of the IERS
Conventions (2010), Technical Note 36 (TN36):
<https://iers-conventions.obspm.fr/content/chapter6/icc6.pdf>

## The Three Tide Systems

Geopotential models are published in one of three *tide systems*, which differ
in how the permanent (zero-frequency) part of the tidal potential is handled.
This distinction affects only the degree-2 zonal coefficient $\bar{C}_{20}$.

| System | $\bar{C}_{20}$ contains | Notes |
|---|---|---|
| **Mean-tide** | direct + indirect permanent terms | Physically consistent with the mean sea surface; rare for satellite models |
| **Zero-tide** | indirect term only (elastic deformation) | Used by some older models (e.g. EGM96 S11) |
| **Tide-free** | neither term | Conventional choice; used by EGM2008, GGM05S, JGM-3 |

The *direct* term is the permanent tidal potential itself (IERS Eq. 6.14, $A_0 H_0$);
the *indirect* term is Earth's permanent elastic response to that potential
(scaled by the secular Love number $k_{20} = 0.30190$, IERS Table 6.3):

$$
\delta\bar{C}_{20}^{\text{direct}} = A_0 H_0 = 4.4228 \times 10^{-8} \times (-0.31460) \approx -1.391 \times 10^{-8}
$$

$$
\delta\bar{C}_{20}^{\text{indirect}} = A_0 H_0 k_{20} \approx -4.201 \times 10^{-9}
$$

(IERS Conventions (2010), §6.2.2, Eqs. 6.13–6.14)

The solid-tide acceleration model (§6.2.1) assumes a *conventional tide-free*
background. When using a model whose $\bar{C}_{20}$ is in a different system,
the permanent term must be removed before adding the time-varying solid-tide
corrections, so the two contributions are not double-counted.

## Permanent Tide Configuration

`PermanentTideConfig` controls how Brahe reconciles the loaded model's
$\bar{C}_{20}$ with the conventional tide-free convention:

| Variant | Behavior |
|---|---|
| `Auto` *(default)* | Reads the tide-system flag stored in the model file and converts $\bar{C}_{20}$ to conventional tide-free automatically. If the flag is `Unknown`, no conversion is applied (a warning is emitted). |
| `ConvertTo(system)` | Forces the model into the specified tide system. Errors at propagator construction if the stored flag is `Unknown`. |
| `Off` | Leaves $\bar{C}_{20}$ untouched. Use when you have pre-corrected the model yourself, or for debugging. |

`Auto` is the right choice for almost all practical cases.

## Solid Earth Tides

The solid Earth tide model adds time-varying corrections $\Delta\bar{C}_{nm}$
and $\Delta\bar{S}_{nm}$ to the static geopotential coefficients at each
integration step.

### Step 1 — Frequency-Independent Corrections (Always On)

When solid tides are enabled, Step 1 is always computed. For each tide-raising
body (Moon and Sun) at ECEF position $\mathbf{r}_j$ with distance $r_j$,
geocentric latitude $\phi_j$, and longitude $\lambda_j$:

$$
\Delta\bar{C}_{nm} - i\,\Delta\bar{S}_{nm} =
\frac{k_{nm}}{2n+1} \sum_j \frac{GM_j}{GM_\oplus}
\left(\frac{R_\oplus}{r_j}\right)^{n+1}
\bar{P}_{nm}(\sin\phi_j)\, e^{-im\lambda_j}
$$

(IERS Eq. 6.6, degrees $n = 2, 3$)

The complex Love numbers $k_{nm} = k^{\rm re}_{nm} + i\,k^{\rm im}_{nm}$ are
taken from IERS Table 6.3 (anelastic values). The imaginary parts are non-zero
only for the degree-2 terms and produce small out-of-phase corrections.

Degree-2 tides also feed back into degree-4 via coupling Love numbers
$k_m^+$ (IERS Eq. 6.7, $m = 0, 1, 2$):

$$
\Delta\bar{C}_{4m} - i\,\Delta\bar{S}_{4m} =
\frac{k_m^+}{5} \sum_j \frac{GM_j}{GM_\oplus}
\left(\frac{R_\oplus}{r_j}\right)^{3}
\bar{P}_{2m}(\sin\phi_j)\, e^{-im\lambda_j}
$$

The resulting $\Delta\bar{C}_{nm}/\Delta\bar{S}_{nm}$ corrections (up to
degree 4) are evaluated as a spherical-harmonic acceleration using the
Cunningham V/W recursion, then added directly to the static-gravity
acceleration.

### Step 2 — Frequency-Dependent Corrections (Optional)

Setting `frequency_dependent=True` on `SolidTideConfig` activates the IERS
Step 2 tables (Tables 6.5a/b/c). These corrections account for the frequency
dependence of the anelastic Love numbers near tidal resonances, primarily
affecting the degree-2 coefficients:

| Quantity | Equation | Amplitudes |
|---|---|---|
| $\Delta\bar{C}_{20}$ | Eq. 6.8a (real part, $m=0$, 21 lines) | up to ~17 × 10⁻¹² |
| $\Delta\bar{C}_{21}, \Delta\bar{S}_{21}$ | Eq. 6.8b ($m=1$, 48 lines) | up to ~471 × 10⁻¹² |
| $\Delta\bar{C}_{22}, \Delta\bar{S}_{22}$ | Eq. 6.8c ($m=2$, 2 lines) | up to ~1.2 × 10⁻¹² |

Each correction line is computed from Doodson/Delaunay arguments
$\theta_f = m(\theta_G + \pi) - (n_l l + n_{l'} l' + n_F F + n_D D + n_\Omega \Omega)$
(IERS §6.2.1), where $\theta_G$ is GMST and $l, l', F, D, \Omega$ are the five
Delaunay fundamental arguments evaluated at the current TT epoch.

Step 2 corrections are at the $10^{-10}$–$10^{-11}$ level, contributing a
sub-millimetre position effect per orbit for LEO satellites. They are
recommended for precise orbit determination but can be omitted for most
mission-analysis applications.

## Configuring Tides

### Rust

=== "Tides ON (Step 1 + Step 2)"

    ```rust
    use brahe as bh;

    // Solid tides: IERS Step 1 always-on + Step 2 frequency-dependent corrections.
    // PermanentTideConfig::Auto converts C̄20 to conventional tide-free automatically.
    let tides = bh::TidesConfiguration {
        permanent: bh::PermanentTideConfig::Auto,
        solid: Some(bh::SolidTideConfig { frequency_dependent: true }),
    };

    let mut force_config = bh::ForceModelConfig::earth_gravity();
    force_config.tides = Some(tides);
    ```

=== "Step 1 only (faster)"

    ```rust
    use brahe as bh;

    let tides = bh::TidesConfiguration {
        permanent: bh::PermanentTideConfig::Auto,
        solid: Some(bh::SolidTideConfig { frequency_dependent: false }),
    };

    let mut force_config = bh::ForceModelConfig::earth_gravity();
    force_config.tides = Some(tides);
    ```

=== "Permanent-tide correction only"

    ```rust
    use brahe as bh;

    // Corrects C̄20 for the tide system of the loaded model, but adds no
    // time-varying solid-tide accelerations.
    let tides = bh::TidesConfiguration {
        permanent: bh::PermanentTideConfig::Auto,
        solid: None,
    };

    let mut force_config = bh::ForceModelConfig::earth_gravity();
    force_config.tides = Some(tides);
    ```

### Python

=== "Tides ON (Step 1 + Step 2)"

    ```python
    import brahe as bh

    solid = bh.SolidTideConfig(frequency_dependent=True)
    tides = bh.TidesConfiguration(
        permanent=bh.PermanentTideConfig.AUTO,
        solid=solid,
    )

    force_config = bh.ForceModelConfig.earth_gravity()
    force_config.tides = tides
    ```

=== "Step 1 only (faster)"

    ```python
    import brahe as bh

    solid = bh.SolidTideConfig(frequency_dependent=False)
    tides = bh.TidesConfiguration(
        permanent=bh.PermanentTideConfig.AUTO,
        solid=solid,
    )

    force_config = bh.ForceModelConfig.earth_gravity()
    force_config.tides = tides
    ```

=== "Permanent-tide correction only"

    ```python
    import brahe as bh

    tides = bh.TidesConfiguration(
        permanent=bh.PermanentTideConfig.AUTO,
        solid=None,
    )

    force_config = bh.ForceModelConfig.earth_gravity()
    force_config.tides = tides
    ```

## Worked Example

The example below propagates a 500 km LEO satellite for one full orbital period
with tides enabled and disabled, then prints the peak position difference.

=== "Python"

    ```python
    --8<-- "./examples/numerical_propagation/force_model_tides.py:7"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/numerical_propagation/force_model_tides.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_tides.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/numerical_propagation/force_model_tides.rs.txt"
        ```

## When to Enable Tides

| Application | Recommendation |
|---|---|
| Mission analysis, maneuver planning | Off (saves compute, effect is ~1 m per orbit) |
| Long-arc orbit determination (≥ 1 day) | Step 1 (`frequency_dependent=False`) |
| Precise orbit determination, POD | Step 1 + Step 2 (`frequency_dependent=True`) |
| Geodesy, altimetry calibration | Step 1 + Step 2 |

## Required Force Model Setup

Solid-tide computation requires Sun and Moon positions in the ECEF frame. The
propagator obtains these from the third-body force model when it is enabled.
Make sure `third_body` includes at least `Sun` and `Moon` when solid tides are
on, otherwise the tidal accelerations will be zero.

## See Also

- [Gravity Models](gravity.md) — tide systems of packaged models, `GravityModelTideSystem`
- [Force Models](../orbit_propagation/numerical_propagation/force_models.md) — wiring the full force model

## References

- Petit, G., & Luzum, B. (Eds.) (2010). *IERS Conventions (2010)*, Technical Note 36, Chapter 6: Geopotential. IERS. <https://iers-conventions.obspm.fr/content/chapter6/icc6.pdf>
- Lemoine, F. G., et al. (1998). *The Development of the Joint NASA GSFC and the National Imagery and Mapping Agency (NIMA) Geopotential Model EGM96*. NASA/TP-1998-206861. (EGM96 S11 zero-tide C̄20.)
