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

!!! question "Validation Help Desired"

    Tidal corrections have been implemented based on a best-effort reading of 
    the IERS conventions. However, it is generally a complex model
    with multiple steps and assumptions, so any experts who 
    could review the implementation and confirm its correctness would be greatly appreciated.

## The Three Tide Systems

Geopotential models are published in one of three *tide systems*, which differ
in how the permanent (zero-frequency) part of the tidal potential is handled.
This distinction affects only the degree-2 zonal coefficient $\bar{C}_{20}$.

| System | $\bar{C}_{20}$ contains | Notes |
|---|---|---|
| **Mean-tide** | direct + indirect permanent terms | Physically consistent with the mean sea surface; rare for satellite models |
| **Zero-tide** | indirect term only (elastic deformation) | Used by some older models (e.g. EGM96 S11) |
| **Tide-free** | neither term | Conventional choice; used by EGM2008, GGM05S, JGM-3 |

The *direct* term is the permanent part of the tide-generating potential
itself; the *indirect* term is Earth's permanent elastic deformation in
response to it, given by IERS Eq. 6.14 with the nominal degree-2 Love number
$k_{20} = 0.30190$ (IERS Table 6.3):

$$
\delta\bar{C}_{20}^{\text{direct}} = A_0 H_0 = 4.4228 \times 10^{-8} \times (-0.31460) \approx -1.391 \times 10^{-8}
$$

$$
\delta\bar{C}_{20}^{\text{indirect}} = A_0 H_0 k_{20} \approx -4.201 \times 10^{-9}
$$

The solid-tide acceleration model (§6.2.1) assumes a *conventional tide-free*
background. When using a model whose $\bar{C}_{20}$ is in a different system,
the permanent term must be removed before adding the time-varying solid-tide
corrections, so the two contributions are not double-counted.

## Design: Normalize to Tide-Free, Then Layer

Brahe's tide handling follows one mental model: **when tides are configured
(`ForceModelConfig.tides` is set), the static gravity field is by default
normalized to conventional tide-free — a field with no tidal contributions in
$\bar{C}_{20}$ — so that tidal effects can be explicitly layered back on
top.** (With no `TidesConfiguration` at all, the model's $\bar{C}_{20}$ is
left exactly as published.) The solid Earth tide model computes the *total* tidal
contribution, including its time average (the permanent part), so a tide-free
background is exactly what it composes with; stacking it on a zero-tide or
mean-tide field would count the permanent part twice.

This layering is why `Auto` always targets tide-free, regardless of which
tidal force models are enabled. If you do not want the normalization — for
example, you propagate without solid tides and prefer to keep a zero-tide
model's $\bar{C}_{20}$ as the better time-average field — set the permanent
handling to `Off`, or use `ConvertTo` to select a specific system. Both are
valid even when no tidal force model is enabled.

## Permanent Tide Configuration

`PermanentTideConfig` controls how Brahe reconciles the loaded model's
$\bar{C}_{20}$ with the conventional tide-free convention:

| Variant | Behavior |
|---|---|
| `Auto` *(default)* | Reads the tide-system flag stored in the model file and converts $\bar{C}_{20}$ to conventional tide-free automatically. If the flag is `Unknown`, no conversion is applied (a warning is emitted). |
| `ConvertTo(system)` | Forces the model into the specified tide system. Errors at propagator construction if the stored flag is `Unknown`. |
| `Off` | Leaves $\bar{C}_{20}$ untouched. Use when you have pre-corrected the model yourself, or for debugging. |

`Auto` is the right choice for almost all practical cases.

!!! warning "Inconsistent combination: `ConvertTo(ZeroTide/MeanTide)` + solid tides"
    Converting the background field to zero-tide or mean-tide while solid Earth
    tides are enabled double-counts the permanent tide (once in the static
    $\bar{C}_{20}$, once in the tide model's time average). Brahe emits a
    warning for this combination — at propagator construction in Rust, and as a
    suppressible `UserWarning` when constructing `TidesConfiguration` in
    Python. For gravity models owned by the propagator
    (`GravityModelSource.ModelType`) the conversion is still applied as
    requested, since Step-3-style workflows that pre-subtract the permanent
    part externally are legitimate.

Shared global gravity models (`GravityModelSource.Global`) are read-only across
every propagator that references them, so their permanent-tide handling is
resolved once, when the model is installed as the global — not per propagator.
The propagator trusts the global model's tide system as-is and never mutates
shared state, so the `PermanentTideConfig` in a `Global`-source force model has
no effect. Set the tide system on the global model up front, either with the
convenience setter `set_global_gravity_model_to_tide_system(model, target)` or
by calling `GravityModel.convert_tide_system` before `set_global_gravity_model`:

=== "Python"

    ```python
    --8<-- "./examples/numerical_propagation/tides_global_tide_system.py:3"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/numerical_propagation/tides_global_tide_system.rs:3"
    ```

## Solid Earth Tides

The solid Earth tide model adds time-varying corrections $\Delta\bar{C}_{nm}$
and $\Delta\bar{S}_{nm}$ to the static geopotential coefficients at each
integration step.

### Static Correction (Always On)

When solid tides are enabled, the static correction is always computed. For each tide-raising
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

### Time-Varying Correction (Optional)

Setting `frequency_dependent=True` on `SolidTideConfig` activates the IERS
time-varying correction tables (Tables 6.5a/b/c). These corrections account for the frequency
dependence of the anelastic Love numbers near tidal resonances, primarily
affecting the degree-2 coefficients:

| Quantity | Equation |
|---|---|
| $\Delta\bar{C}_{20}$ | Eq. 6.8a (real part, $m=0$, 21 lines) |
| $\Delta\bar{C}_{21}, \Delta\bar{S}_{21}$ | Eq. 6.8b ($m=1$, 48 lines) |
| $\Delta\bar{C}_{22}, \Delta\bar{S}_{22}$ | Eq. 6.8c ($m=2$, 2 lines) |

Each correction line is computed from Doodson/Delaunay arguments
$\theta_f = m(\theta_G + \pi) - (n_l l + n_{l'} l' + n_F F + n_D D + n_\Omega \Omega)$
(IERS §6.2.1), where $\theta_G$ is GMST and $l, l', F, D, \Omega$ are the five
Delaunay fundamental arguments evaluated at the current TT epoch.

Time-varying corrections are at the $10^{-10}$–$10^{-11}$ level, contributing a
sub-millimetre position effect per orbit for LEO satellites. They are
recommended for precise orbit determination but can be omitted for most
mission-analysis applications.

## Configuring Tides

### Permanent Tide Only

Normalizes the static field to conventional tide-free without adding any
time-varying tidal accelerations. Use this when you want a field with no tidal
contributions (e.g. for consistency across models, or before layering your own
tide model). If you instead want to keep a zero-tide model's $\bar{C}_{20}$
as-is, use `PermanentTideConfig.OFF`.

=== "Python"

    ```python
    --8<-- "./examples/numerical_propagation/tides_permanent_only.py:3"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/numerical_propagation/tides_permanent_only.rs:3"
    ```

### Permanent + Static Solid Earth Tide

=== "Python"

    ```python
    --8<-- "./examples/numerical_propagation/tides_static.py:3"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/numerical_propagation/tides_static.rs:3"
    ```

### Permanent + Static + Time-Varying Solid Earth Tide

=== "Python"

    ```python
    --8<-- "./examples/numerical_propagation/tides_static_time_varying.py:3"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/numerical_propagation/tides_static_time_varying.rs:3"
    ```

## Worked Example

The example below propagates a 500 km LEO satellite for one full orbital period
with tides enabled and disabled, then prints the peak position difference.

=== "Python"

    ```python
    --8<-- "./examples/numerical_propagation/force_model_tides.py:10"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/numerical_propagation/force_model_tides.rs:5"
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

## Ephemeris Source Notes

Solid-tide computation requires Sun and Moon positions in the ECEF frame. The
propagator computes these internally using its own low-precision analytical
ephemeris (`sun_position` / `moon_position`) — it does **not** read positions
from the `third_bodies` force model entries. Enabling or disabling `third_bodies` entries has
no effect on the tidal correction; you can use solid tides with or without a
third-body configuration.

## See Also

- [Gravity Models](gravity.md) — tide systems of packaged models, `GravityModelTideSystem`
- [Force Models](../orbit_propagation/numerical_propagation/force_models.md) — wiring the full force model

## References

- Petit, G., & Luzum, B. (Eds.) (2010). *IERS Conventions (2010)*, Technical Note 36, Chapter 6: Geopotential. IERS. <https://iers-conventions.obspm.fr/content/chapter6/icc6.pdf>
- Petit, G., & Luzum, B. (Eds.) (2010). *IERS Conventions (2010)*, Technical Note 36, Chapter 1: General Definitions and Numerical Standards. IERS. (§1.1 tide-system definitions.) <https://iers-conventions.obspm.fr/content/chapter1/icc1.pdf>
- Lemoine, F. G., et al. (1998). *The Development of the Joint NASA GSFC and the National Imagery and Mapping Agency (NIMA) Geopotential Model EGM96*. NASA/TP-1998-206861. (§11.1: tide-system definitions, direct/indirect terminology, and permanent-tide conversion formulas.) <https://cddis.nasa.gov/926/egm96/doc/S11.HTML>
