# Tidal Corrections

Earth's oceans and solid body are elastically deformed by the gravitational
attraction of the Sun and Moon. This deformation causes time-varying changes
to the geopotential, producing small but measurable accelerations on low-Earth
orbit satellites. For a 500 km LEO satellite propagated over one orbital period,
solid Earth tides shift the position by roughly 1–2 m relative to a tide-free
model.

Brahe implements the solid Earth tide, ocean tide, and pole tide models from
Chapter 6 of the IERS Conventions (2010), Technical Note 36 (TN36):
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

### Solid Earth Pole Tide (Optional)

Setting `pole_tide=True` on `SolidTideConfig` adds the solid Earth pole tide
(IERS TN36 §6.4), the elastic deformation caused by the centrifugal effect of
polar motion. It contributes to $\bar{C}_{21}$ and $\bar{S}_{21}$ only:

$$
\Delta\bar{C}_{21} = -1.333\times10^{-9}\,(m_1 + 0.0115\,m_2), \qquad
\Delta\bar{S}_{21} = -1.333\times10^{-9}\,(m_2 - 0.0115\,m_1)
$$

The wobble parameters $(m_1, m_2)$ are the polar motion coordinates $(x_p,
y_p)$ relative to the IERS secular pole $(\bar{x}_s, \bar{y}_s)$:

$$
m_1 = x_p - \bar{x}_s, \qquad m_2 = -(y_p - \bar{y}_s)
$$

Brahe evaluates the secular pole with the updated linear model (IERS
Conventions §7.1.4, Eq. 21, version 2018/02/01), which supersedes the cubic
mean-pole model published in TN36 (2010) and is consistent with ITRF2014:

$$
\bar{x}_s = 55.0 + 1.677\,(t - 2000.0)\ \text{mas}, \qquad
\bar{y}_s = 320.5 + 3.460\,(t - 2000.0)\ \text{mas}
$$

with $t$ in Julian years of TT since J2000.0.

The solid Earth pole tide requires initialized global EOP data (for $x_p$,
$y_p$); enabling `pole_tide` without EOP initialized returns an error at
propagator construction.

## Ocean Tides

Ocean tides redistribute mass as the tidal bulge moves through the world's
oceans, producing a geopotential contribution distinct from, and generally
smaller than, the solid Earth tide. Brahe implements the FES2004 ocean tide
model (IERS TN36 §6.3):

$$
\Delta\bar{C}_{nm} = \sum_f \left[ (C_f^{+} + C_f^{-})\cos\theta_f
+ (S_f^{+} + S_f^{-})\sin\theta_f \right]
$$

$$
\Delta\bar{S}_{nm} = \sum_f \left[ (S_f^{+} - S_f^{-})\cos\theta_f
- (C_f^{+} - C_f^{-})\sin\theta_f \right]
$$

(the real form of Eq. 6.15, summed over tidal constituents $f$, exactly as
implemented), where
$C_f^{\pm}, S_f^{\pm}$ are the prograde (+) and retrograde (−) fully
normalized Stokes-coefficient amplitudes of constituent $f$, and $\theta_f$ is
its Doodson/Delaunay tidal argument (§6.2.1) — the same argument construction
used by the solid-tide time-varying correction.

### Coefficient Download

FES2004 coefficients are not bundled with Brahe. The first time a propagator
is constructed with ocean tides enabled, Brahe downloads the IERS
coefficient file (`fes2004_Cnm-Snm.dat`, ~3.7 MB) into `$BRAHE_CACHE/tides/`
and reuses the cached copy on every subsequent call; no network access is
needed once the file is cached. Degree-1 rows present in the file are skipped:
they represent geocenter motion, not geopotential coefficients about the
Earth's center of mass.

### Degree, Order, and Admittance

`OceanTideConfig.degree` / `.order` truncate the expansion (2–100, default
20). The FES2004 file tabulates 18 main tidal constituents directly; setting
`include_admittance=True` (the default) completes the model with the ~63
secondary constituents of Table 6.7, obtained by linear admittance
interpolation between neighboring main waves (Eq. 6.16). The admittance-wave
coefficients are constant linear combinations of the main-wave coefficients,
computed once when the model is loaded.

### Ocean Pole Tide

Setting `pole_tide=True` on `OceanTideConfig` adds the dominant $(2,1)$ term
of the ocean's self-consistent equilibrium response to polar motion (IERS
TN36 §6.5, Eq. 6.24), which captures roughly 90% of the ocean pole tide
potential variance:

$$
\Delta\bar{C}_{21} = -2.1778\times10^{-10}\,(m_1 - 0.01724\,m_2), \qquad
\Delta\bar{S}_{21} = -1.7232\times10^{-10}\,(m_2 - 0.03365\,m_1)
$$

using the same wobble parameters $(m_1, m_2)$ and secular pole as the solid
Earth pole tide above. Like the solid pole tide, it requires initialized
global EOP data.

## Configuring Tides

!!! note "Loading configurations saved before ocean tides"
    `TidesConfiguration` gained an `ocean` field when ocean tides were added.
    Configurations serialized before this change deserialize without error,
    with `ocean` defaulting to `None` — ocean tides stay disabled unless the
    field is set explicitly.

### Sun and Moon Ephemeris Source

The tidal corrections are driven by the Sun and Moon positions. The
`ephemeris_source` field of `TidesConfiguration` selects how those positions
are computed and defaults to `EphemerisSource.LowPrecision` (the analytic
geocentric ephemerides), which is accurate enough for the ~$10^{-7}$ m/s²
tidal perturbation. Set it to a high-precision source
(`EphemerisSource.DE440s`/`DE440`) to match a third-body perturbation
configured against the same source — when the sources match, the propagator
evaluates the Sun and Moon positions once per epoch and shares them between the
tidal and third-body force terms. `ForceModelConfig.high_fidelity()` sets
`ephemeris_source=EphemerisSource.DE440s` for exactly this consistency with its
third-body configuration.

A configuration serialized before this field existed deserializes without
error, defaulting to `EphemerisSource.LowPrecision`.

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

### Full Tide Model

Enables every tidal correction Brahe supports: static and time-varying solid
Earth tides, the solid Earth pole tide, and FES2004 ocean tides (with
admittance and the ocean pole tide) to degree/order 30. This is the same
tide configuration used internally by `ForceModelConfig.high_fidelity()`.
Building the configuration does not touch the network; the FES2004 download
(see [Ocean Tides](#ocean-tides)) happens once, the first time a propagator
with ocean tides enabled is constructed.

=== "Python"

    ```python
    --8<-- "./examples/numerical_propagation/tides_full.py:9"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/numerical_propagation/tides_full.rs:6"
    ```

## Worked Example

The example below propagates a 500 km LEO satellite for one full orbital period
with tides enabled and disabled, then prints the peak position difference.

=== "Python"

    ```python
    --8<-- "./examples/numerical_propagation/force_model_tides.py:16"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/numerical_propagation/force_model_tides.rs:12"
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

## Evaluation Cost

When gravity is configured as spherical harmonics, the propagator folds the
static $\bar{C}_{nm}/\bar{S}_{nm}$ coefficients together with every enabled
tidal correction (solid Step 1/2, both pole tides, and the ocean tide) into a
single packed coefficient table, evaluated with one Clenshaw pass per
dynamics call. This fold-in is exact by linearity of the spherical-harmonic
sum: the marginal cost of enabling tides is only the per-epoch
$\Delta\bar{C}_{nm}/\Delta\bar{S}_{nm}$ computation, not a second gravity
evaluation. Point-mass and zonal gravity configurations still get a
delta-only field, since there is no static spherical-harmonic table to fold
into.

## Ephemeris Source Notes

Solid-tide computation requires Sun and Moon positions in the ECEF frame. The
propagator computes these internally using its own low-precision analytical
ephemeris (`sun_position` / `moon_position`) — it does **not** read positions
from the `third_body` force model. Enabling or disabling `third_body` bodies has
no effect on the tidal correction; you can use solid tides with or without a
third-body configuration.

## See Also

- [Gravity Models](gravity.md) — tide systems of packaged models, `GravityModelTideSystem`
- [Force Models](../orbit_propagation/numerical_propagation/force_models.md) — wiring the full force model

## References

- Petit, G., & Luzum, B. (Eds.) (2010). *IERS Conventions (2010)*, Technical Note 36, Chapter 6: Geopotential. IERS. <https://iers-conventions.obspm.fr/content/chapter6/icc6.pdf>
- Petit, G., & Luzum, B. (Eds.) (2010). *IERS Conventions (2010)*, Technical Note 36, Chapter 1: General Definitions and Numerical Standards. IERS. (§1.1 tide-system definitions.) <https://iers-conventions.obspm.fr/content/chapter1/icc1.pdf>
- Petit, G., & Luzum, B. (Eds.) (2010). *IERS Conventions (2010)*, Technical Note 36, Chapter 7: Displacement of Reference Points (§7.1.4, secular pole; updated for the linear secular-pole model). IERS. <https://iers-conventions.obspm.fr/content/chapter7/icc7.pdf>
- Lemoine, F. G., et al. (1998). *The Development of the Joint NASA GSFC and the National Imagery and Mapping Agency (NIMA) Geopotential Model EGM96*. NASA/TP-1998-206861. (§11.1: tide-system definitions, direct/indirect terminology, and permanent-tide conversion formulas.) <https://cddis.nasa.gov/926/egm96/doc/S11.HTML>
- Lyard, F., Lefèvre, F., Letellier, T., & Francis, O. (2006). Modelling the global ocean tides: modern insights from FES2004. *Ocean Dynamics*, 56, 394–415. (FES2004 ocean tide model.)
- Desai, S. D. (2002). Observing the pole tide with satellite altimetry. *Journal of Geophysical Research*, 107(C11), 3186. (Ocean pole tide equilibrium model.)
