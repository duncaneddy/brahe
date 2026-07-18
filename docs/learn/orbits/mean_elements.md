# Mean Elements

Osculating Keplerian elements describe the exact, instantaneous orbit an object would follow
if all perturbations vanished at that instant. Mean elements remove the short-period
oscillations caused by those perturbations, leaving the slowly-varying part of the motion.
Mean elements are the natural input for orbit design, station-keeping budgets, and any
comparison against catalog data (e.g. TLEs) that is itself defined in mean elements.

The `brahe.orbits` module provides two independent methods for converting between mean and
osculating Keplerian elements `[a, e, i, Ω, ω, anomaly]`, selected via the `MeanElementMethod`
enum:

- **`MeanElementMethod.BROUWER_LYDDANE`** — first-order analytical $J_2$ mapping (Brouwer-Lyddane
  theory), evaluated independently at each state.
- **`MeanElementMethod.numerical(config)`** — windowed averaging of a numerically propagated
  trajectory, which captures whatever perturbations are present in the force model used to
  generate that trajectory (not just $J_2$).

For complete API documentation, see the [Mean Elements API Reference](../../library_api/orbits/mean_elements.md).

## Brouwer-Lyddane (Analytical)

`state_koe_osc_to_mean` and `state_koe_mean_to_osc` apply the first-order Brouwer-Lyddane $J_2$
mapping[^1][^2] to a single Keplerian state. Because the mapping is a closed-form function of the
state alone, it works pointwise — no trajectory history is required — and is the default choice
for LEO orbits where $J_2$ dominates the short-period variation.

=== "Python"

    ``` python
    --8<-- "./examples/orbits/mean_osculating_analytical.py:14"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbits/mean_osculating_analytical.rs:9"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/orbits/mean_osculating_analytical.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/orbits/mean_osculating_analytical.rs.txt"
        ```

Because the transformation is a first-order truncation of an infinite series, round-tripping
through it (`mean → osc → mean`) recovers the original state only to $O(J_2^2)$: a residual set by
the truncated higher-order terms of the series remains, growing with eccentricity and inclination.

`MeanElementMethod::Numerical` is **batch-only**: calling `state_koe_osc_to_mean` or
`state_koe_mean_to_osc` with it on a single state returns an error. Use the batch functions below.

## Numerical (Windowed Averaging)

`batch_state_koe_osc_to_mean` and `batch_state_koe_mean_to_osc`, given
`MeanElementMethod.numerical(config)`, average an osculating trajectory over a moving window to
produce mean elements that reflect whatever dynamics were used to generate the input trajectory —
not just $J_2$. This is the appropriate method when higher-fidelity perturbations (drag, third-body,
tides, higher-order gravity) meaningfully affect the short-period content of the orbit and an
analytical mapping would not capture it.

The example below synthesizes one period of osculating states analytically (by evaluating the
Brouwer-Lyddane mean→osc mapping at a sweep of mean anomalies, with no numerical propagation
involved) and then recovers a mean state by averaging that trajectory over a centered window
spanning the full period:

=== "Python"

    ``` python
    --8<-- "./examples/orbits/mean_osculating_numerical.py:15"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbits/mean_osculating_numerical.rs:10"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/orbits/mean_osculating_numerical.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/orbits/mean_osculating_numerical.rs.txt"
        ```

### Averaging in Equinoctial Space

Averaging is performed in equinoctial elements `[a, h, k, p, q, l]` rather than directly on
Keplerian elements, because equinoctial elements remove the zero-eccentricity singularity where
Keplerian angles like $\Omega$ and $\omega$ become ill-defined or fast-varying. The retrograde
factor is selected per window from the orbit's inclination, so the appropriate chart is regular
for the orbit being averaged. Within a window, the slowly-varying components $a, h, k, p, q$ are
arithmetic-averaged, while the fast mean longitude $l$ is linearly detrended and evaluated at the
window's anchor epoch, before the averaged state is converted back to Keplerian elements. See
[Equinoctial Elements](../../library_api/orbits/equinoctial.md) for the element definitions and
conversion functions used internally.

The slow components are averaged as an unweighted sample mean, which approximates a true
time-average when the input samples are (approximately) uniformly spaced in time — the expected
case for a propagated trajectory. Input epochs must be strictly ascending; markedly non-uniform
cadence weights densely-sampled portions of the window more heavily.

### Window Length, Alignment, and Edge Handling

`MeanElementNumericalMethodConfig` controls the averaging window:

- **`window_seconds`** — the window length $W$, in seconds. Typically set to one orbital period
  (`orbital_period(a)`) so the average spans a full revolution.
- **`alignment`** (`WindowAlignment`) — placement of the window relative to each output epoch $t$:
    - `Centered`: $[t - W/2,\ t + W/2]$
    - `Trailing`: $[t - W,\ t]$ (causal — uses only past data)
    - `Leading`: $[t,\ t + W]$ (uses only future data)
- **`edge`** (`WindowEdgeHandling`) — how to handle output epochs whose window extends past the input
  trajectory's time bounds:
    - `Truncate`: drop unsupported output epochs. The returned `(epoch, state)` list is shorter
      than the input; each surviving pair's epoch identifies which input sample it maps to; there is
      no positional correspondence between input and output indices once epochs are dropped.
    - `PreserveWindow`: keep the window length fixed and slide its anchor inside the data bounds
      instead of shrinking it. Output length always equals input length. (If the entire input
      trajectory is shorter than $W$, the window is clamped to the available data span, which is
      then narrower than $W$.)

### Iterative Mean-to-Osculating Inverse

Unlike osculating→mean (a direct average), mean→osculating with the numerical method has no
closed form: it must search for the osculating state whose forward-averaged mean matches the
target. `numerical(config)` therefore requires `MeanElementNumericalMethodConfig.inverse` (a
`MeanElementInverseConfig`) when used for mean→osc; passing `inverse=None` for that direction
raises an error (`inverse` is unused for osc→mean).

`MeanElementInverseConfig` supplies the dynamics used to test each trial osculating state:

- **`force_model`** (`ForceModelConfig`) — force model used to numerically propagate the trial state
  across the averaging window.
- **`propagation`** (`NumericalPropagationConfig`) — integrator settings for that trial propagation.
- **`tolerance`** — convergence tolerance on the mean-element residual. The residual is reduced to a
  single scalar via the mixed norm $|\Delta a|\,(\text{m}) + 10^6\,|\Delta e| + 10^6\,\lVert\Delta(i, \Omega, \omega, M)\rVert\,(\text{rad})$,
  where the $10^6$ weights make eccentricity and angle errors commensurate with a semi-major-axis
  error in meters. For example, `tolerance = 1.0` demands roughly 1 m in $a$, $10^{-6}$ in $e$, and
  $10^{-6}$ rad (~0.2 arcsec) across the angles combined.
- **`max_iterations`** — maximum number of differential-correction iterations before giving up
  (returns an error if exceeded).

The solver seeds its first guess from the analytical Brouwer-Lyddane inverse, then iterates:
propagate the trial state, average the result back down with `forward_average`, and apply the
observed mean-element residual as a fixed-point correction, until the residual falls below
`tolerance` or `max_iterations` is exhausted.

!!! note "Convergence near singular geometries"
    The correction is a fixed-point step applied directly to Keplerian elements, whose angles
    $\Omega$, $\omega$, and $M$ are ill-conditioned near zero eccentricity and near-equatorial or
    critical ($\approx 63.4°/116.6°$) inclinations. The analytical seed shares these weaknesses.
    For such geometries the solver may need more iterations or fail to converge — in which case it
    returns an error rather than a silently biased result.

=== "Python"

    ``` python
    --8<-- "./examples/orbits/mean_osculating_inverse.py:15"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/orbits/mean_osculating_inverse.rs:10"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/orbits/mean_osculating_inverse.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/orbits/mean_osculating_inverse.rs.txt"
        ```

## Limitations

The numerical method is a first implementation with a known limitation that affects robustness
only in specific regimes and is a candidate for future improvement:

- **Iterative inverse near singular geometries.** The mean → osculating inverse applies its
  fixed-point correction directly to Keplerian elements and seeds from the analytical
  Brouwer-Lyddane mapping. Both are ill-conditioned near zero eccentricity and near-equatorial or
  critical ($\approx 63.4°/116.6°$) inclinations, so the solver may require more iterations or fail
  to converge there. It returns an error rather than a silently biased result, but a correction
  formulated in equinoctial elements (with a proper Jacobian) would extend robust convergence into
  these regimes.

The slow equinoctial components (`a, h, k, p, q`) are combined with a trapezoidal time-weighted
average over the window, so the result is independent of input sampling cadence (uniform or not).

## Function Reference

| Conversion | Scope | Function |
|---|---|---|
| Osculating → mean | Single state | [`state_koe_osc_to_mean`](../../library_api/orbits/mean_elements.md#brahe.orbits.state_koe_osc_to_mean) |
| Mean → osculating | Single state | [`state_koe_mean_to_osc`](../../library_api/orbits/mean_elements.md#brahe.orbits.state_koe_mean_to_osc) |
| Osculating → mean | Batch | [`batch_state_koe_osc_to_mean`](../../library_api/orbits/mean_elements.md#brahe.orbits.batch_state_koe_osc_to_mean) |
| Mean → osculating | Batch | [`batch_state_koe_mean_to_osc`](../../library_api/orbits/mean_elements.md#brahe.orbits.batch_state_koe_mean_to_osc) |

All functions operate on Keplerian elements `[a, e, i, Ω, ω, anomaly]` in SI units (`a` in meters)
with angles in the format specified by `AngleFormat`.

---

## See Also

- [Mean Elements API Reference](../../library_api/orbits/mean_elements.md) - Complete API documentation
- [Equinoctial Elements](../../library_api/orbits/equinoctial.md) - Element set used internally by numerical averaging
- [Orbital Properties](properties.md) - `orbital_period` and other properties used to size averaging windows
- [Force Models Guide](../orbit_propagation/numerical_propagation/force_models.md) - Configuring `ForceModelConfig` for the numerical inverse
- [Trajectories](../trajectories/index.md) - `DOrbitTrajectory` and trajectory-based orbit representations

[^1]: Brouwer, D., "Solution of the Problem of Artificial Satellite Theory Without Drag," Astronautical Journal, Vol. 64, No. 1274, 1959, pp. 378-397.
[^2]: Lyddane, R. H., "Small Eccentricities or Inclinations in the Brouwer Theory of the Artificial Satellite," Astronomical Journal, Vol. 68, No. 8, 1963, pp. 555-558.
