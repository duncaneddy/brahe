# Mean Elements

Osculating Keplerian elements describe the exact, instantaneous orbit an object would follow
if all perturbations vanished at that instant. Mean elements remove the short-period
oscillations caused by those perturbations, leaving the slowly-varying part of the motion.
Mean elements are the natural input for orbit design, station-keeping budgets, and any
comparison against catalog data (e.g. TLEs) that is itself defined in mean elements.

The `brahe.orbits` module provides two independent methods for converting between mean and
osculating Keplerian elements `[a, e, i, Ω, ω, anomaly]`, selected via the `MeanElementMethod`
enum:

- **`MeanElementMethod.brouwer_lyddane()`** — first-order analytical $J_2$ mapping (Brouwer-Lyddane
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

    ```python
    import brahe as bh
    import numpy as np

    mean = np.array([bh.R_EARTH + 500e3, 0.001, 45.0, 0.0, 0.0, 0.0])
    osc = bh.state_koe_mean_to_osc(
        mean, bh.MeanElementMethod.brouwer_lyddane(), bh.AngleFormat.DEGREES
    )
    ```

=== "Rust"

    ```rust
    use brahe::constants::{AngleFormat, R_EARTH};
    use brahe::orbits::{state_koe_mean_to_osc, MeanElementMethod};
    use nalgebra::SVector;

    let mean = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.001, 45.0, 0.0, 0.0, 0.0);
    let osc = state_koe_mean_to_osc(&mean, MeanElementMethod::BrouwerLyddane, AngleFormat::Degrees)
        .unwrap();
    ```

Because the transformation is a first-order truncation of an infinite series, round-tripping
through it (`mean → osc → mean`) recovers the original state only to $O(J_2^2)$: expect residuals
on the order of $\pm 100$ m in semi-major axis and $\pm 0.01$ rad ($\approx 0.6°$) in the angular
elements for typical LEO orbits.

`MeanElementMethod::Numerical` is **batch-only**: calling `state_koe_osc_to_mean` or
`state_koe_mean_to_osc` with it on a single state returns an error. Use the batch functions below.

## Numerical (Windowed Averaging)

`batch_state_koe_osc_to_mean` and `batch_state_koe_mean_to_osc`, given
`MeanElementMethod.numerical(config)`, average an osculating trajectory over a moving window to
produce mean elements that reflect whatever dynamics were used to generate the input trajectory —
not just $J_2$. This is the appropriate method when higher-fidelity perturbations (drag, third-body,
tides, higher-order gravity) meaningfully affect the short-period content of the orbit and an
analytical mapping would not capture it.

### Averaging in Equinoctial Space

Averaging is performed in equinoctial elements `[a, h, k, p, q, l]` rather than directly on
Keplerian elements, because equinoctial elements are singularity-free at $e \to 0$ and $i \to 0$
(or $i \to 180°$) — exactly the regimes where Keplerian angles like $\Omega$ and $\omega$ become
ill-defined or fast-varying. Within a window, the slowly-varying components $a, h, k, p, q$ are
arithmetic-averaged, while the fast mean longitude $l$ is linearly detrended and evaluated at the
window's anchor epoch, before the averaged state is converted back to Keplerian elements. See
[Equinoctial Elements](../../library_api/orbits/equinoctial.md) for the element definitions and
conversion functions used internally.

### Window Length, Alignment, and Edge Handling

`NumericalConfig` controls the averaging window:

- **`window_seconds`** — the window length $W$, in seconds. Typically set to one orbital period
  (`orbital_period(a)`) so the average spans a full revolution.
- **`alignment`** (`WindowAlignment`) — placement of the window relative to each output epoch $t$:
    - `Centered`: $[t - W/2,\ t + W/2]$
    - `Trailing`: $[t - W,\ t]$ (causal — uses only past data)
    - `Leading`: $[t,\ t + W]$ (uses only future data)
- **`edge`** (`EdgeHandling`) — how to handle output epochs whose window extends past the input
  trajectory's time bounds:
    - `Truncate`: drop unsupported output epochs. The returned `(epoch, state)` list is shorter
      than the input; each surviving pair's epoch identifies which input sample it maps to; there is
      no positional correspondence between input and output indices once epochs are dropped.
    - `PreserveWindow`: keep the window length fixed and slide its anchor inside the data bounds
      instead of shrinking it. Output length always equals input length.

### Iterative Mean-to-Osculating Inverse

Unlike osculating→mean (a direct average), mean→osculating with the numerical method has no
closed form: it must search for the osculating state whose forward-averaged mean matches the
target. `numerical(config)` therefore requires `NumericalConfig.inverse` (an `InverseConfig`) when
used for mean→osc; passing `inverse=None` for that direction raises an error (`inverse` is unused
for osc→mean).

`InverseConfig` supplies the dynamics used to test each trial osculating state:

- **`force_model`** (`ForceModelConfig`) — force model used to numerically propagate the trial state
  across the averaging window.
- **`propagation`** (`NumericalPropagationConfig`) — integrator settings for that trial propagation.
- **`tolerance`** — convergence tolerance on the mean-element residual.
- **`max_iterations`** — maximum number of differential-correction iterations before giving up
  (returns an error if exceeded).

The solver seeds its first guess from the analytical Brouwer-Lyddane inverse, then iterates:
propagate the trial state, average the result back down with `forward_average`, and apply the
observed mean-element residual as a fixed-point correction, until the residual falls below
`tolerance` or `max_iterations` is exhausted.

=== "Python"

    ```python
    import brahe as bh
    import numpy as np

    mean = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 0.0])
    period = bh.orbital_period(mean[0])

    inverse = bh.InverseConfig(
        bh.ForceModelConfig.earth_gravity(),
        bh.NumericalPropagationConfig.default(),
        1.0,   # tolerance
        25,    # max_iterations
    )
    config = bh.NumericalConfig(
        period, bh.WindowAlignment.CENTERED, bh.EdgeHandling.PRESERVE_WINDOW, inverse
    )

    epochs = [bh.Epoch.from_gps_seconds(0.0)]
    osc = bh.batch_state_koe_mean_to_osc(
        epochs, mean.reshape(1, 6), bh.MeanElementMethod.numerical(config), bh.AngleFormat.DEGREES
    )
    ```

=== "Rust"

    ```rust
    use brahe::constants::{AngleFormat, R_EARTH};
    use brahe::orbits::{
        batch_state_koe_mean_to_osc, orbital_period, EdgeHandling, InverseConfig,
        MeanElementMethod, NumericalConfig, WindowAlignment,
    };
    use brahe::propagators::{ForceModelConfig, NumericalPropagationConfig};
    use brahe::time::Epoch;
    use nalgebra::SVector;

    let mean = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 0.0);
    let period = orbital_period(mean[0]);

    let inverse = InverseConfig {
        force_model: ForceModelConfig::earth_gravity(),
        propagation: NumericalPropagationConfig::default(),
        tolerance: 1.0,
        max_iterations: 25,
    };
    let config = NumericalConfig {
        window_seconds: period,
        alignment: WindowAlignment::Centered,
        edge: EdgeHandling::PreserveWindow,
        inverse: Some(inverse),
    };

    let epochs = vec![Epoch::from_gps_seconds(0.0)];
    let osc = batch_state_koe_mean_to_osc(
        &epochs, &[mean], MeanElementMethod::Numerical(config), AngleFormat::Degrees,
    )
    .unwrap();
    ```

## Trajectory Adapters (Rust)

`batch_state_koe_osc_to_mean_trajectory` and `batch_state_koe_mean_to_osc_trajectory` wrap the
batch functions above to accept a `DOrbitTrajectory` directly, converting it to Keplerian elements
internally before dispatching to the same `(epochs, states, method, angle_format)` core. They
accept the same `MeanElementMethod` values and follow the same truncation/preservation semantics
described above.

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
- [Trajectories](../trajectories/index.md) - `DOrbitTrajectory` and the trajectory adapters

[^1]: Brouwer, D., "Solution of the Problem of Artificial Satellite Theory Without Drag," Astronautical Journal, Vol. 64, No. 1274, 1959, pp. 378-397.
[^2]: Lyddane, R. H., "Small Eccentricities or Inclinations in the Brouwer Theory of the Artificial Satellite," Astronomical Journal, Vol. 68, No. 8, 1963, pp. 555-558.
