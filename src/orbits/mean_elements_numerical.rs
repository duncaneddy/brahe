/*!
 * Numerical mean-element averaging.
 *
 * Osculating states are averaged over a moving window in equinoctial space
 * (singularity-safe at e→0, i→0). Slow components a,h,k,p,q are trapezoidally
 * time-weighted over the window (cadence-independent); the fast mean-longitude l is
 * linearly detrended and evaluated at the anchor epoch.
 */

use crate::constants::AngleFormat;
use crate::coordinates::{state_eci_to_koe, state_koe_to_eci};
use crate::orbits::equinoctial::{state_equinoctial_to_koe, state_koe_to_equinoctial};
use crate::orbits::mean_elements::{
    MeanElementInverseConfig, MeanElementNumericalMethodConfig, WindowAlignment, WindowEdgeHandling,
};
use crate::orbits::{MeanElementMethod, state_koe_mean_to_osc};
use crate::propagators::DNumericalOrbitPropagator;
use crate::propagators::traits::{DStatePropagator, DStateProvider};
use crate::time::Epoch;
use crate::utils::errors::BraheError;
use nalgebra::{DVector, SVector};

/// Computes the averaging window bounds for a given anchor epoch, applying edge handling.
///
/// # Arguments
///
/// * `anchor` - Output epoch at which the window is centered/trailing/leading
/// * `data_start` - First epoch present in the input trajectory
/// * `data_end` - Last epoch present in the input trajectory
/// * `config` - Numerical averaging configuration (window length, alignment, edge handling)
///
/// # Returns
///
/// `Some((start, end))` window bounds in seconds-comparable [`Epoch`]s, or `None` when
/// `config.edge` is [`WindowEdgeHandling::Truncate`] and the window would extend past the data
/// bounds.
// Not yet called from non-test production code; wired up by a later task.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn window_bounds(
    anchor: Epoch,
    data_start: Epoch,
    data_end: Epoch,
    config: &MeanElementNumericalMethodConfig,
) -> Option<(Epoch, Epoch)> {
    let w = config.window_seconds;
    let (mut start, mut end) = match config.alignment {
        WindowAlignment::Centered => (anchor - w / 2.0, anchor + w / 2.0),
        WindowAlignment::Trailing => (anchor - w, anchor),
        WindowAlignment::Leading => (anchor, anchor + w),
    };
    let underflow = data_start - start; // >0 if start is before data
    let overflow = end - data_end; // >0 if end is after data
    if underflow > 0.0 || overflow > 0.0 {
        match config.edge {
            WindowEdgeHandling::Truncate => return None,
            WindowEdgeHandling::PreserveWindow => {
                // Slide the fixed-length window inside the data bounds.
                if underflow > 0.0 {
                    start = data_start;
                    end = start + w;
                }
                if end - data_end > 0.0 {
                    end = data_end;
                    start = end - w;
                }
                if start - data_start < 0.0 {
                    start = data_start; // window longer than data: clamp both
                }
            }
        }
    }
    Some((start, end))
}

/// Numerically averages osculating Keplerian states into mean Keplerian elements.
///
/// For each input epoch, gathers samples within the configured averaging window,
/// converts each osculating sample to equinoctial elements, trapezoidally
/// time-weights the slow components `a, h, k, p, q` over the window (so the result is
/// independent of sampling cadence), and linearly detrends the fast mean longitude `l`
/// to the anchor epoch. The averaged equinoctial state is converted back to Keplerian
/// elements.
///
/// # Arguments
///
/// * `epochs` - Sample epochs of the osculating trajectory (ascending order)
/// * `states_rad` - Osculating Keplerian states `[a, e, i, raan, argp, M]`, radians
///   (`a` in meters), one per epoch
/// * `config` - Numerical averaging configuration (window length, alignment, edge handling)
///
/// # Returns
///
/// Vector of `(epoch, mean_state)` pairs, mean states in radians. Output epochs whose
/// window lacks full data support are dropped when `config.edge` is
/// [`WindowEdgeHandling::Truncate`].
///
/// # Errors
///
/// Returns [`BraheError::Error`] if `epochs` and `states_rad` have unequal length, or if
/// `config.window_seconds` is not positive.
pub(crate) fn numerical_osc_to_mean(
    epochs: &[Epoch],
    states_rad: &[SVector<f64, 6>],
    config: &MeanElementNumericalMethodConfig,
) -> Result<Vec<(Epoch, SVector<f64, 6>)>, BraheError> {
    if epochs.len() != states_rad.len() {
        return Err(BraheError::Error(
            "epochs and states must have equal length".to_string(),
        ));
    }
    if epochs.is_empty() {
        return Ok(Vec::new());
    }
    if !config.window_seconds.is_finite() || config.window_seconds <= 0.0 {
        return Err(BraheError::Error(
            "window_seconds must be finite and positive".to_string(),
        ));
    }
    for w in epochs.windows(2) {
        if (w[1] - w[0]) <= 0.0 {
            return Err(BraheError::Error(
                "epochs must be strictly ascending".to_string(),
            ));
        }
    }
    let data_start = epochs[0];
    let data_end = epochs[epochs.len() - 1];

    let mut out = Vec::new();
    for (anchor_idx, &anchor) in epochs.iter().enumerate() {
        let Some((start, end)) = window_bounds(anchor, data_start, data_end, config) else {
            continue;
        };
        // Gather sample indices inside [start, end].
        let idx: Vec<usize> = epochs
            .iter()
            .enumerate()
            .filter(|&(_, &e)| (e - start) >= 0.0 && (end - e) >= 0.0)
            .map(|(j, _)| j)
            .collect();
        if idx.is_empty() {
            continue;
        }

        // Choose a single retrograde factor for this window, derived from the anchor
        // sample's inclination, and use it consistently for every forward conversion
        // and the final inverse conversion below. `tan(i/2)^fr` is only invertible when
        // the same `fr` is used both ways; switching sign only near the i=180 deg
        // singularity keeps prograde/polar/sun-synchronous orbits at fr=+1.
        let anchor_inclination = states_rad[anchor_idx][2];
        let fr: i8 = if anchor_inclination > std::f64::consts::FRAC_PI_2 * 1.999 {
            -1
        } else {
            1
        };

        // Convert each sample to equinoctial; trapezoidally time-average a,h,k,p,q;
        // detrend l.
        let mut ts = Vec::with_capacity(idx.len());
        let mut ls = Vec::with_capacity(idx.len());
        let mut slow = Vec::with_capacity(idx.len()); // [a,h,k,p,q] per sample
        for &j in &idx {
            let eqn = state_koe_to_equinoctial(&states_rad[j], AngleFormat::Radians, fr);
            ts.push(epochs[j] - anchor); // seconds relative to anchor
            ls.push(eqn[5]);
            slow.push([eqn[0], eqn[1], eqn[2], eqn[3], eqn[4]]);
        }
        let avg = trapezoidal_time_average(&ts, &slow);

        // Linear fit of unwrapped l vs relative time, evaluated at anchor (t=0 => intercept).
        let l_at_anchor = detrended_fast_angle(&ts, &ls);

        let eqn_mean = SVector::<f64, 6>::new(avg[0], avg[1], avg[2], avg[3], avg[4], l_at_anchor);
        // Recover Keplerian using the same fr chosen for the forward conversions above.
        let koe = state_equinoctial_to_koe(&eqn_mean, AngleFormat::Radians, fr);
        out.push((anchor, koe));
    }
    Ok(out)
}

/// Inverts numerical mean-element averaging via fixed-point differential correction.
///
/// For each target mean state, seeds a candidate osculating state from the analytical
/// Brouwer-Lyddane inverse ([`state_koe_mean_to_osc`]), then iterates: propagate the
/// candidate state across a window bracketing the target epoch with the dynamics in
/// `config.inverse`, average the resulting trajectory back to mean elements with
/// [`numerical_osc_to_mean`], and correct the candidate by the angle-aware mean-element
/// residual. Iteration stops once a mixed-norm residual falls below
/// `config.inverse.tolerance`, or fails after `config.inverse.max_iterations`.
///
/// # Arguments
///
/// * `epochs` - Target epochs at which osculating states are desired
/// * `mean_states_rad` - Target mean Keplerian states `[a, e, i, raan, argp, M]`, radians
///   (`a` in meters), one per epoch
/// * `config` - Numerical averaging configuration; `config.inverse` supplies the
///   propagation dynamics and convergence settings and is required (must not be `None`)
///
/// # Returns
///
/// Vector of `(epoch, osc_state)` pairs, osculating Keplerian states in radians, one per
/// input epoch.
///
/// # Errors
///
/// Returns [`BraheError::Error`] if `config.inverse` is `None`, or if `epochs` and
/// `mean_states_rad` have unequal length. Returns [`BraheError::NumericalError`] if the
/// differential correction fails to converge within `max_iterations` for any target
/// epoch.
pub(crate) fn numerical_mean_to_osc(
    epochs: &[Epoch],
    mean_states_rad: &[SVector<f64, 6>],
    config: &MeanElementNumericalMethodConfig,
) -> Result<Vec<(Epoch, SVector<f64, 6>)>, BraheError> {
    let inverse = config.inverse.as_ref().ok_or_else(|| {
        BraheError::Error(
            "numerical mean-to-osc requires MeanElementNumericalMethodConfig.inverse (dynamics)"
                .to_string(),
        )
    })?;
    if epochs.len() != mean_states_rad.len() {
        return Err(BraheError::Error(
            "epochs and states must have equal length".to_string(),
        ));
    }
    if !config.window_seconds.is_finite() || config.window_seconds <= 0.0 {
        return Err(BraheError::Error(
            "window_seconds must be finite and positive".to_string(),
        ));
    }
    if !inverse.tolerance.is_finite() || inverse.tolerance <= 0.0 {
        return Err(BraheError::Error(
            "inverse.tolerance must be finite and positive".to_string(),
        ));
    }
    if inverse.max_iterations < 1 {
        return Err(BraheError::Error(
            "inverse.max_iterations must be at least 1".to_string(),
        ));
    }

    let mut out = Vec::with_capacity(epochs.len());
    for (k, &t) in epochs.iter().enumerate() {
        let target = mean_states_rad[k];

        // Seed with the analytical Brouwer-Lyddane inverse (radians in/out). This
        // divides by tan(i) and (1 - 5cos^2 i), which is singular at equatorial
        // (i=0) and critical (~63.435/116.565 deg) inclinations; fall back to
        // seeding from the target mean elements themselves if that produces a
        // non-finite candidate.
        let mut x = state_koe_mean_to_osc(
            &target,
            MeanElementMethod::BrouwerLyddane,
            AngleFormat::Radians,
        )?;
        if !x.iter().all(|v| v.is_finite()) {
            x = target;
        }

        let mut converged = false;
        for _ in 0..inverse.max_iterations {
            let m_k = forward_average(t, &x, config, inverse)?;
            let residual = element_residual(&target, &m_k);
            if mixed_norm(&residual) < inverse.tolerance {
                converged = true;
                break;
            }
            x += residual; // fixed-point correction
            x[1] = x[1].max(0.0); // keep e physical
        }
        if !converged {
            return Err(BraheError::NumericalError(format!(
                "numerical mean-to-osc did not converge within {} iterations",
                inverse.max_iterations
            )));
        }
        out.push((t, x));
    }
    Ok(out)
}

/// Propagates a candidate osculating state `x_rad` (defined at epoch `t`) across a
/// window bracketing `t` and returns the numerically averaged mean elements at
/// exactly `t`.
///
/// Samples epochs on a grid `t + j*dt` (integer `j`, `dt = 0.01*W`) spanning a range
/// that fully supports a `config.alignment` window of length `W = config.window_seconds`
/// anchored at `t`, padded by ~10% of `W` on each side (so the boundary of the
/// alignment's window at `t` is comfortably inside the sampled trajectory, avoiding
/// bit-exact-boundary fragility). Because `j=0` is always a grid point, `t` itself is
/// always one of the sampled epochs:
///
/// - [`WindowAlignment::Centered`]: `j` from `-60` to `60`, i.e. `[t - 0.6*W, t + 0.6*W]`
/// - [`WindowAlignment::Trailing`]: `j` from `-110` to `10`, i.e. `[t - 1.1*W, t + 0.1*W]`
/// - [`WindowAlignment::Leading`]: `j` from `-10` to `110`, i.e. `[t - 0.1*W, t + 1.1*W]`
///
/// The candidate state is first propagated backward to the window start (to obtain a
/// valid initial condition there), then a fresh propagator integrates forward across
/// the full window so the sampled trajectory has a single, monotonically increasing
/// time history suitable for interpolation.
///
/// The averaged output is selected at exactly `t` (not the nearest sample) because the
/// fast mean-longitude/anomaly is evaluated at the anchor epoch; averaging near-but-not-
/// at `t` and taking the nearest neighbor would bias the recovered fast angle by up to
/// one grid step. If no output exists at exactly `t` (which should not happen given `t`
/// is a grid point), this is treated as a numerical failure rather than silently falling
/// back to a near neighbor.
fn forward_average(
    t: Epoch,
    x_rad: &SVector<f64, 6>,
    config: &MeanElementNumericalMethodConfig,
    inverse: &MeanElementInverseConfig,
) -> Result<SVector<f64, 6>, BraheError> {
    let w = config.window_seconds;
    // Sample spacing: 1% of the window length. Combined with the per-alignment grid
    // extents below, this yields 121 samples spanning the padded window (matching the
    // ~100-sample budget used previously) while guaranteeing `t` is exactly on the grid.
    let dt = 0.01 * w;
    // (back_n, fwd_n): number of grid steps behind/ahead of `t` needed to cover the
    // alignment-aware padded span documented above.
    let (back_n, fwd_n): (i64, i64) = match config.alignment {
        WindowAlignment::Centered => (60, 60),
        WindowAlignment::Trailing => (110, 10),
        WindowAlignment::Leading => (10, 110),
    };

    let start = t - (back_n as f64) * dt;
    let end = t + (fwd_n as f64) * dt;

    let cart_t = state_koe_to_eci(*x_rad, AngleFormat::Radians);

    // Propagate backward from t to the window start to obtain a valid initial state
    // there.
    let mut back = DNumericalOrbitPropagator::builder()
        .epoch(t)
        .state(DVector::from_row_slice(cart_t.as_slice()))
        .force_config(inverse.force_model.clone())
        .propagation_config(inverse.propagation.clone())
        .build()?;
    back.propagate_to(start)?;
    let cart_start = back.state(start)?;

    // Integrate forward across the full window from a single starting condition, so
    // the trajectory samples are in strictly increasing epoch order.
    let mut fwd = DNumericalOrbitPropagator::builder()
        .epoch(start)
        .state(cart_start)
        .force_config(inverse.force_model.clone())
        .propagation_config(inverse.propagation.clone())
        .build()?;
    fwd.propagate_to(end)?;

    let n_samples = (back_n + fwd_n + 1) as usize;
    let mut sample_epochs = Vec::with_capacity(n_samples);
    let mut sample_states = Vec::with_capacity(n_samples);
    for j in -back_n..=fwd_n {
        // Use `t` itself (not a recomputed `t + 0*dt`) at j=0 so the exact-anchor
        // lookup below is a bit-exact match rather than relying on floating-point
        // round-trip equality through Epoch arithmetic.
        let e = if j == 0 { t } else { t + (j as f64) * dt };
        let cart = fwd.state(e)?;
        let cart6 = SVector::<f64, 6>::from_row_slice(cart.as_slice());
        let koe = state_eci_to_koe(cart6, AngleFormat::Radians);
        sample_epochs.push(e);
        sample_states.push(koe);
    }

    let averaged = numerical_osc_to_mean(&sample_epochs, &sample_states, config)?;
    averaged
        .into_iter()
        .find(|(e, _)| *e == t)
        .map(|(_, s)| s)
        .ok_or_else(|| {
            BraheError::NumericalError(
                "numerical mean-to-osc: averaging window produced no output at the exact \
                 anchor epoch"
                    .to_string(),
            )
        })
}

/// Angle-aware element residual `target - value` (`a`, `e` linear; angles wrapped to
/// `[-pi, pi]` to avoid spurious corrections across the `0`/`2*pi` branch cut).
fn element_residual(target: &SVector<f64, 6>, value: &SVector<f64, 6>) -> SVector<f64, 6> {
    let mut r = SVector::<f64, 6>::zeros();
    r[0] = target[0] - value[0];
    r[1] = target[1] - value[1];
    let two_pi = 2.0 * std::f64::consts::PI;
    for idx in 2..6 {
        let mut d = target[idx] - value[idx];
        while d > std::f64::consts::PI {
            d -= two_pi;
        }
        while d < -std::f64::consts::PI {
            d += two_pi;
        }
        r[idx] = d;
    }
    r
}

/// Mixed-norm residual magnitude: meters for `a`, scaled for `e` and the angular
/// elements (radians) so a single scalar tolerance applies across dissimilar units.
fn mixed_norm(r: &SVector<f64, 6>) -> f64 {
    let ang = (r[2] * r[2] + r[3] * r[3] + r[4] * r[4] + r[5] * r[5]).sqrt();
    r[0].abs() + 1e6 * r[1].abs() + 1e6 * ang
}

/// Trapezoidal time-weighted average of the slow equinoctial components `[a,h,k,p,q]`
/// over the window, so unevenly-sampled windows give the same result as evenly-sampled
/// ones (an unweighted sample mean would instead be biased toward regions with denser
/// sampling).
///
/// # Arguments
///
/// * `ts` - Sample times relative to the anchor epoch, seconds, ascending
/// * `values` - Per-sample `[a,h,k,p,q]`, one entry per `ts`
///
/// # Returns
///
/// `Σ 0.5*(v_i + v_{i+1})*(t_{i+1}-t_i) / (t_last - t_first)` for each of the 5
/// components. Falls back to the single sample's value when only one sample is present
/// (avoids dividing by a zero-length window).
fn trapezoidal_time_average(ts: &[f64], values: &[[f64; 5]]) -> [f64; 5] {
    let n = ts.len();
    let total = ts[n - 1] - ts[0];
    if n == 1 || total <= 0.0 {
        return values[0];
    }
    let mut integral = [0.0f64; 5];
    for w in 0..n - 1 {
        let dt = ts[w + 1] - ts[w];
        for c in 0..5 {
            integral[c] += 0.5 * (values[w][c] + values[w + 1][c]) * dt;
        }
    }
    integral.map(|s| s / total)
}

/// Unwraps `l` samples, least-squares fits against relative time, and returns the value
/// at `t = 0` (the anchor epoch).
///
/// # Arguments
///
/// * `ts` - Sample times relative to the anchor epoch, seconds
/// * `ls` - Mean-longitude samples, radians, wrapped to `[0, 2*pi)`
///
/// # Returns
///
/// Detrended mean longitude at the anchor epoch, radians, wrapped to `[0, 2*pi)`.
fn detrended_fast_angle(ts: &[f64], ls: &[f64]) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    // Unwrap.
    let mut unwrapped = Vec::with_capacity(ls.len());
    let mut prev = ls[0];
    let mut offset = 0.0;
    unwrapped.push(ls[0]);
    for &l in &ls[1..] {
        let mut d = l - prev;
        while d > std::f64::consts::PI {
            d -= two_pi;
        }
        while d < -std::f64::consts::PI {
            d += two_pi;
        }
        offset += d;
        unwrapped.push(ls[0] + offset);
        prev = l;
    }
    // Least squares slope/intercept of unwrapped vs ts; return intercept (t=0 == anchor).
    let n = ts.len() as f64;
    let mean_t = ts.iter().sum::<f64>() / n;
    let mean_l = unwrapped.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (t, l) in ts.iter().zip(unwrapped.iter()) {
        num += (t - mean_t) * (l - mean_l);
        den += (t - mean_t) * (t - mean_t);
    }
    let slope = if den.abs() > 0.0 { num / den } else { 0.0 };
    let intercept = mean_l - slope * mean_t;
    crate::math::angles::wrap_to_2pi(intercept)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{AngleFormat, R_EARTH};
    use crate::orbits::{MeanElementMethod, state_koe_mean_to_osc};
    use crate::time::Epoch;
    use approx::assert_abs_diff_eq;
    use nalgebra::SVector;
    use serial_test::parallel;

    // Build a synthetic osculating trajectory by analytically expanding a fixed mean
    // state across one period (varying only M). Averaging must recover the mean a,e,i.
    fn synthetic_osc_series_with_elements(
        altitude_m: f64,
        eccentricity: f64,
        inclination_deg: f64,
    ) -> (Vec<Epoch>, Vec<SVector<f64, 6>>, SVector<f64, 6>) {
        let mean_deg = SVector::<f64, 6>::new(
            R_EARTH + altitude_m,
            eccentricity,
            inclination_deg,
            30.0,
            60.0,
            0.0,
        );
        let period = crate::orbits::orbital_period(mean_deg[0]);
        let t0 = Epoch::from_gps_seconds(0.0);
        let n = 201usize;
        let mut epochs = Vec::new();
        let mut states = Vec::new();
        for j in 0..n {
            let frac = j as f64 / (n as f64 - 1.0);
            let t = t0 + frac * period;
            let mut m = mean_deg;
            m[5] = (frac * 360.0) % 360.0;
            let osc_deg =
                state_koe_mean_to_osc(&m, MeanElementMethod::BrouwerLyddane, AngleFormat::Degrees)
                    .unwrap();
            let osc_rad = crate::math::angles::oe_to_radians(osc_deg, AngleFormat::Degrees);
            epochs.push(t);
            states.push(osc_rad);
        }
        let mean_rad = crate::math::angles::oe_to_radians(mean_deg, AngleFormat::Degrees);
        (epochs, states, mean_rad)
    }

    fn synthetic_osc_series() -> (Vec<Epoch>, Vec<SVector<f64, 6>>, SVector<f64, 6>) {
        synthetic_osc_series_with_elements(500e3, 0.01, 45.0)
    }

    // Analytically evaluates the same underlying osculating trajectory as
    // `synthetic_osc_series_with_elements` at an arbitrary fraction `frac` of one
    // period (mean anomaly = frac*360 deg), so it can be resampled at non-uniform
    // times without re-deriving the trajectory.
    fn osc_at_frac(
        mean_deg: SVector<f64, 6>,
        period: f64,
        t0: Epoch,
        frac: f64,
    ) -> (Epoch, SVector<f64, 6>) {
        let t = t0 + frac * period;
        let mut m = mean_deg;
        m[5] = (frac * 360.0) % 360.0;
        let osc_deg =
            state_koe_mean_to_osc(&m, MeanElementMethod::BrouwerLyddane, AngleFormat::Degrees)
                .unwrap();
        let osc_rad = crate::math::angles::oe_to_radians(osc_deg, AngleFormat::Degrees);
        (t, osc_rad)
    }

    // Cadence-independence regression: an unweighted sample mean over the window
    // biases the average toward whichever part of the orbit is sampled more densely,
    // so a non-uniformly-resampled version of the SAME underlying trajectory would
    // recover a visibly different mean a,e,i than a uniformly-sampled one. The
    // trapezoidal time-weighted average must agree closely regardless of sampling
    // cadence.
    #[test]
    #[parallel]
    fn test_numerical_osc_to_mean_cadence_independent() {
        let mean_deg = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 0.0);
        let period = crate::orbits::orbital_period(mean_deg[0]);
        let t0 = Epoch::from_gps_seconds(0.0);
        let mean_rad = crate::math::angles::oe_to_radians(mean_deg, AngleFormat::Degrees);

        // Uniform sampling: 201 evenly-spaced fractions spanning one full period.
        let uniform_fracs: Vec<f64> = (0..=200).map(|j| j as f64 / 200.0).collect();

        // Non-uniform resampling of the SAME underlying trajectory: densely clustered
        // (quadratic warp) over the first half, coarser and uniform over the second
        // half. Still strictly ascending, still spans the full [0,1] fraction range,
        // and includes frac=0.5 exactly so both series share a common anchor epoch.
        let n1 = 150usize;
        let n2 = 50usize;
        let mut nonuniform_fracs: Vec<f64> = (0..=n1)
            .map(|j| 0.5 * (j as f64 / n1 as f64).powi(2))
            .collect();
        nonuniform_fracs.extend((1..=n2).map(|j| 0.5 + 0.5 * (j as f64 / n2 as f64)));

        let build = |fracs: &[f64]| -> (Vec<Epoch>, Vec<SVector<f64, 6>>) {
            fracs
                .iter()
                .map(|&frac| osc_at_frac(mean_deg, period, t0, frac))
                .unzip()
        };
        let (epochs_u, states_u) = build(&uniform_fracs);
        let (epochs_n, states_n) = build(&nonuniform_fracs);

        // Centered window spanning the full [0,1] fraction range, anchored at the
        // midpoint (frac=0.5) shared by both sample sets.
        let cfg = MeanElementNumericalMethodConfig {
            window_seconds: period,
            alignment: WindowAlignment::Centered,
            edge: WindowEdgeHandling::Truncate,
            inverse: None,
        };
        let out_u = numerical_osc_to_mean(&epochs_u, &states_u, &cfg).unwrap();
        let out_n = numerical_osc_to_mean(&epochs_n, &states_n, &cfg).unwrap();

        let anchor = t0 + 0.5 * period;
        let (_, mean_u) = out_u
            .iter()
            .find(|(t, _)| *t == anchor)
            .expect("uniform series must produce output at the shared anchor epoch");
        let (_, mean_n) = out_n
            .iter()
            .find(|(t, _)| *t == anchor)
            .expect("non-uniform series must produce output at the shared anchor epoch");

        // The two cadences must agree closely with each other...
        assert_abs_diff_eq!(mean_u[0], mean_n[0], epsilon = 50.0);
        assert_abs_diff_eq!(mean_u[1], mean_n[1], epsilon = 5e-5);
        assert_abs_diff_eq!(mean_u[2], mean_n[2], epsilon = 5e-5);

        // ...and both with the true analytical mean.
        assert_abs_diff_eq!(mean_u[0], mean_rad[0], epsilon = 500.0);
        assert_abs_diff_eq!(mean_n[0], mean_rad[0], epsilon = 500.0);
        assert_abs_diff_eq!(mean_u[1], mean_rad[1], epsilon = 2e-3);
        assert_abs_diff_eq!(mean_n[1], mean_rad[1], epsilon = 2e-3);
        assert_abs_diff_eq!(mean_u[2], mean_rad[2], epsilon = 2e-3);
        assert_abs_diff_eq!(mean_n[2], mean_rad[2], epsilon = 2e-3);
    }

    #[test]
    #[parallel]
    fn test_numerical_osc_to_mean_recovers_mean_ae_i() {
        let (epochs, states, mean_rad) = synthetic_osc_series();
        let period = crate::orbits::orbital_period(mean_rad[0]);
        let cfg = MeanElementNumericalMethodConfig {
            window_seconds: period * 0.6,
            alignment: WindowAlignment::Centered,
            edge: WindowEdgeHandling::Truncate,
            inverse: None,
        };
        let out = numerical_osc_to_mean(&epochs, &states, &cfg).unwrap();
        assert!(!out.is_empty());
        let (_, mid) = &out[out.len() / 2];
        assert_abs_diff_eq!(mid[0], mean_rad[0], epsilon = 500.0);
        assert_abs_diff_eq!(mid[1], mean_rad[1], epsilon = 2e-3);
        assert_abs_diff_eq!(mid[2], mean_rad[2], epsilon = 2e-3);
    }

    // Regression test for the fr (retrograde factor) inconsistency bug: sun-synchronous
    // inclinations (~98 deg, in the (90,180) band away from the i=180 singularity) must
    // recover the true inclination rather than 180-i due to a forward/inverse fr mismatch.
    #[test]
    #[parallel]
    fn test_numerical_osc_to_mean_recovers_mean_ae_i_sun_synchronous() {
        let (epochs, states, mean_rad) = synthetic_osc_series_with_elements(700e3, 0.01, 98.0);
        let period = crate::orbits::orbital_period(mean_rad[0]);
        let cfg = MeanElementNumericalMethodConfig {
            window_seconds: period * 0.5,
            alignment: WindowAlignment::Centered,
            edge: WindowEdgeHandling::Truncate,
            inverse: None,
        };
        let out = numerical_osc_to_mean(&epochs, &states, &cfg).unwrap();
        assert!(!out.is_empty());
        let (_, mid) = &out[out.len() / 2];
        assert_abs_diff_eq!(mid[0], mean_rad[0], epsilon = 500.0);
        assert_abs_diff_eq!(mid[1], mean_rad[1], epsilon = 2e-3);
        assert_abs_diff_eq!(mid[2], mean_rad[2], epsilon = 2e-3);
    }

    #[test]
    #[parallel]
    fn test_truncate_shortens_preserve_keeps_length() {
        let (epochs, states, mean_rad) = synthetic_osc_series();
        let period = crate::orbits::orbital_period(mean_rad[0]);
        let base = |edge| MeanElementNumericalMethodConfig {
            window_seconds: period / 2.0,
            alignment: WindowAlignment::Centered,
            edge,
            inverse: None,
        };
        let trunc =
            numerical_osc_to_mean(&epochs, &states, &base(WindowEdgeHandling::Truncate)).unwrap();
        let preserve =
            numerical_osc_to_mean(&epochs, &states, &base(WindowEdgeHandling::PreserveWindow))
                .unwrap();
        assert!(trunc.len() < epochs.len());
        assert_eq!(preserve.len(), epochs.len());
    }

    #[test]
    #[serial_test::serial]
    fn test_numerical_mean_to_osc_round_trip() {
        use crate::propagators::{ForceModelConfig, NumericalPropagationConfig};
        crate::utils::testing::setup_global_test_eop();

        let mean_rad = crate::math::angles::oe_to_radians(
            SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 0.0),
            AngleFormat::Degrees,
        );
        let period = crate::orbits::orbital_period(mean_rad[0]);
        let inverse = MeanElementInverseConfig {
            force_model: ForceModelConfig::earth_gravity(),
            propagation: NumericalPropagationConfig::default(),
            tolerance: 1.0, // mixed-norm tolerance; see mixed_norm()
            max_iterations: 25,
        };
        let cfg = MeanElementNumericalMethodConfig {
            window_seconds: period,
            alignment: WindowAlignment::Centered,
            edge: WindowEdgeHandling::PreserveWindow,
            inverse: Some(inverse),
        };
        let epochs = vec![Epoch::from_gps_seconds(0.0)];
        let out = numerical_mean_to_osc(&epochs, &[mean_rad], &cfg).unwrap();
        assert_eq!(out.len(), 1);
        let (_, osc) = &out[0];
        assert!(osc.iter().all(|v| v.is_finite()));

        // Real round-trip: average the recovered osculating state back over a freshly
        // propagated window and confirm it returns to the target mean a, e, i.
        let cart = state_koe_to_eci(*osc, AngleFormat::Radians);
        let anchor = epochs[0];
        let half = period * 0.6;
        let start = anchor - half;
        let end = anchor + half;
        let inv = cfg.inverse.as_ref().unwrap();
        let mut back = DNumericalOrbitPropagator::builder()
            .epoch(anchor)
            .state(DVector::from_row_slice(cart.as_slice()))
            .force_config(inv.force_model.clone())
            .propagation_config(inv.propagation.clone())
            .build()
            .unwrap();
        back.propagate_to(start).unwrap();
        let cart_start = back.state(start).unwrap();
        let mut fwd = DNumericalOrbitPropagator::builder()
            .epoch(start)
            .state(cart_start)
            .force_config(inv.force_model.clone())
            .propagation_config(inv.propagation.clone())
            .build()
            .unwrap();
        fwd.propagate_to(end).unwrap();

        // Sample on a grid that includes the anchor epoch exactly (mirroring
        // `forward_average`'s exact-anchor design), so the recovered fast angle
        // (mean anomaly) is evaluated AT the target epoch rather than at the nearest
        // sample of an evenly-spaced grid that never lands exactly on it.
        let back_n = 60i64;
        let fwd_n = 60i64;
        let dt = 0.01 * period;
        let n = (back_n + fwd_n + 1) as usize;
        let mut sample_epochs = Vec::with_capacity(n);
        let mut sample_states = Vec::with_capacity(n);
        for j in -back_n..=fwd_n {
            let e = if j == 0 {
                anchor
            } else {
                anchor + (j as f64) * dt
            };
            let s = fwd.state(e).unwrap();
            let koe = state_eci_to_koe(
                SVector::<f64, 6>::from_row_slice(s.as_slice()),
                AngleFormat::Radians,
            );
            sample_epochs.push(e);
            sample_states.push(koe);
        }
        let averaged = numerical_osc_to_mean(&sample_epochs, &sample_states, &cfg).unwrap();
        let (_, mean_recovered) = averaged
            .into_iter()
            .find(|(e, _)| *e == anchor)
            .expect("averaging window must contain the exact anchor epoch");
        assert_abs_diff_eq!(mean_recovered[0], mean_rad[0], epsilon = 500.0);
        assert_abs_diff_eq!(mean_recovered[1], mean_rad[1], epsilon = 2e-3);
        assert_abs_diff_eq!(mean_recovered[2], mean_rad[2], epsilon = 2e-3);

        // Fast-angle check: with a nearest-sample (rather than exact-anchor) selection,
        // the recovered mean anomaly is evaluated up to ~0.006*W away from the target
        // epoch, biasing it by several hundredths of a radian even though a, e, i
        // converge cleanly. With the exact-anchor pipeline this must match tightly.
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut d_m = mean_recovered[5] - mean_rad[5];
        while d_m > std::f64::consts::PI {
            d_m -= two_pi;
        }
        while d_m < -std::f64::consts::PI {
            d_m += two_pi;
        }
        assert_abs_diff_eq!(d_m, 0.0, epsilon = 1e-3);
    }

    /// `element_residual` wraps angular residuals into `[-pi, pi]` in both directions;
    /// linear components (`a`, `e`) pass through unwrapped.
    #[test]
    #[parallel]
    fn test_element_residual_wraps_both_directions() {
        let two_pi = 2.0 * std::f64::consts::PI;
        let hi = SVector::<f64, 6>::new(7.0e6, 0.01, 3.0, 0.0, 0.0, 0.0);
        let lo = SVector::<f64, 6>::new(7.0e6, 0.02, -3.0, 0.0, 0.0, 0.0);
        // target - value = +6 rad > pi: wraps down by 2*pi.
        let r = element_residual(&hi, &lo);
        assert_abs_diff_eq!(r[2], 6.0 - two_pi, epsilon = 1e-12);
        // target - value = -6 rad < -pi: wraps up by 2*pi.
        let r2 = element_residual(&lo, &hi);
        assert_abs_diff_eq!(r2[2], -6.0 + two_pi, epsilon = 1e-12);
        // Linear components are plain differences.
        assert_abs_diff_eq!(r[0], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(r2[1], 0.01, epsilon = 1e-12);
    }

    /// `detrended_fast_angle` unwraps a positive `>pi` jump between adjacent samples
    /// and returns a value normalized to `[0, 2*pi)`.
    #[test]
    #[parallel]
    fn test_detrended_fast_angle_unwraps_positive_jump() {
        let two_pi = 2.0 * std::f64::consts::PI;
        let ts = vec![-1.0, 0.0, 1.0];
        // Consecutive deltas exceed pi (0.1 -> 6.2 is +6.1), forcing the unwrap branch.
        let ls = vec![0.1, 6.2, 0.2 + two_pi];
        let val = detrended_fast_angle(&ts, &ls);
        assert!(val.is_finite());
        assert!((0.0..two_pi).contains(&val));
    }

    /// A target whose analytical Brouwer-Lyddane seed is non-finite (exactly equatorial
    /// inclination makes the seed divide by `tan(i) = 0`) must fall back to seeding from
    /// the target itself and never return a NaN result.
    #[test]
    #[serial_test::serial]
    fn test_numerical_mean_to_osc_nonfinite_seed_fallback() {
        use crate::propagators::{ForceModelConfig, NumericalPropagationConfig};
        crate::utils::testing::setup_global_test_eop();

        let mean_deg = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 0.0, 0.0, 0.0, 0.0);
        let mean_rad = crate::math::angles::oe_to_radians(mean_deg, AngleFormat::Degrees);
        // Confirm the analytical seed is genuinely non-finite for this equatorial target,
        // so this test actually exercises the fallback rather than the normal path.
        let seed = state_koe_mean_to_osc(
            &mean_deg,
            MeanElementMethod::BrouwerLyddane,
            AngleFormat::Degrees,
        )
        .unwrap();
        assert!(
            !seed.iter().all(|v| v.is_finite()),
            "expected non-finite Brouwer-Lyddane seed at i = 0"
        );

        let period = crate::orbits::orbital_period(mean_rad[0]);
        let cfg = MeanElementNumericalMethodConfig {
            window_seconds: period,
            alignment: WindowAlignment::Centered,
            edge: WindowEdgeHandling::PreserveWindow,
            inverse: Some(MeanElementInverseConfig {
                force_model: ForceModelConfig::earth_gravity(),
                propagation: NumericalPropagationConfig::default(),
                tolerance: 1.0,
                max_iterations: 25,
            }),
        };
        let epochs = vec![Epoch::from_gps_seconds(0.0)];
        // Either converges to a finite osculating state or reports a clean NumericalError,
        // but never a NaN result or panic.
        match numerical_mean_to_osc(&epochs, &[mean_rad], &cfg) {
            Ok(out) => assert!(out[0].1.iter().all(|v| v.is_finite())),
            Err(BraheError::NumericalError(_)) => {}
            Err(e) => panic!("unexpected error variant: {e:?}"),
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_numerical_mean_to_osc_requires_inverse_config() {
        let mean_rad = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 0.78, 0.5, 1.0, 0.0);
        let cfg = MeanElementNumericalMethodConfig {
            window_seconds: 5400.0,
            alignment: WindowAlignment::Centered,
            edge: WindowEdgeHandling::PreserveWindow,
            inverse: None,
        };
        let epochs = vec![Epoch::from_gps_seconds(0.0)];
        assert!(numerical_mean_to_osc(&epochs, &[mean_rad], &cfg).is_err());
    }

    /// Regression test for the non-finite analytical-seed guard: at the critical
    /// inclination (~63.435 deg), the Brouwer-Lyddane seed used by
    /// `numerical_mean_to_osc` divides by `(1 - 5cos^2 i)`, which is exactly zero here
    /// and produces a non-finite seed. The solver must fall back to seeding from the
    /// target mean elements rather than propagating NaNs.
    #[test]
    #[serial_test::serial]
    fn test_numerical_mean_to_osc_near_critical_inclination_seed_fallback() {
        use crate::propagators::{ForceModelConfig, NumericalPropagationConfig};
        crate::utils::testing::setup_global_test_eop();

        let i_critical = (1.0 / 5.0_f64.sqrt()).acos();
        let mean_rad = SVector::<f64, 6>::new(
            R_EARTH + 700e3,
            0.01,
            i_critical,
            30.0_f64.to_radians(),
            60.0_f64.to_radians(),
            0.0,
        );
        // Confirm this target actually exercises the non-finite-seed path being
        // guarded against; otherwise the test would pass vacuously.
        let seed = state_koe_mean_to_osc(
            &mean_rad,
            MeanElementMethod::BrouwerLyddane,
            AngleFormat::Radians,
        )
        .unwrap();
        assert!(
            !seed.iter().all(|v| v.is_finite()),
            "test target does not exercise a non-finite analytical seed; adjust inclination"
        );

        let period = crate::orbits::orbital_period(mean_rad[0]);
        let inverse = MeanElementInverseConfig {
            force_model: ForceModelConfig::earth_gravity(),
            propagation: NumericalPropagationConfig::default(),
            tolerance: 1.0,
            max_iterations: 25,
        };
        let cfg = MeanElementNumericalMethodConfig {
            window_seconds: period,
            alignment: WindowAlignment::Centered,
            edge: WindowEdgeHandling::PreserveWindow,
            inverse: Some(inverse),
        };
        let epochs = vec![Epoch::from_gps_seconds(0.0)];
        match numerical_mean_to_osc(&epochs, &[mean_rad], &cfg) {
            Ok(out) => {
                assert_eq!(out.len(), 1);
                assert!(out[0].1.iter().all(|v| v.is_finite()));
            }
            Err(BraheError::NumericalError(_)) => {}
            Err(e) => panic!("unexpected error variant: {e:?}"),
        }
    }

    #[test]
    #[parallel]
    fn test_numerical_osc_to_mean_rejects_unsorted_epochs() {
        let (mut epochs, states, _mean_rad) = synthetic_osc_series();
        epochs.swap(0, 1); // break strict ascending order
        let cfg = MeanElementNumericalMethodConfig {
            window_seconds: 3600.0,
            alignment: WindowAlignment::Centered,
            edge: WindowEdgeHandling::Truncate,
            inverse: None,
        };
        assert!(numerical_osc_to_mean(&epochs, &states, &cfg).is_err());
    }

    #[test]
    #[parallel]
    fn test_numerical_osc_to_mean_rejects_non_finite_window_seconds() {
        let (epochs, states, _mean_rad) = synthetic_osc_series();
        let cfg = MeanElementNumericalMethodConfig {
            window_seconds: f64::NAN,
            alignment: WindowAlignment::Centered,
            edge: WindowEdgeHandling::Truncate,
            inverse: None,
        };
        assert!(numerical_osc_to_mean(&epochs, &states, &cfg).is_err());
    }

    #[test]
    #[parallel]
    fn test_numerical_mean_to_osc_rejects_invalid_inverse_config() {
        use crate::propagators::{ForceModelConfig, NumericalPropagationConfig};
        let mean_rad = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 0.78, 0.5, 1.0, 0.0);
        let epochs = vec![Epoch::from_gps_seconds(0.0)];

        let bad_tolerance = MeanElementInverseConfig {
            force_model: ForceModelConfig::earth_gravity(),
            propagation: NumericalPropagationConfig::default(),
            tolerance: 0.0,
            max_iterations: 25,
        };
        let cfg = MeanElementNumericalMethodConfig {
            window_seconds: 5400.0,
            alignment: WindowAlignment::Centered,
            edge: WindowEdgeHandling::PreserveWindow,
            inverse: Some(bad_tolerance),
        };
        assert!(numerical_mean_to_osc(&epochs, &[mean_rad], &cfg).is_err());

        let bad_iterations = MeanElementInverseConfig {
            force_model: ForceModelConfig::earth_gravity(),
            propagation: NumericalPropagationConfig::default(),
            tolerance: 1.0,
            max_iterations: 0,
        };
        let cfg2 = MeanElementNumericalMethodConfig {
            window_seconds: 5400.0,
            alignment: WindowAlignment::Centered,
            edge: WindowEdgeHandling::PreserveWindow,
            inverse: Some(bad_iterations),
        };
        assert!(numerical_mean_to_osc(&epochs, &[mean_rad], &cfg2).is_err());
    }

    // ---- window_bounds ----

    #[test]
    #[parallel]
    fn test_window_bounds_trailing() {
        let t0 = Epoch::from_gps_seconds(0.0);
        let data_start = t0;
        let data_end = t0 + 10_000.0;
        let anchor = t0 + 5_000.0;
        let cfg = MeanElementNumericalMethodConfig {
            window_seconds: 1_000.0,
            alignment: WindowAlignment::Trailing,
            edge: WindowEdgeHandling::Truncate,
            inverse: None,
        };
        let (start, end) = window_bounds(anchor, data_start, data_end, &cfg).unwrap();
        // Trailing: [anchor - W, anchor]
        assert_abs_diff_eq!(start - (anchor - 1_000.0), 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(end - anchor, 0.0, epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_window_bounds_leading() {
        let t0 = Epoch::from_gps_seconds(0.0);
        let data_start = t0;
        let data_end = t0 + 10_000.0;
        let anchor = t0 + 5_000.0;
        let cfg = MeanElementNumericalMethodConfig {
            window_seconds: 1_000.0,
            alignment: WindowAlignment::Leading,
            edge: WindowEdgeHandling::Truncate,
            inverse: None,
        };
        let (start, end) = window_bounds(anchor, data_start, data_end, &cfg).unwrap();
        // Leading: [anchor, anchor + W]
        assert_abs_diff_eq!(start - anchor, 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(end - (anchor + 1_000.0), 0.0, epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_window_bounds_preserve_window_clamps_when_longer_than_data_span() {
        let t0 = Epoch::from_gps_seconds(0.0);
        let data_start = t0;
        let data_end = t0 + 100.0; // total data span is only 100s
        let anchor = t0 + 50.0;
        let cfg = MeanElementNumericalMethodConfig {
            window_seconds: 10_000.0, // much longer than the data span
            alignment: WindowAlignment::Centered,
            edge: WindowEdgeHandling::PreserveWindow,
            inverse: None,
        };
        let (start, end) = window_bounds(anchor, data_start, data_end, &cfg).unwrap();
        // Window is clamped to the data bounds rather than sliding past them.
        assert_abs_diff_eq!(start - data_start, 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(end - data_end, 0.0, epsilon = 1e-9);
    }

    // ---- numerical_osc_to_mean guards / edge cases ----

    #[test]
    #[parallel]
    fn test_numerical_osc_to_mean_mismatched_lengths_is_error() {
        let epochs = vec![Epoch::from_gps_seconds(0.0), Epoch::from_gps_seconds(60.0)];
        let states = vec![SVector::<f64, 6>::new(
            R_EARTH + 500e3,
            0.01,
            0.78,
            0.5,
            1.0,
            0.0,
        )];
        let cfg = MeanElementNumericalMethodConfig {
            window_seconds: 3600.0,
            alignment: WindowAlignment::Centered,
            edge: WindowEdgeHandling::Truncate,
            inverse: None,
        };
        assert!(numerical_osc_to_mean(&epochs, &states, &cfg).is_err());
    }

    #[test]
    #[parallel]
    fn test_numerical_osc_to_mean_empty_input_returns_empty_ok() {
        let epochs: Vec<Epoch> = Vec::new();
        let states: Vec<SVector<f64, 6>> = Vec::new();
        let cfg = MeanElementNumericalMethodConfig {
            window_seconds: 3600.0,
            alignment: WindowAlignment::Centered,
            edge: WindowEdgeHandling::Truncate,
            inverse: None,
        };
        let out = numerical_osc_to_mean(&epochs, &states, &cfg).unwrap();
        assert!(out.is_empty());
    }

    // Regression coverage for the `fr = -1` retrograde branch: a near-180-degree
    // (retrograde) inclination must still recover finite mean elements. Sweeps mean
    // anomaly across a fixed, unperturbed Keplerian ellipse (rather than routing through
    // the analytical Brouwer-Lyddane mean->osc mapping, which is itself singular in this
    // near-180-degree regime and cannot be used to synthesize the input trajectory) so
    // the averaged elements should reproduce the input a, e, i essentially exactly.
    #[test]
    #[parallel]
    fn test_numerical_osc_to_mean_retrograde_orbit() {
        let inclination_deg = 179.95_f64;
        let inclination_rad = inclination_deg.to_radians();
        let threshold = std::f64::consts::FRAC_PI_2 * 1.999;
        assert!(
            inclination_rad > threshold,
            "test target does not exceed the fr=-1 threshold; adjust inclination"
        );
        let a = R_EARTH + 500e3;
        let e = 0.01;
        let raan = 30.0_f64.to_radians();
        let argp = 60.0_f64.to_radians();
        let period = crate::orbits::orbital_period(a);
        let t0 = Epoch::from_gps_seconds(0.0);
        let n = 201usize;
        let mut epochs = Vec::with_capacity(n);
        let mut states = Vec::with_capacity(n);
        for j in 0..n {
            let frac = j as f64 / (n as f64 - 1.0);
            let t = t0 + frac * period;
            let m_anom = frac * 2.0 * std::f64::consts::PI;
            epochs.push(t);
            states.push(SVector::<f64, 6>::new(
                a,
                e,
                inclination_rad,
                raan,
                argp,
                m_anom,
            ));
        }
        let cfg = MeanElementNumericalMethodConfig {
            window_seconds: period * 0.5,
            alignment: WindowAlignment::Centered,
            edge: WindowEdgeHandling::Truncate,
            inverse: None,
        };
        let out = numerical_osc_to_mean(&epochs, &states, &cfg).unwrap();
        assert!(!out.is_empty());
        let (_, mid) = &out[out.len() / 2];
        assert!(mid.iter().all(|v| v.is_finite()));
        assert_abs_diff_eq!(mid[0], a, epsilon = 1e-3);
        assert_abs_diff_eq!(mid[1], e, epsilon = 1e-9);
        assert_abs_diff_eq!(mid[2], inclination_rad, epsilon = 1e-9);
    }

    // ---- numerical_mean_to_osc guards (fire before propagation) ----

    #[test]
    #[parallel]
    fn test_numerical_mean_to_osc_mismatched_lengths_is_error() {
        use crate::propagators::{ForceModelConfig, NumericalPropagationConfig};
        let inverse = MeanElementInverseConfig {
            force_model: ForceModelConfig::earth_gravity(),
            propagation: NumericalPropagationConfig::default(),
            tolerance: 1.0,
            max_iterations: 25,
        };
        let cfg = MeanElementNumericalMethodConfig {
            window_seconds: 5400.0,
            alignment: WindowAlignment::Centered,
            edge: WindowEdgeHandling::PreserveWindow,
            inverse: Some(inverse),
        };
        let epochs = vec![Epoch::from_gps_seconds(0.0), Epoch::from_gps_seconds(60.0)];
        let states = vec![SVector::<f64, 6>::new(
            R_EARTH + 500e3,
            0.01,
            0.78,
            0.5,
            1.0,
            0.0,
        )];
        assert!(numerical_mean_to_osc(&epochs, &states, &cfg).is_err());
    }

    #[test]
    #[parallel]
    fn test_numerical_mean_to_osc_rejects_non_finite_window_seconds() {
        use crate::propagators::{ForceModelConfig, NumericalPropagationConfig};
        let inverse = MeanElementInverseConfig {
            force_model: ForceModelConfig::earth_gravity(),
            propagation: NumericalPropagationConfig::default(),
            tolerance: 1.0,
            max_iterations: 25,
        };
        let cfg = MeanElementNumericalMethodConfig {
            window_seconds: -1.0,
            alignment: WindowAlignment::Centered,
            edge: WindowEdgeHandling::PreserveWindow,
            inverse: Some(inverse),
        };
        let epochs = vec![Epoch::from_gps_seconds(0.0)];
        let states = vec![SVector::<f64, 6>::new(
            R_EARTH + 500e3,
            0.01,
            0.78,
            0.5,
            1.0,
            0.0,
        )];
        assert!(numerical_mean_to_osc(&epochs, &states, &cfg).is_err());
    }

    // ---- numerical_mean_to_osc: propagation-dependent behavior ----

    // Non-convergence: an impossibly tight tolerance combined with a single allowed
    // iteration must surface a clean `NumericalError` rather than looping forever or
    // silently returning an under-converged result.
    #[test]
    #[serial_test::serial]
    fn test_numerical_mean_to_osc_non_convergence_returns_numerical_error() {
        use crate::propagators::{ForceModelConfig, NumericalPropagationConfig};
        crate::utils::testing::setup_global_test_eop();

        let mean_rad = crate::math::angles::oe_to_radians(
            SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 0.0),
            AngleFormat::Degrees,
        );
        let period = crate::orbits::orbital_period(mean_rad[0]);
        let inverse = MeanElementInverseConfig {
            force_model: ForceModelConfig::earth_gravity(),
            propagation: NumericalPropagationConfig::default(),
            tolerance: 1e-30,
            max_iterations: 1,
        };
        let cfg = MeanElementNumericalMethodConfig {
            window_seconds: period,
            alignment: WindowAlignment::Centered,
            edge: WindowEdgeHandling::PreserveWindow,
            inverse: Some(inverse),
        };
        let epochs = vec![Epoch::from_gps_seconds(0.0)];
        match numerical_mean_to_osc(&epochs, &[mean_rad], &cfg) {
            Err(BraheError::NumericalError(_)) => {}
            other => panic!("expected NumericalError, got {other:?}"),
        }
    }

    // Trailing/Leading `forward_average` span arms: a single LEO mean state must still
    // converge and recover a, e, i under both non-centered alignments.
    #[test]
    #[serial_test::serial]
    fn test_numerical_mean_to_osc_trailing_alignment() {
        use crate::propagators::{ForceModelConfig, NumericalPropagationConfig};
        crate::utils::testing::setup_global_test_eop();

        let mean_rad = crate::math::angles::oe_to_radians(
            SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 0.0),
            AngleFormat::Degrees,
        );
        let period = crate::orbits::orbital_period(mean_rad[0]);
        let inverse = MeanElementInverseConfig {
            force_model: ForceModelConfig::earth_gravity(),
            propagation: NumericalPropagationConfig::default(),
            tolerance: 1.0,
            max_iterations: 25,
        };
        let cfg = MeanElementNumericalMethodConfig {
            window_seconds: period,
            alignment: WindowAlignment::Trailing,
            edge: WindowEdgeHandling::PreserveWindow,
            inverse: Some(inverse),
        };
        let epochs = vec![Epoch::from_gps_seconds(0.0)];
        let out = numerical_mean_to_osc(&epochs, &[mean_rad], &cfg).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].1.iter().all(|v| v.is_finite()));
        assert_abs_diff_eq!(out[0].1[0], mean_rad[0], epsilon = 30_000.0);
    }

    #[test]
    #[serial_test::serial]
    fn test_numerical_mean_to_osc_leading_alignment() {
        use crate::propagators::{ForceModelConfig, NumericalPropagationConfig};
        crate::utils::testing::setup_global_test_eop();

        let mean_rad = crate::math::angles::oe_to_radians(
            SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 0.0),
            AngleFormat::Degrees,
        );
        let period = crate::orbits::orbital_period(mean_rad[0]);
        let inverse = MeanElementInverseConfig {
            force_model: ForceModelConfig::earth_gravity(),
            propagation: NumericalPropagationConfig::default(),
            tolerance: 1.0,
            max_iterations: 25,
        };
        let cfg = MeanElementNumericalMethodConfig {
            window_seconds: period,
            alignment: WindowAlignment::Leading,
            edge: WindowEdgeHandling::PreserveWindow,
            inverse: Some(inverse),
        };
        let epochs = vec![Epoch::from_gps_seconds(0.0)];
        let out = numerical_mean_to_osc(&epochs, &[mean_rad], &cfg).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].1.iter().all(|v| v.is_finite()));
        assert_abs_diff_eq!(out[0].1[0], mean_rad[0], epsilon = 30_000.0);
    }

    // ---- element_residual ----

    // A target/value pair whose raw difference exceeds pi in magnitude must be wrapped
    // into [-pi, pi] rather than left as the raw (unwrapped) difference.
    #[test]
    #[parallel]
    fn test_element_residual_wraps_large_angular_difference() {
        let two_pi = 2.0 * std::f64::consts::PI;
        let target = SVector::<f64, 6>::new(0.0, 0.0, 0.0, 0.0, 0.1, 0.0);
        let value = SVector::<f64, 6>::new(0.0, 0.0, 0.0, 0.0, two_pi - 0.1, 0.0);
        // Raw difference is 0.1 - (2*pi - 0.1) ~= -6.083 rad, which exceeds pi in
        // magnitude and must be wrapped to +0.2 rad.
        let raw_diff = target[4] - value[4];
        assert!(raw_diff.abs() > std::f64::consts::PI);
        let r = element_residual(&target, &value);
        assert!(r[4] <= std::f64::consts::PI && r[4] >= -std::f64::consts::PI);
        assert_abs_diff_eq!(r[4], 0.2, epsilon = 1e-9);
    }

    // ---- trapezoidal_time_average ----

    #[test]
    #[parallel]
    fn test_trapezoidal_time_average_single_sample_fallback() {
        let ts = [0.0];
        let values = [[1.0, 2.0, 3.0, 4.0, 5.0]];
        let avg = trapezoidal_time_average(&ts, &values);
        assert_eq!(avg, values[0]);
    }

    #[test]
    #[parallel]
    fn test_trapezoidal_time_average_zero_elapsed_time_fallback() {
        let ts = [10.0, 10.0, 10.0];
        let values = [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
        ];
        let avg = trapezoidal_time_average(&ts, &values);
        // Zero elapsed time over the window falls back to the first sample's value.
        assert_eq!(avg, values[0]);
    }

    // ---- detrended_fast_angle ----

    // A perfectly linear mean longitude that wraps across 2*pi within the window must
    // be unwrapped before the linear fit; the intercept at t=0 should recover the true
    // (unwrapped) anchor value rather than being corrupted by the 2*pi discontinuity.
    #[test]
    #[parallel]
    fn test_detrended_fast_angle_unwraps_across_2pi() {
        let two_pi = 2.0 * std::f64::consts::PI;
        let ts: Vec<f64> = (0..100).map(|j| j as f64).collect();
        let ls: Vec<f64> = ts.iter().map(|&t| (0.1 * t).rem_euclid(two_pi)).collect();
        // Sanity check: the raw (wrapped) samples do jump by more than pi somewhere in
        // the window, so this test actually exercises the unwrap loop.
        assert!(
            ls.windows(2)
                .any(|w| (w[1] - w[0]).abs() > std::f64::consts::PI)
        );
        let l0 = detrended_fast_angle(&ts, &ls);
        // The true underlying continuous value at t=0 (the anchor) is exactly 0.
        assert_abs_diff_eq!(l0, 0.0, epsilon = 1e-6);
    }
}
