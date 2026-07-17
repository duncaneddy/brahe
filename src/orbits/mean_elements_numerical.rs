/*!
 * Numerical mean-element averaging.
 *
 * Osculating states are averaged over a moving window in equinoctial space
 * (singularity-safe at e→0, i→0). Slow components a,h,k,p,q are arithmetic-averaged;
 * the fast mean-longitude l is linearly detrended and evaluated at the anchor epoch.
 */

use crate::constants::AngleFormat;
use crate::coordinates::{state_eci_to_koe, state_koe_to_eci};
use crate::orbits::equinoctial::{state_equinoctial_to_koe, state_koe_to_equinoctial};
use crate::orbits::mean_elements::{EdgeHandling, InverseConfig, NumericalConfig, WindowAlignment};
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
/// `config.edge` is [`EdgeHandling::Truncate`] and the window would extend past the data
/// bounds.
// Not yet called from non-test production code; wired up by a later task.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn window_bounds(
    anchor: Epoch,
    data_start: Epoch,
    data_end: Epoch,
    config: &NumericalConfig,
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
            EdgeHandling::Truncate => return None,
            EdgeHandling::PreserveWindow => {
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
/// converts each osculating sample to equinoctial elements, arithmetic-averages the
/// slow components `a, h, k, p, q`, and linearly detrends the fast mean longitude `l`
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
/// [`EdgeHandling::Truncate`].
///
/// # Errors
///
/// Returns [`BraheError::Error`] if `epochs` and `states_rad` have unequal length, or if
/// `config.window_seconds` is not positive.
// Not yet called from non-test production code; wired up by a later task.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn numerical_osc_to_mean(
    epochs: &[Epoch],
    states_rad: &[SVector<f64, 6>],
    config: &NumericalConfig,
) -> Result<Vec<(Epoch, SVector<f64, 6>)>, BraheError> {
    if epochs.len() != states_rad.len() {
        return Err(BraheError::Error(
            "epochs and states must have equal length".to_string(),
        ));
    }
    if epochs.is_empty() {
        return Ok(Vec::new());
    }
    if config.window_seconds <= 0.0 {
        return Err(BraheError::Error(
            "window_seconds must be positive".to_string(),
        ));
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

        // Convert each sample to equinoctial; average a,h,k,p,q; detrend l.
        let mut sum = [0.0f64; 5]; // a,h,k,p,q
        let mut ts = Vec::with_capacity(idx.len());
        let mut ls = Vec::with_capacity(idx.len());
        for &j in &idx {
            let eqn = state_koe_to_equinoctial(&states_rad[j], AngleFormat::Radians, fr);
            for (c, s) in sum.iter_mut().enumerate() {
                *s += eqn[c];
            }
            ts.push(epochs[j] - anchor); // seconds relative to anchor
            ls.push(eqn[5]);
        }
        let n = idx.len() as f64;
        let avg: Vec<f64> = sum.iter().map(|s| s / n).collect();

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
// Not yet called from non-test production code; wired up by a later task.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn numerical_mean_to_osc(
    epochs: &[Epoch],
    mean_states_rad: &[SVector<f64, 6>],
    config: &NumericalConfig,
) -> Result<Vec<(Epoch, SVector<f64, 6>)>, BraheError> {
    let inverse = config.inverse.as_ref().ok_or_else(|| {
        BraheError::Error(
            "numerical mean-to-osc requires NumericalConfig.inverse (dynamics)".to_string(),
        )
    })?;
    if epochs.len() != mean_states_rad.len() {
        return Err(BraheError::Error(
            "epochs and states must have equal length".to_string(),
        ));
    }

    let mut out = Vec::with_capacity(epochs.len());
    for (k, &t) in epochs.iter().enumerate() {
        let target = mean_states_rad[k];

        // Seed with the analytical Brouwer-Lyddane inverse (radians in/out).
        let mut x = state_koe_mean_to_osc(
            &target,
            MeanElementMethod::BrouwerLyddane,
            AngleFormat::Radians,
        )?;

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
/// window bracketing `t` and returns the numerically averaged mean elements nearest
/// `t`.
///
/// Samples 100 evenly-spaced epochs across
/// `[t - 0.6*W, t + 0.6*W]`, where `W = config.window_seconds`, i.e. a window 20%
/// wider on each side than the averaging window itself. This ensures the
/// centered-`W` window at `t` (which [`numerical_osc_to_mean`] needs to average
/// correctly regardless of `config.alignment`) is comfortably supported by the
/// sampled trajectory, avoiding bit-exact-boundary fragility.
///
/// The candidate state is first propagated backward to the window start (to obtain a
/// valid initial condition there), then a fresh propagator integrates forward across
/// the full window so the sampled trajectory has a single, monotonically increasing
/// time history suitable for interpolation.
fn forward_average(
    t: Epoch,
    x_rad: &SVector<f64, 6>,
    config: &NumericalConfig,
    inverse: &InverseConfig,
) -> Result<SVector<f64, 6>, BraheError> {
    const N_SAMPLES: usize = 100;

    let half_width = 0.6 * config.window_seconds;
    let start = t - half_width;
    let end = t + half_width;

    let cart_t = state_koe_to_eci(*x_rad, AngleFormat::Radians);

    // Propagate backward from t to the window start to obtain a valid initial state
    // there.
    let mut back = DNumericalOrbitPropagator::builder()
        .epoch(t)
        .state(DVector::from_row_slice(cart_t.as_slice()))
        .force_config(inverse.force_model.clone())
        .propagation_config(inverse.propagation.clone())
        .build()?;
    back.propagate_to(start);
    let cart_start = back.state(start)?;

    // Integrate forward across the full window from a single starting condition, so
    // the trajectory samples are in strictly increasing epoch order.
    let mut fwd = DNumericalOrbitPropagator::builder()
        .epoch(start)
        .state(cart_start)
        .force_config(inverse.force_model.clone())
        .propagation_config(inverse.propagation.clone())
        .build()?;
    fwd.propagate_to(end);

    let mut sample_epochs = Vec::with_capacity(N_SAMPLES);
    let mut sample_states = Vec::with_capacity(N_SAMPLES);
    for j in 0..N_SAMPLES {
        let frac = j as f64 / (N_SAMPLES as f64 - 1.0);
        let e = start + frac * (end - start);
        let cart = fwd.state(e)?;
        let cart6 = SVector::<f64, 6>::from_row_slice(cart.as_slice());
        let koe = state_eci_to_koe(cart6, AngleFormat::Radians);
        sample_epochs.push(e);
        sample_states.push(koe);
    }

    let averaged = numerical_osc_to_mean(&sample_epochs, &sample_states, config)?;
    averaged
        .into_iter()
        .min_by(|a, b| {
            (a.0 - t)
                .abs()
                .partial_cmp(&(b.0 - t).abs())
                .expect("epoch difference is never NaN")
        })
        .map(|(_, s)| s)
        .ok_or_else(|| BraheError::NumericalError("empty averaging window".to_string()))
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

    #[test]
    #[parallel]
    fn test_numerical_osc_to_mean_recovers_mean_ae_i() {
        let (epochs, states, mean_rad) = synthetic_osc_series();
        let period = crate::orbits::orbital_period(mean_rad[0]);
        let cfg = NumericalConfig {
            window_seconds: period * 0.6,
            alignment: WindowAlignment::Centered,
            edge: EdgeHandling::Truncate,
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
        let cfg = NumericalConfig {
            window_seconds: period * 0.5,
            alignment: WindowAlignment::Centered,
            edge: EdgeHandling::Truncate,
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
        let base = |edge| NumericalConfig {
            window_seconds: period / 2.0,
            alignment: WindowAlignment::Centered,
            edge,
            inverse: None,
        };
        let trunc = numerical_osc_to_mean(&epochs, &states, &base(EdgeHandling::Truncate)).unwrap();
        let preserve =
            numerical_osc_to_mean(&epochs, &states, &base(EdgeHandling::PreserveWindow)).unwrap();
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
        let inverse = InverseConfig {
            force_model: ForceModelConfig::earth_gravity(),
            propagation: NumericalPropagationConfig::default(),
            tolerance: 1.0, // mixed-norm tolerance; see mixed_norm()
            max_iterations: 25,
        };
        let cfg = NumericalConfig {
            window_seconds: period,
            alignment: WindowAlignment::Centered,
            edge: EdgeHandling::PreserveWindow,
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
        back.propagate_to(start);
        let cart_start = back.state(start).unwrap();
        let mut fwd = DNumericalOrbitPropagator::builder()
            .epoch(start)
            .state(cart_start)
            .force_config(inv.force_model.clone())
            .propagation_config(inv.propagation.clone())
            .build()
            .unwrap();
        fwd.propagate_to(end);

        let n = 100usize;
        let mut sample_epochs = Vec::with_capacity(n);
        let mut sample_states = Vec::with_capacity(n);
        for j in 0..n {
            let frac = j as f64 / (n as f64 - 1.0);
            let e = start + frac * (end - start);
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
            .min_by(|a, b| {
                (a.0 - anchor)
                    .abs()
                    .partial_cmp(&(b.0 - anchor).abs())
                    .unwrap()
            })
            .unwrap();
        assert_abs_diff_eq!(mean_recovered[0], mean_rad[0], epsilon = 500.0);
        assert_abs_diff_eq!(mean_recovered[1], mean_rad[1], epsilon = 2e-3);
        assert_abs_diff_eq!(mean_recovered[2], mean_rad[2], epsilon = 2e-3);
    }

    #[test]
    #[serial_test::serial]
    fn test_numerical_mean_to_osc_requires_inverse_config() {
        let mean_rad = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 0.78, 0.5, 1.0, 0.0);
        let cfg = NumericalConfig {
            window_seconds: 5400.0,
            alignment: WindowAlignment::Centered,
            edge: EdgeHandling::PreserveWindow,
            inverse: None,
        };
        let epochs = vec![Epoch::from_gps_seconds(0.0)];
        assert!(numerical_mean_to_osc(&epochs, &[mean_rad], &cfg).is_err());
    }
}
