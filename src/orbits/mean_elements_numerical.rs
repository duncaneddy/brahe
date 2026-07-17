/*!
 * Numerical mean-element averaging.
 *
 * Osculating states are averaged over a moving window in equinoctial space
 * (singularity-safe at e→0, i→0). Slow components a,h,k,p,q are arithmetic-averaged;
 * the fast mean-longitude l is linearly detrended and evaluated at the anchor epoch.
 */

use crate::constants::AngleFormat;
use crate::orbits::equinoctial::{state_equinoctial_to_koe, state_koe_to_equinoctial};
use crate::orbits::mean_elements::{EdgeHandling, NumericalConfig, WindowAlignment};
use crate::time::Epoch;
use crate::utils::errors::BraheError;
use nalgebra::SVector;

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
    for &anchor in epochs {
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

        // Convert each sample to equinoctial; average a,h,k,p,q; detrend l.
        let mut sum = [0.0f64; 5]; // a,h,k,p,q
        let mut ts = Vec::with_capacity(idx.len());
        let mut ls = Vec::with_capacity(idx.len());
        for &j in &idx {
            let fr = if states_rad[j][2] > std::f64::consts::FRAC_PI_2 * 1.999 {
                -1
            } else {
                1
            };
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
        // Recover Keplerian using fr from the averaged (p,q) implied inclination sign.
        let fr = if (avg[3] * avg[3] + avg[4] * avg[4]).sqrt() > 1.0 {
            -1
        } else {
            1
        };
        let koe = state_equinoctial_to_koe(&eqn_mean, AngleFormat::Radians, fr);
        out.push((anchor, koe));
    }
    Ok(out)
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
    fn synthetic_osc_series() -> (Vec<Epoch>, Vec<SVector<f64, 6>>, SVector<f64, 6>) {
        let mean_deg = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 0.0);
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

    #[test]
    #[parallel]
    fn test_numerical_osc_to_mean_recovers_mean_ae_i() {
        let (epochs, states, mean_rad) = synthetic_osc_series();
        let period = crate::orbits::orbital_period(mean_rad[0]);
        let cfg = NumericalConfig {
            window_seconds: period,
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
}
