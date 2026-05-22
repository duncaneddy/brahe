//! Orbital-element conversions using Nyx/ANISE's Orbit type.
//!
//! ## Units and conventions
//!
//! Brahe input/output:
//!   - Semi-major axis: **meters** (a_m)
//!   - Angles: degrees (i, raan, argp)
//!   - Anomaly: **mean anomaly** M in degrees
//!   - Cartesian: **meters** and **m/s**
//!
//! ANISE Orbit:
//!   - Semi-major axis: **km**
//!   - Angles: degrees
//!   - Anomaly: **true anomaly** (ta) in degrees
//!   - Cartesian fields: `radius_km` (km) and `velocity_km_s` (km/s)
//!
//! Boundary conversions applied here:
//!   - a: divide by 1000 on input (m → km), multiply by 1000 on output (km → m)
//!   - cartesian position: divide/multiply by 1000 (m ↔ km)
//!   - cartesian velocity: divide/multiply by 1000 (m/s ↔ km/s)
//!   - anomaly: M ↔ ν via ANISE's built-in `try_keplerian_mean_anomaly`
//!     (keplerian_to_cartesian) and `ta_deg()` + our `true_to_mean_anomaly_deg`
//!     helper (cartesian_to_keplerian)
//!
//! ## Frame setup
//!
//! `EARTH_J2000` from `anise::constants::frames` is a bare `Frame` with no mu.
//! We inject Earth's gravitational parameter (398600.4418 km³/s²) via
//! `Frame::with_mu_km3_s2(...)`. This is a pure in-memory operation — no
//! Almanac download or file I/O is required — which is appropriate for the
//! purely geometric element ↔ Cartesian conversions tested here.
//!
//! Task 9 will set up a shared `OnceLock<Almanac>` for tasks that need a full
//! planetary ephemeris; this module deliberately avoids that dependency.

use anise::constants::frames::EARTH_J2000;
use anise::prelude::Orbit;
use hifitime::Epoch;
use std::time::Instant;

/// Earth gravitational parameter in km³/s² (EGM-2008 value, consistent with brahe).
const GM_EARTH_KM3_S2: f64 = 3.986004418e5;

/// Keplerian elements → Cartesian state.
///
/// Input:  `params["elements"]` = [[a_m, e, i_deg, raan_deg, argp_deg, M_deg], ...]
/// Output: [[x_m, y_m, z_m, vx_m_s, vy_m_s, vz_m_s], ...]
pub fn keplerian_to_cartesian(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let elements: Vec<Vec<f64>> =
        serde_json::from_value(params["elements"].clone()).unwrap();

    // Earth J2000 frame with mu injected — constructed once, copy-cheap.
    let frame = EARTH_J2000.with_mu_km3_s2(GM_EARTH_KM3_S2);

    // A fixed J2000 epoch (2000-01-01 12:00:00 TDB). The element-to-Cartesian
    // conversion is epoch-independent; we just need a valid Epoch placeholder.
    let epoch = Epoch::from_gregorian_tai(2000, 1, 1, 12, 0, 0, 0);

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results: Vec<Vec<f64>> = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(elements.len());

        for oe in &elements {
            // oe = [a_m, e, i_deg, raan_deg, argp_deg, M_deg]
            let a_km = oe[0] / 1000.0;
            let ecc = oe[1];
            let inc_deg = oe[2];
            let raan_deg = oe[3];
            let aop_deg = oe[4];
            let ma_deg = oe[5];

            let orbit = Orbit::try_keplerian_mean_anomaly(
                a_km, ecc, inc_deg, raan_deg, aop_deg, ma_deg, epoch, frame,
            )
            .expect("keplerian_to_cartesian: invalid orbit elements");

            // Convert km → m and km/s → m/s
            let r = orbit.radius_km;
            let v = orbit.velocity_km_s;
            results.push(vec![
                r[0] * 1000.0,
                r[1] * 1000.0,
                r[2] * 1000.0,
                v[0] * 1000.0,
                v[1] * 1000.0,
                v[2] * 1000.0,
            ]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

/// Cartesian state → Keplerian elements.
///
/// Input:  `params["states"]` = [[x_m, y_m, z_m, vx_m_s, vy_m_s, vz_m_s], ...]
/// Output: [[a_m, e, i_deg, raan_deg, argp_deg, M_deg], ...]
pub fn cartesian_to_keplerian(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let states: Vec<Vec<f64>> =
        serde_json::from_value(params["states"].clone()).unwrap();

    let frame = EARTH_J2000.with_mu_km3_s2(GM_EARTH_KM3_S2);
    let epoch = Epoch::from_gregorian_tai(2000, 1, 1, 12, 0, 0, 0);

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results: Vec<Vec<f64>> = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(states.len());

        for state in &states {
            // state = [x_m, y_m, z_m, vx_m_s, vy_m_s, vz_m_s]
            let orbit = Orbit::cartesian(
                state[0] / 1000.0,
                state[1] / 1000.0,
                state[2] / 1000.0,
                state[3] / 1000.0,
                state[4] / 1000.0,
                state[5] / 1000.0,
                epoch,
                frame,
            );

            let a_m = orbit.sma_km().expect("cartesian_to_keplerian: sma") * 1000.0;
            let ecc = orbit.ecc().expect("cartesian_to_keplerian: ecc");
            let inc_deg = orbit.inc_deg().expect("cartesian_to_keplerian: inc");
            let raan_deg = orbit.raan_deg().expect("cartesian_to_keplerian: raan");
            let aop_deg = orbit.aop_deg().expect("cartesian_to_keplerian: aop");
            let ta_deg = orbit.ta_deg().expect("cartesian_to_keplerian: ta");
            let ma_deg = true_to_mean_anomaly_deg(ta_deg, ecc);

            results.push(vec![a_m, ecc, inc_deg, raan_deg, aop_deg, ma_deg]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

/// True anomaly → mean anomaly (degrees).
///
/// Uses the standard half-angle substitution: E = 2·atan2(√(1−e)·sin(ν/2), √(1+e)·cos(ν/2))
/// then M = E − e·sin(E), normalised to [0, 360).
///
/// `pub(crate)` so Task 11 (Keplerian propagation) can reuse it.
pub(crate) fn true_to_mean_anomaly_deg(ta_deg: f64, e: f64) -> f64 {
    let nu = ta_deg.to_radians();
    // Eccentric anomaly via half-angle
    let big_e = f64::atan2(
        (1.0 - e * e).sqrt() * nu.sin(),
        e + nu.cos(),
    );
    let m = big_e - e * big_e.sin();
    m.to_degrees().rem_euclid(360.0)
}

/// Mean anomaly → true anomaly (degrees).
///
/// Newton iteration on Kepler's equation: E − e·sin(E) = M.
/// Converges to machine precision in ≤ 50 iterations for any e < 1.
///
/// `pub(crate)` for potential reuse in Task 11.
pub(crate) fn mean_to_true_anomaly_deg(m_deg: f64, e: f64) -> f64 {
    let m = m_deg.to_radians();
    let mut big_e = if e < 0.8 { m } else { std::f64::consts::PI };
    for _ in 0..50 {
        let f = big_e - e * big_e.sin() - m;
        let fp = 1.0 - e * big_e.cos();
        let delta = f / fp;
        big_e -= delta;
        if delta.abs() < 1e-12 {
            break;
        }
    }
    let nu = f64::atan2(
        (1.0 + e).sqrt() * (big_e / 2.0).sin(),
        (1.0 - e).sqrt() * (big_e / 2.0).cos(),
    ) * 2.0;
    nu.to_degrees().rem_euclid(360.0)
}
