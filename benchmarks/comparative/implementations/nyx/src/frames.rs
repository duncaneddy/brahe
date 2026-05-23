//! ECI ↔ ECEF state transformations using ANISE's Almanac.
//!
//! ECI maps to ANISE's EARTH_J2000 (GCRF-equivalent). ECEF maps to
//! EARTH_ITRF93 (the high-precision Earth body-fixed frame from
//! earth_latest_high_prec.bpc, auto-downloaded by MetaAlmanac::latest()).
//!
//! Per the spec, ANISE's default BPC-based rotation chain is the variable
//! under test — accuracy residual vs Orekit's IERS+IAU2000A precession-
//! nutation algorithm reflects that algorithmic difference.

use crate::data::almanac;
use anise::constants::frames::{EARTH_ITRF93, EARTH_J2000};
use anise::math::cartesian::CartesianState;
use anise::prelude::Aberration;
use hifitime::Epoch;
use std::time::Instant;

#[derive(serde::Deserialize)]
struct FrameCase {
    jd: f64,
    state: Vec<f64>,
}

fn parse_cases(params: &serde_json::Value) -> Vec<FrameCase> {
    serde_json::from_value(params["cases"].clone()).unwrap()
}

pub fn eci_to_ecef(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let cases = parse_cases(params);
    let alm = almanac();

    // Pre-build CartesianState inputs in EARTH_J2000 (ECI).
    // Input state is in m / m·s⁻¹; ANISE uses km / km·s⁻¹ internally.
    let inputs: Vec<CartesianState> = cases
        .iter()
        .map(|c| {
            let epoch = Epoch::from_jde_utc(c.jd);
            CartesianState::new(
                c.state[0] / 1e3, // x: m → km
                c.state[1] / 1e3, // y: m → km
                c.state[2] / 1e3, // z: m → km
                c.state[3] / 1e3, // vx: m/s → km/s
                c.state[4] / 1e3, // vy: m/s → km/s
                c.state[5] / 1e3, // vz: m/s → km/s
                epoch,
                EARTH_J2000,
            )
        })
        .collect();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(inputs.len());

        for state in &inputs {
            let out = alm
                .transform_to(*state, EARTH_ITRF93, None::<Aberration>)
                .expect("ECI → ECEF transform must succeed");

            results.push(vec![
                out.radius_km.x * 1e3,    // km → m
                out.radius_km.y * 1e3,
                out.radius_km.z * 1e3,
                out.velocity_km_s.x * 1e3, // km/s → m/s
                out.velocity_km_s.y * 1e3,
                out.velocity_km_s.z * 1e3,
            ]);
        }

        all_times.push(start.elapsed().as_secs_f64());
        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn ecef_to_eci(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let cases = parse_cases(params);
    let alm = almanac();

    // Pre-build CartesianState inputs in EARTH_ITRF93 (ECEF).
    // Input state is in m / m·s⁻¹; ANISE uses km / km·s⁻¹ internally.
    let inputs: Vec<CartesianState> = cases
        .iter()
        .map(|c| {
            let epoch = Epoch::from_jde_utc(c.jd);
            CartesianState::new(
                c.state[0] / 1e3, // x: m → km
                c.state[1] / 1e3, // y: m → km
                c.state[2] / 1e3, // z: m → km
                c.state[3] / 1e3, // vx: m/s → km/s
                c.state[4] / 1e3, // vy: m/s → km/s
                c.state[5] / 1e3, // vz: m/s → km/s
                epoch,
                EARTH_ITRF93,
            )
        })
        .collect();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(inputs.len());

        for state in &inputs {
            let out = alm
                .transform_to(*state, EARTH_J2000, None::<Aberration>)
                .expect("ECEF → ECI transform must succeed");

            results.push(vec![
                out.radius_km.x * 1e3,    // km → m
                out.radius_km.y * 1e3,
                out.radius_km.z * 1e3,
                out.velocity_km_s.x * 1e3, // km/s → m/s
                out.velocity_km_s.y * 1e3,
                out.velocity_km_s.z * 1e3,
            ]);
        }

        all_times.push(start.elapsed().as_secs_f64());
        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}
