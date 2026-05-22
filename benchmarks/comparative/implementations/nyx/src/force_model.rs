//! Function-level acceleration benchmarks: single accel vector per state.
//!
//! ## Task decisions
//!
//! ### `accel_point_mass_gravity` — DROPPED
//! Nyx 2.3.1 has no standalone point-mass API for a single (epoch, position)
//! pair.  The two-body central-body term is evaluated inside
//! `OrbitalDynamics::eom()` as `(-mu/|r|^3) * r`, with `mu` read from
//! `frame.mu_km3_s2()` (ANISE frame constant, not an ephemeris call).
//! Exposing that formula as a benchmark would be pure hand-rolled physics with
//! a single constant lookup — not a meaningful Nyx/ANISE benchmark.
//!
//! ### `accel_spherical_harmonics_20` / `_80` — IMPLEMENTED
//! `Harmonics::eom(&orbit, almanac)` is the Nyx API that evaluates the full
//! GMAT-algorithm spherical-harmonic acceleration: it calls
//! `almanac.transform_to(orbit, iau_earth, None)` to rotate the state into
//! the body-fixed frame, runs the normalized Legendre polynomial recursion,
//! then calls `almanac.rotate(iau_earth, j2000, epoch)` to rotate the
//! resulting acceleration back to J2000.  Both calls are real ANISE operations.
//!
//! ### `accel_third_body_sun` / `_moon` — IMPLEMENTED
//! `PointMasses::eom(&orbit, almanac)` is the Nyx API.  Internally it calls
//! `almanac.transform(third_body_frame, osc.frame, epoch, correction)` to
//! obtain the body position from ANISE's de440s ephemeris, then evaluates
//! the Battin third-body formula.  The `almanac.transform` call is a real
//! ANISE SPK lookup — this is exactly what makes the benchmark meaningful.

use anise::constants::celestial_objects::{MOON, SUN};
use anise::constants::frames::{EARTH_J2000, IAU_EARTH_FRAME};
use anise::math::Vector3;
use anise::prelude::Orbit;
use hifitime::Epoch;
use nyx_space::dynamics::orbital::PointMasses;
use nyx_space::dynamics::{AccelModel, Harmonics};
use nyx_space::io::gravity::HarmonicsMem;
use std::sync::Arc;
use std::time::Instant;

/// Gravitational parameter of Earth (EGM2008) in km³/s² — matches brahe.
const GM_EARTH_KM3_S2: f64 = 3.986004418e5;

// ── Shared input/output helpers ──────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct Case {
    jd: f64,
    state_eci: Vec<f64>,
}

/// Build an `Orbit` (= ANISE `CartesianState`) in J2000 from a Julian Date
/// (UTC) and a 6-element ECI state vector `[x,y,z,vx,vy,vz]` in metres.
///
/// The Harmonics and PointMasses `eom` methods only read `radius_km`,
/// `epoch`, and `frame` from the orbit, so the velocity slots (set to
/// `vx/vy/vz` from input) do not affect the acceleration result.
#[inline]
fn orbit_from_state(jd: f64, state: &[f64]) -> Orbit {
    let epoch = Epoch::from_jde_utc(jd);
    let frame = EARTH_J2000.with_mu_km3_s2(GM_EARTH_KM3_S2);
    Orbit {
        radius_km: Vector3::new(
            state[0] * 1e-3,
            state[1] * 1e-3,
            state[2] * 1e-3,
        ),
        velocity_km_s: Vector3::new(
            state[3] * 1e-3,
            state[4] * 1e-3,
            state[5] * 1e-3,
        ),
        epoch,
        frame,
    }
}

/// Decode the cases array from a `{cases: [...]}` accuracy-path input.
fn maybe_cases(params: &serde_json::Value) -> Option<Vec<Case>> {
    let cases_val = params.get("cases")?;
    serde_json::from_value(cases_val.clone()).ok()
}

/// Decode the perf path: `{jd, state_eci, n_samples}`.
fn single_state(params: &serde_json::Value) -> (f64, Vec<f64>, usize) {
    let jd: f64 = serde_json::from_value(params["jd"].clone()).unwrap();
    let state_eci: Vec<f64> = serde_json::from_value(params["state_eci"].clone()).unwrap();
    let n_samples: usize = serde_json::from_value(params["n_samples"].clone()).unwrap();
    (jd, state_eci, n_samples)
}

/// Run a force-model evaluator in either perf mode (single IC, inner loop) or
/// accuracy mode (multi-IC sweep).  Returns `(times_per_iter, result_vectors)`
/// where each result row is `[ax, ay, az]` in m/s².
fn run_force_model<F>(
    params: &serde_json::Value,
    iterations: usize,
    mut eval: F,
) -> (Vec<f64>, serde_json::Value)
where
    F: FnMut(f64, &[f64]) -> [f64; 3],
{
    let mut all_times = Vec::with_capacity(iterations);
    let mut first_result: Vec<Vec<f64>> = Vec::new();

    if let Some(cases) = maybe_cases(params) {
        for iter in 0..iterations {
            let start = Instant::now();
            let mut out = Vec::with_capacity(cases.len());
            for case in &cases {
                let a = eval(case.jd, &case.state_eci);
                out.push(vec![a[0], a[1], a[2]]);
            }
            all_times.push(start.elapsed().as_secs_f64());
            if iter == 0 {
                first_result = out;
            }
        }
    } else {
        let (jd, state_eci, n_samples) = single_state(params);
        for iter in 0..iterations {
            let start = Instant::now();
            let mut a = [0.0f64; 3];
            for _ in 0..n_samples {
                a = eval(jd, &state_eci);
            }
            all_times.push(start.elapsed().as_secs_f64());
            if iter == 0 {
                first_result = vec![vec![a[0], a[1], a[2]]];
            }
        }
    }

    (all_times, serde_json::to_value(first_result).unwrap())
}

// ── Load EGM2008 coefficients (reused from propagation.rs) ──────────────────

/// Load EGM2008 spherical-harmonic coefficients from the ICGEM `.gfc` file
/// up to the requested `degree` × `order`.
///
/// Mirrors `load_egm2008_icgem` in `propagation.rs`; duplicated here so
/// `force_model.rs` has no intra-module dependency.
fn load_egm2008_icgem(degree: usize, order: usize) -> HarmonicsMem {
    let path = std::env::var("BRAHE_GRAVITY_FILE")
        .unwrap_or_else(|_| "data/gravity_models/EGM2008_360.gfc".to_string());

    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Cannot read gravity file {path}: {e}"));

    let mut c_nm = nyx_space::linalg::DMatrix::from_element(degree + 1, degree + 1, 0.0_f64);
    let mut s_nm = nyx_space::linalg::DMatrix::from_element(degree + 1, degree + 1, 0.0_f64);

    for line in content.lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() || tokens[0].to_ascii_lowercase() != "gfc" {
            continue;
        }
        if tokens.len() < 5 {
            continue;
        }
        let n: usize = match tokens[1].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if n > degree {
            break;
        }
        let m: usize = match tokens[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if m > order {
            continue;
        }
        let c: f64 = match tokens[3].replace(['D', 'd'], "E").parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let s: f64 = match tokens[4].replace(['D', 'd'], "E").parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        c_nm[(n, m)] = c;
        s_nm[(n, m)] = s;
    }

    let mut shadr_lines = String::from("SHADR_HEADER\n");
    for n in 0..=degree {
        for m in 0..=n.min(order) {
            shadr_lines.push_str(&format!(
                "{} {} {:.18E} {:.18E}\n",
                n,
                m,
                c_nm[(n, m)],
                s_nm[(n, m)]
            ));
        }
    }

    let tmp_path = format!(
        "{}/nyx_egm2008_force_{}x{}.shadr",
        std::env::temp_dir().display(),
        degree,
        order
    );
    std::fs::write(&tmp_path, &shadr_lines)
        .unwrap_or_else(|e| panic!("Cannot write temp SHADR file {tmp_path}: {e}"));

    let mem = HarmonicsMem::from_shadr(&tmp_path, degree, order, false)
        .unwrap_or_else(|e| panic!("HarmonicsMem::from_shadr failed for {tmp_path}: {e:?}"));

    let _ = std::fs::remove_file(&tmp_path);
    mem
}

// ── Spherical-harmonic acceleration benchmarks ───────────────────────────────

fn accel_spherical_harmonics_run(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let degree: usize = serde_json::from_value(params["degree"].clone()).unwrap();
    let order: usize = serde_json::from_value(params["order"].clone()).unwrap();

    let almanac = Arc::new(crate::data::almanac().clone());

    // IAU Earth body-fixed frame: `Harmonics::eom` transforms the orbit into
    // this frame via `almanac.transform_to(...)`, runs the SH recursion, then
    // rotates the result back to J2000 via `almanac.rotate(...)`.
    let iau_earth = almanac
        .frame_info(IAU_EARTH_FRAME)
        .expect("IAU_EARTH_FRAME not found — PCK11 kernel must be loaded");

    let stor = load_egm2008_icgem(degree, order);
    // `Harmonics::from_stor` pre-computes all B_nm, C_nm, vr01, vr11 tables —
    // same setup cost as in propagation; done once outside the timing loop.
    let harmonics = Harmonics::from_stor(iau_earth, stor);

    run_force_model(params, iterations, |jd, state| {
        let osc = orbit_from_state(jd, state);

        // `Harmonics::eom` is the core Nyx SH evaluator: it handles frame
        // transformation (ANISE), Legendre recursion, and result rotation.
        // It returns the perturbation only (degrees 2+; C[0,0] and C[1,0]=0
        // are excluded by the GMAT algorithm design), matching how Nyx uses
        // it inside OrbitalDynamics.
        let pert_km_s2 = harmonics
            .eom(&osc, almanac.clone())
            .expect("Harmonics::eom failed");

        // Add the two-body (central-body) term to match the convention used by
        // brahe and Orekit, which return the full gravitational acceleration
        // (central body + perturbations). The two-body term is:
        //   a_2body = -mu/|r|^3 * r
        // where mu is read from `orbit.frame.mu_km3_s2()` — an ANISE frame
        // constant call, the same call OrbitalDynamics makes in its own EOM.
        let mu = osc
            .frame
            .mu_km3_s2()
            .expect("mu_km3_s2: frame missing mu");
        let r3 = osc.rmag_km().powi(3);
        let two_body = osc.radius_km * (-mu / r3);

        let total = pert_km_s2 + two_body;
        // Convert km/s² → m/s²
        [total[0] * 1e3, total[1] * 1e3, total[2] * 1e3]
    })
}

/// 20×20 EGM2008 spherical-harmonic acceleration via `Harmonics::eom`.
///
/// ANISE API invoked: `almanac.transform_to(orbit, IAU_EARTH, None)` +
/// `almanac.rotate(IAU_EARTH, J2000, epoch)` inside `Harmonics::eom`.
pub fn accel_spherical_harmonics_20(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    accel_spherical_harmonics_run(params, iterations)
}

/// 80×80 EGM2008 spherical-harmonic acceleration via `Harmonics::eom`.
///
/// Same API as `accel_spherical_harmonics_20`; only the coefficient truncation
/// changes (degree = order = 80 passed in `params`).
pub fn accel_spherical_harmonics_80(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    accel_spherical_harmonics_run(params, iterations)
}

// ── Third-body acceleration benchmarks ───────────────────────────────────────

/// Sun third-body acceleration via `PointMasses::eom` with ANISE de440s.
///
/// ANISE API invoked: `almanac.transform(sun_frame, earth_j2000, epoch, None)`
/// inside `PointMasses::eom` to obtain the Sun's position relative to Earth.
/// The resulting acceleration vector is the canonical Battin third-body formula
/// applied to that ephemeris-sourced position — the data comes from ANISE.
pub fn accel_third_body_sun(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let almanac = Arc::new(crate::data::almanac().clone());
    // Build the PointMasses model for the Sun only (NAIF 10).
    let sun_pm = PointMasses::new(vec![SUN]);

    run_force_model(params, iterations, |jd, state| {
        let osc = orbit_from_state(jd, state);
        // `PointMasses::eom` calls `almanac.transform(sun_frame, earth_j2000,
        // epoch, None)` — a real ANISE de440s ephemeris lookup.
        let accel_km_s2 = sun_pm
            .eom(&osc, almanac.clone())
            .expect("PointMasses(Sun)::eom failed");
        [
            accel_km_s2[0] * 1e3,
            accel_km_s2[1] * 1e3,
            accel_km_s2[2] * 1e3,
        ]
    })
}

/// Moon third-body acceleration via `PointMasses::eom` with ANISE de440s.
///
/// ANISE API invoked: `almanac.transform(moon_frame, earth_j2000, epoch, None)`
/// inside `PointMasses::eom` to obtain the Moon's position relative to Earth.
pub fn accel_third_body_moon(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let almanac = Arc::new(crate::data::almanac().clone());
    // Build the PointMasses model for the Moon only (NAIF 301).
    let moon_pm = PointMasses::new(vec![MOON]);

    run_force_model(params, iterations, |jd, state| {
        let osc = orbit_from_state(jd, state);
        // `PointMasses::eom` calls `almanac.transform(moon_frame, earth_j2000,
        // epoch, None)` — a real ANISE de440s ephemeris lookup.
        let accel_km_s2 = moon_pm
            .eom(&osc, almanac.clone())
            .expect("PointMasses(Moon)::eom failed");
        [
            accel_km_s2[0] * 1e3,
            accel_km_s2[1] * 1e3,
            accel_km_s2[2] * 1e3,
        ]
    })
}
