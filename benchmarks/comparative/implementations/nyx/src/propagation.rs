//! Orbit propagation benchmarks — analytic Keplerian and numerical propagation.
//!
//! ## Keplerian API used
//!
//! `Orbit::at_epoch` (ANISE 0.10.1, `src/astro/orbit.rs`) performs
//! closed-form two-body Keplerian propagation: it advances the mean anomaly
//! by Δt using the analytic mean-motion formula, then reconstructs the full
//! Cartesian state via `try_keplerian_mean_anomaly`.  No integrator is
//! invoked; this is equivalent to brahe-Rust's `KeplerianPropagator`.
//!
//! ## Unit conventions (matching brahe-Rust)
//!
//! Input elements: `[a_m, e, i_deg, raan_deg, argp_deg, M_deg]`
//!   – semi-major axis in **meters**, angles in degrees, anomaly is **mean** (M).
//! Output state:   `[x, y, z, vx, vy, vz]` in **meters** and **m/s**.
//! ANISE internals use km / km s⁻¹; boundary conversions applied at entry/exit.

use anise::almanac::Almanac;
use anise::constants::celestial_objects::{MOON, SUN};
use anise::constants::frames::{EARTH_J2000, IAU_EARTH_FRAME};
use anise::prelude::Orbit;
use hifitime::{Duration, Epoch};
use nyx_space::cosmic::Spacecraft;
use nyx_space::dynamics::orbital::{OrbitalDynamics, PointMasses};
use nyx_space::dynamics::{Drag, GravityField, SolarPressure, SpacecraftDynamics};
use nyx_space::io::gravity::GravityFieldData;
use nyx_space::propagators::{IntegratorMethod, IntegratorOptions, Propagator};
use std::sync::Arc;
use std::time::Instant;

/// Earth gravitational parameter in km³/s² (EGM-2008, matching brahe).
const GM_EARTH_KM3_S2: f64 = 3.986004418e5;

/// Earth J2000 frame with mu injected — copy-cheap, no Almanac I/O needed.
#[inline]
fn earth_frame() -> anise::prelude::Frame {
    EARTH_J2000.with_mu_km3_s2(GM_EARTH_KM3_S2)
}

/// Build an `Orbit` from the brahe element convention.
///
/// Input: `[a_m, e, i_deg, raan_deg, argp_deg, M_deg]`
#[inline]
fn orbit_from_elements(oe: &[f64], epoch: Epoch) -> Orbit {
    let frame = earth_frame();
    Orbit::try_keplerian_mean_anomaly(
        oe[0] / 1000.0, // a: m → km
        oe[1],          // e (dimensionless)
        oe[2],          // i (deg)
        oe[3],          // raan (deg)
        oe[4],          // argp (deg)
        oe[5],          // M (deg)
        epoch,
        frame,
    )
    .expect("orbit_from_elements: invalid Keplerian elements")
}

/// Extract `[x_m, y_m, z_m, vx_m_s, vy_m_s, vz_m_s]` from an `Orbit`.
#[inline]
fn state_vec(orb: &Orbit) -> Vec<f64> {
    let r = orb.radius_km;
    let v = orb.velocity_km_s;
    vec![
        r[0] * 1e3,
        r[1] * 1e3,
        r[2] * 1e3,
        v[0] * 1e3,
        v[1] * 1e3,
        v[2] * 1e3,
    ]
}

/// Propagate each case by its own `dt` seconds and report the final state.
///
/// Input shape: `{ "cases": [{ "jd": f64, "elements": [6×f64], "dt": f64 }, ...] }`
/// Output shape: `[[x,y,z,vx,vy,vz], ...]` — one row per case, metres / m s⁻¹.
pub fn keplerian_single(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    #[derive(serde::Deserialize)]
    struct Case {
        jd: f64,
        elements: Vec<f64>,
        dt: f64,
    }
    let cases: Vec<Case> = serde_json::from_value(params["cases"].clone()).unwrap();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results: Vec<Vec<f64>> = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(cases.len());

        for case in &cases {
            let epoch = Epoch::from_jde_utc(case.jd);
            let initial = orbit_from_elements(&case.elements, epoch);
            let target_epoch = epoch + Duration::from_seconds(case.dt);
            let final_orb = initial
                .at_epoch(target_epoch)
                .expect("keplerian_single: at_epoch failed");
            results.push(state_vec(&final_orb));
        }

        all_times.push(start.elapsed().as_secs_f64());
        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

/// Analytic Keplerian trajectory.
///
/// ## Perf path (single-IC)
/// Input:  `{ "jd": f64, "elements": [6×f64], "step_size": f64, "n_steps": usize }`
/// Output: `[[x,y,z,vx,vy,vz], ...]` — `n_steps` rows at `step_size`-second intervals.
///
/// ## Accuracy path (multi-IC)
/// Input:  `{ "cases": [{ "jd", "elements" }, ...], "step_size": f64, "n_steps": usize }`
/// Output: `[[x,y,z,vx,vy,vz], ...]` — one final-state row per case.
pub fn keplerian_trajectory(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let step_size: f64 = serde_json::from_value(params["step_size"].clone()).unwrap();
    let n_steps: usize = serde_json::from_value(params["n_steps"].clone()).unwrap();

    #[derive(serde::Deserialize)]
    struct Case {
        jd: f64,
        elements: Vec<f64>,
    }
    let cases: Option<Vec<Case>> = params
        .get("cases")
        .map(|v| serde_json::from_value(v.clone()).unwrap());

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results: Vec<Vec<f64>> = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let results: Vec<Vec<f64>>;

        if let Some(cases) = &cases {
            // Accuracy path: propagate each IC to epoch + n_steps*step_size,
            // report one final state per case.
            results = cases
                .iter()
                .map(|case| {
                    let epoch = Epoch::from_jde_utc(case.jd);
                    let initial = orbit_from_elements(&case.elements, epoch);
                    let target_epoch =
                        epoch + Duration::from_seconds((n_steps as f64) * step_size);
                    let final_orb = initial
                        .at_epoch(target_epoch)
                        .expect("keplerian_trajectory accuracy: at_epoch failed");
                    state_vec(&final_orb)
                })
                .collect();
        } else {
            // Perf path: single IC, generate full trajectory of n_steps states.
            let jd: f64 = serde_json::from_value(params["jd"].clone()).unwrap();
            let elements: Vec<f64> = serde_json::from_value(params["elements"].clone()).unwrap();
            let epoch = Epoch::from_jde_utc(jd);
            let initial = orbit_from_elements(&elements, epoch);

            let mut traj = Vec::with_capacity(n_steps);
            for step_idx in 0..n_steps {
                let target_epoch =
                    epoch + Duration::from_seconds((step_idx as f64 + 1.0) * step_size);
                let orb = initial
                    .at_epoch(target_epoch)
                    .expect("keplerian_trajectory perf: at_epoch failed");
                traj.push(state_vec(&orb));
            }
            results = traj;
        }

        all_times.push(start.elapsed().as_secs_f64());
        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}


/// Numerical two-body propagation using Nyx's RK4 fixed-step integrator.
///
/// ## Integrator choice
///
/// `IntegratorOptions::with_fixed_step_s(step_size)` sets `fixed_step = true`
/// and pins the step to exactly `step_size` seconds, mirroring brahe-Rust's
/// `IntegratorConfig::fixed_step(step_size)` with `IntegratorMethod::RK4`.
/// `IntegratorMethod::RungeKutta4` is the Nyx enum variant for the classical
/// 4th-order Runge–Kutta method (4 stages, fixed step — same algorithm as brahe).
///
/// ## Unit conventions (matching brahe-Rust)
///
/// Input elements: `[a_m, e, i_deg, raan_deg, argp_deg, M_deg]`
///   – semi-major axis in **meters**, angles in degrees, anomaly is **mean** (M).
/// Output state:   `[x, y, z, vx, vy, vz]` in **meters** and **m/s**.
///
/// ## Almanac
///
/// `OrbitalDynamics::two_body()` computes gravity solely from the frame's
/// embedded `mu_km3_s2`; no SPK/EOP data are consumed, so `Almanac::default()`
/// (empty) is sufficient.
///
/// ## Perf path (single-IC)
/// Input:  `{ "jd": f64, "elements": [6×f64], "step_size": f64, "n_steps": usize }`
/// Output: `[[x,y,z,vx,vy,vz], ...]` — `n_steps` rows at `step_size`-second intervals.
///
/// ## Accuracy path (multi-IC)
/// Input:  `{ "cases": [{ "jd", "elements" }, ...], "step_size": f64, "n_steps": usize }`
/// Output: `[[x,y,z,vx,vy,vz], ...]` — one final-state row per case.
pub fn numerical_twobody(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let step_size: f64 = serde_json::from_value(params["step_size"].clone()).unwrap();
    let n_steps: usize = serde_json::from_value(params["n_steps"].clone()).unwrap();

    #[derive(serde::Deserialize)]
    struct Case {
        jd: f64,
        elements: Vec<f64>,
    }
    let cases: Option<Vec<Case>> = params
        .get("cases")
        .map(|v| serde_json::from_value(v.clone()).unwrap());

    // Build propagator once — reused across iterations and cases.
    //
    // Nyx's `Propagator` only accepts types that implement `Dynamics`, and in
    // Nyx 2.4.0 the only public implementation is `SpacecraftDynamics`.
    // `OrbitalDynamics::two_body()` provides the pure central-body force model;
    // `SpacecraftDynamics::new()` wraps it. The propagated state is `Spacecraft`,
    // which carries an embedded `orbit: Orbit` accessed via `.orbit`.
    //
    // RungeKutta4: classical fixed-step RK4 — matches brahe's IntegratorMethod::RK4.
    // with_fixed_step_s: sets fixed_step = true and init_step = step_size seconds.
    let orbital_dyn = OrbitalDynamics::two_body();
    let sc_dyn = SpacecraftDynamics::new(orbital_dyn);
    let opts = IntegratorOptions::with_fixed_step_s(step_size);
    let prop = Propagator::new(sc_dyn, IntegratorMethod::RungeKutta4, opts);

    // Empty almanac: two-body gravity only reads frame.mu_km3_s2(), no SPK needed.
    let almanac = Arc::new(Almanac::default());
    let total_dt = (n_steps as f64) * step_size;

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results: Vec<Vec<f64>> = Vec::new();

    for iter in 0..iterations {
        let start = std::time::Instant::now();
        let results: Vec<Vec<f64>>;

        if let Some(cases) = &cases {
            // Accuracy path: propagate each IC forward n_steps * step_size seconds.
            results = cases
                .iter()
                .map(|case| {
                    let epoch = Epoch::from_jde_utc(case.jd);
                    let orbit = orbit_from_elements(&case.elements, epoch);
                    let sc: Spacecraft = orbit.into();
                    let mut instance = prop.with(sc, almanac.clone());
                    let final_sc = instance
                        .for_duration(Duration::from_seconds(total_dt))
                        .expect("numerical_twobody accuracy: propagation failed");
                    state_vec(&final_sc.orbit)
                })
                .collect();
        } else {
            // Perf path: single IC, generate full trajectory of n_steps states.
            let jd: f64 = serde_json::from_value(params["jd"].clone()).unwrap();
            let elements: Vec<f64> =
                serde_json::from_value(params["elements"].clone()).unwrap();
            let epoch = Epoch::from_jde_utc(jd);
            let orbit = orbit_from_elements(&elements, epoch);
            let sc: Spacecraft = orbit.into();
            let mut instance = prop.with(sc, almanac.clone());

            let mut traj = Vec::with_capacity(n_steps);
            for _ in 0..n_steps {
                let stepped = instance
                    .for_duration(Duration::from_seconds(step_size))
                    .expect("numerical_twobody perf: step failed");
                traj.push(state_vec(&stepped.orbit));
            }
            results = traj;
        }

        all_times.push(start.elapsed().as_secs_f64());
        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}


// ── Shared infrastructure for spherical-harmonic propagation ────────────────

/// Load EGM2008 spherical-harmonic coefficients from an ICGEM-format `.gfc`
/// file (the format brahe ships) up to the requested degree and order.
///
/// ICGEM format lines look like:
/// ```text
/// gfc  <degree>  <order>  <C_nm>  <S_nm>  <sigma_C>  <sigma_S>
/// ```
/// Only lines whose first token is `gfc` carry coefficient data; all other
/// lines (header, comments) are skipped.  Exponent character `D`/`d` is
/// replaced with `E` for Fortran-style notation (e.g. `1.0d0` → `1.0E0`).
///
/// `BRAHE_GRAVITY_FILE` env-var overrides the path
/// (`data/gravity_models/EGM2008_120.gfc` relative to the working directory).
///
/// Because `GravityFieldData` has no public constructor for pre-built matrices,
/// the parsed coefficients are written to a temp SHADR-format text file and
/// loaded back via `GravityFieldData::from_shadr`.  The values are numerically
/// identical to the source; only the serialization format differs.
fn load_egm2008_icgem(degree: usize, order: usize) -> GravityFieldData {
    let path = std::env::var("BRAHE_GRAVITY_FILE")
        .unwrap_or_else(|_| "data/gravity_models/EGM2008_120.gfc".to_string());

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
            break; // File is sorted by degree; safe to stop early.
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

    // Serialize to SHADR format for round-trip through GravityFieldData::from_shadr.
    // SHADR layout: one header line, then `<n> <m> <C_nm> <S_nm>` per row.
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
        "{}/nyx_egm2008_{}x{}.shadr",
        std::env::temp_dir().display(),
        degree,
        order
    );
    std::fs::write(&tmp_path, &shadr_lines)
        .unwrap_or_else(|e| panic!("Cannot write temp SHADR file {tmp_path}: {e}"));

    let mem = GravityFieldData::from_shadr(&tmp_path, degree, order, false)
        .unwrap_or_else(|e| panic!("GravityFieldData::from_shadr failed for {tmp_path}: {e:?}"));

    let _ = std::fs::remove_file(&tmp_path);
    mem
}

/// Input schema shared by both RK4 gravity tasks.
///
/// Perf path:  `jd` + `elements_deg`, `step_size`, `n_steps`, gravity/3-body flags.
/// Accuracy path: `cases: [{jd, elements}, ...]`, `step_size`, `n_steps`, same flags.
///
/// `elements` / `elements_deg` layout: `[a_m, e, i_deg, raan_deg, argp_deg, M_deg]`
#[derive(serde::Deserialize)]
struct RK4GravParams {
    #[serde(default)]
    jd: Option<f64>,
    #[serde(default)]
    elements_deg: Option<Vec<f64>>,
    #[serde(default)]
    cases: Option<Vec<RK4GravCase>>,
    step_size: f64,
    n_steps: usize,
    gravity_degree: usize,
    gravity_order: usize,
    #[serde(default)]
    third_body_sun: bool,
    #[serde(default)]
    third_body_moon: bool,
}

#[derive(serde::Deserialize)]
struct RK4GravCase {
    jd: f64,
    elements: Vec<f64>,
}

/// Common RK4 propagation loop for spherical-harmonic + optional third-body tasks.
///
/// Called by both `numerical_rk4_grav5x5` and `numerical_rk4_grav20x20_sun_moon`.
fn numerical_rk4_grav_run(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let p: RK4GravParams = serde_json::from_value(params.clone()).unwrap();

    // Almanac provides IAU Earth frame constants (PCK11) and ephemerides (de440s).
    // Loaded once per process via OnceLock in data.rs.
    let almanac = Arc::new(crate::data::almanac().clone());

    // IAU Earth body-fixed frame: used as the GravityField compute_frame.
    // Nyx evaluates spherical-harmonic accelerations in this frame, then rotates
    // the result back to J2000 (EARTH_J2000) after each step.
    let iau_earth = almanac
        .frame_info(IAU_EARTH_FRAME)
        .expect("IAU_EARTH_FRAME not found — PCK11 kernel must be loaded by MetaAlmanac");

    let stor = load_egm2008_icgem(p.gravity_degree, p.gravity_order);
    let harmonics = GravityField::from_stor(iau_earth, stor);

    let mut orbital_dyn = OrbitalDynamics::from_model(harmonics);

    // Optionally add Sun (NAIF 10) and Moon (NAIF 301) point-mass perturbations.
    let mut third_body_ids: Vec<i32> = Vec::new();
    if p.third_body_sun {
        third_body_ids.push(SUN);
    }
    if p.third_body_moon {
        third_body_ids.push(MOON);
    }
    if !third_body_ids.is_empty() {
        orbital_dyn
            .accel_models
            .push(PointMasses::new(third_body_ids));
    }

    let sc_dyn = SpacecraftDynamics::new(orbital_dyn);
    let opts = IntegratorOptions::with_fixed_step_s(p.step_size);
    let prop = Propagator::new(sc_dyn, IntegratorMethod::RungeKutta4, opts);
    let total_dt = (p.n_steps as f64) * p.step_size;

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results: Vec<Vec<f64>> = Vec::new();

    for iter in 0..iterations {
        let start = std::time::Instant::now();
        let results: Vec<Vec<f64>>;

        if let Some(cases) = &p.cases {
            // Accuracy path: propagate each IC, report final state.
            results = cases
                .iter()
                .map(|case| {
                    let epoch = Epoch::from_jde_utc(case.jd);
                    let orbit = orbit_from_elements(&case.elements, epoch);
                    let sc: Spacecraft = orbit.into();
                    let mut instance = prop.with(sc, almanac.clone());
                    let final_sc = instance
                        .for_duration(Duration::from_seconds(total_dt))
                        .expect("numerical_rk4_grav accuracy: propagation failed");
                    state_vec(&final_sc.orbit)
                })
                .collect();
        } else {
            // Perf path: single IC, full trajectory of n_steps states.
            let jd = p.jd.expect("perf path requires `jd`");
            let elements = p
                .elements_deg
                .as_ref()
                .expect("perf path requires `elements_deg`");
            let epoch = Epoch::from_jde_utc(jd);
            let orbit = orbit_from_elements(elements, epoch);
            let sc: Spacecraft = orbit.into();
            let mut instance = prop.with(sc, almanac.clone());

            let mut traj = Vec::with_capacity(p.n_steps);
            for _ in 0..p.n_steps {
                let stepped = instance
                    .for_duration(Duration::from_seconds(p.step_size))
                    .expect("numerical_rk4_grav perf: step failed");
                traj.push(state_vec(&stepped.orbit));
            }
            results = traj;
        }

        all_times.push(start.elapsed().as_secs_f64());
        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

/// RK4 numerical propagation with 5×5 EGM2008 spherical-harmonic gravity.
///
/// ## Force model
///
/// - Central-body (two-body) gravity from the frame's embedded mu.
/// - 5×5 EGM2008 spherical harmonics evaluated in the IAU Earth body-fixed frame.
/// - No third-body perturbations.
///
/// ## Gravity-file format note
///
/// Nyx 2.4.0 `GravityFieldData` supports COF and SHADR formats; brahe ships EGM2008
/// in ICGEM format.  A transparent round-trip through a temp SHADR file bridges
/// the formats; coefficient values are numerically unchanged.
pub fn numerical_rk4_grav5x5(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    numerical_rk4_grav_run(params, iterations)
}

/// RK4 numerical propagation with 20×20 EGM2008 gravity plus Sun and Moon
/// third-body point masses.
///
/// ## Force model
///
/// - Central-body gravity.
/// - 20×20 EGM2008 spherical harmonics (IAU Earth body-fixed frame).
/// - Sun (NAIF 10) and Moon (NAIF 301) `PointMasses` sourced from the
///   ANISE de440s ephemeris loaded by `MetaAlmanac::latest()`.
pub fn numerical_rk4_grav20x20_sun_moon(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    numerical_rk4_grav_run(params, iterations)
}

/// RK4 numerical propagation with 80×80 EGM2008 gravity + Sun/Moon third-body
/// + atmospheric drag + solar radiation pressure over ~1 LEO revolution.
///
/// ## Force model
///
/// - 80×80 EGM2008 spherical harmonics (IAU Earth body-fixed frame).
/// - Sun (NAIF 10) and Moon (NAIF 301) point-mass perturbations.
/// - Atmospheric drag (see substitution note below).
/// - Solar radiation pressure with conical Earth-shadow eclipse model.
///
/// ## Drag substitution
///
/// **Nyx 2.4.0 does not expose NRLMSISE-00 publicly.**  brahe and Orekit
/// both use NRLMSISE-00 for this task.  We substitute the **US Standard
/// Atmosphere 1976** model (`Drag::std_atm1976`), which is the highest-
/// fidelity drag model available in the Nyx 2.4.0 public API.
///
/// Harris-Priester is also absent from the Nyx 2.4.0 API; `std_atm1976`
/// is the best available fallback.
///
/// As a result, the accuracy residual vs Orekit (NRLMSISE-00) will be
/// **larger** than other tasks — km-level over a 1-day LEO propagation is
/// the expected and documented signal, not a bug.
/// See `docs/about/benchmarks.md` for the full discussion.
///
/// ## Spacecraft properties
///
/// The task params carry `mass` [kg], `drag_area` [m²], `cd` [-],
/// `srp_area` [m²], and `cr` [-] at the top level, shared across all cases.
/// These are passed directly into `Spacecraft::new`.
///
/// ## Unit conventions
///
/// Input elements (`elements` / `elements_deg`):
///   `[a_m, e, i_deg, raan_deg, argp_deg, M_deg]`
/// Output state: `[x, y, z, vx, vy, vz]` in **metres** and **m/s**.
pub fn numerical_rk4_grav80x80_full(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    // ── Spacecraft properties (shared across all cases) ──────────────────────
    let mass: f64 = serde_json::from_value(params["mass"].clone()).unwrap();
    let drag_area: f64 = serde_json::from_value(params["drag_area"].clone()).unwrap();
    let cd: f64 = serde_json::from_value(params["cd"].clone()).unwrap();
    let srp_area: f64 = serde_json::from_value(params["srp_area"].clone()).unwrap();
    let cr: f64 = serde_json::from_value(params["cr"].clone()).unwrap();

    let step_size: f64 = serde_json::from_value(params["step_size"].clone()).unwrap();
    let n_steps: usize = serde_json::from_value(params["n_steps"].clone()).unwrap();

    #[derive(serde::Deserialize)]
    struct Case {
        jd: f64,
        elements: Vec<f64>,
    }
    let cases: Option<Vec<Case>> = params
        .get("cases")
        .map(|v| serde_json::from_value(v.clone()).unwrap());

    // ── Almanac (PCK11 + de440s) ──────────────────────────────────────────────
    let almanac = Arc::new(crate::data::almanac().clone());

    // IAU Earth body-fixed frame: GravityField compute frame.
    let iau_earth = almanac
        .frame_info(IAU_EARTH_FRAME)
        .expect("IAU_EARTH_FRAME not found — PCK11 kernel must be loaded");

    // Earth J2000 inertial frame: shadow body for SRP.
    let earth_j2000 = almanac
        .frame_info(EARTH_J2000)
        .expect("EARTH_J2000 not found — PCK11 kernel must be loaded");

    // ── Force models (built once, cloned per propagator instance) ────────────

    // 80×80 EGM2008 spherical harmonics.
    let stor = load_egm2008_icgem(80, 80);
    let harmonics = GravityField::from_stor(iau_earth, stor);

    let mut orbital_dyn = OrbitalDynamics::from_model(harmonics);
    orbital_dyn
        .accel_models
        .push(PointMasses::new(vec![SUN, MOON]));

    // Atmospheric drag — US Standard Atmosphere 1976 (best available in Nyx 2.4.0;
    // substitutes for NRLMSISE-00 used by brahe/Orekit).
    let drag = Drag::std_atm1976(almanac.clone())
        .expect("std_atm1976: IAU Earth frame not found");

    // Solar radiation pressure with conical Earth-shadow eclipse model.
    let srp = SolarPressure::default_no_estimation(vec![earth_j2000], almanac.clone())
        .expect("SolarPressure: Sun J2000 frame not found");

    let sc_dyn = SpacecraftDynamics::from_models(orbital_dyn, vec![srp, drag]);
    let opts = IntegratorOptions::with_fixed_step_s(step_size);
    let prop = Propagator::new(sc_dyn, IntegratorMethod::RungeKutta4, opts);
    let total_dt = (n_steps as f64) * step_size;

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results: Vec<Vec<f64>> = Vec::new();

    for iter in 0..iterations {
        let start = std::time::Instant::now();
        let results: Vec<Vec<f64>>;

        if let Some(cases) = &cases {
            // Accuracy path: propagate each IC, report final state.
            results = cases
                .iter()
                .map(|case| {
                    let epoch = Epoch::from_jde_utc(case.jd);
                    let orbit = orbit_from_elements(&case.elements, epoch);
                    let sc = Spacecraft::new(orbit, mass, 0.0, srp_area, drag_area, cr, cd);
                    let mut instance = prop.with(sc, almanac.clone());
                    let final_sc = instance
                        .for_duration(Duration::from_seconds(total_dt))
                        .expect("numerical_rk4_grav80x80_full accuracy: propagation failed");
                    state_vec(&final_sc.orbit)
                })
                .collect();
        } else {
            // Perf path: single IC, full trajectory of n_steps states.
            let jd: f64 = serde_json::from_value(params["jd"].clone()).unwrap();
            let elements: Vec<f64> =
                serde_json::from_value(params["elements_deg"].clone()).unwrap();
            let epoch = Epoch::from_jde_utc(jd);
            let orbit = orbit_from_elements(&elements, epoch);
            let sc = Spacecraft::new(orbit, mass, 0.0, srp_area, drag_area, cr, cd);
            let mut instance = prop.with(sc, almanac.clone());

            let mut traj = Vec::with_capacity(n_steps);
            for _ in 0..n_steps {
                let stepped = instance
                    .for_duration(Duration::from_seconds(step_size))
                    .expect("numerical_rk4_grav80x80_full perf: step failed");
                traj.push(state_vec(&stepped.orbit));
            }
            results = traj;
        }

        all_times.push(start.elapsed().as_secs_f64());
        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}
