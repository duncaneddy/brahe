use brahe::GM_EARTH;
use brahe::frames::rotation_eci_to_ecef;
use brahe::orbit_dynamics::{
    GravityModel, GravityModelType, ParallelMode, accel_gravity_spherical_harmonics,
    accel_point_mass_gravity, accel_third_body_moon_spice, accel_third_body_sun_spice,
};
use brahe::propagators::EphemerisSource;
use brahe::time::{Epoch, TimeSystem};
use nalgebra::Vector3;
use std::time::Instant;

/// One IC for the accuracy sweep path. Mirrors the JSON shape produced by
/// ``_random_leo_state_eci`` in force_model_tasks.py.
#[derive(serde::Deserialize)]
struct Case {
    jd: f64,
    state_eci: Vec<f64>,
}

fn case_to_state(case: &Case) -> (Epoch, Vector3<f64>) {
    let epc = Epoch::from_jd(case.jd, TimeSystem::UTC);
    let r = Vector3::new(case.state_eci[0], case.state_eci[1], case.state_eci[2]);
    (epc, r)
}

/// Collect the IC sweep into `(epoch, r)` pairs, or `None` for the single-IC
/// perf path. Lets each task pick between two execution shapes without
/// duplicating the JSON-shape dispatch.
fn maybe_cases(params: &serde_json::Value) -> Option<Vec<(Epoch, Vector3<f64>)>> {
    let cases_val = params.get("cases")?;
    let cases: Vec<Case> = serde_json::from_value(cases_val.clone()).ok()?;
    Some(cases.iter().map(case_to_state).collect())
}

fn single_state(params: &serde_json::Value) -> (Epoch, Vector3<f64>, usize) {
    let jd: f64 = serde_json::from_value(params["jd"].clone()).unwrap();
    let state_eci: Vec<f64> = serde_json::from_value(params["state_eci"].clone()).unwrap();
    let n_samples: usize = serde_json::from_value(params["n_samples"].clone()).unwrap();
    let epc = Epoch::from_jd(jd, TimeSystem::UTC);
    let r = Vector3::new(state_eci[0], state_eci[1], state_eci[2]);
    (epc, r, n_samples)
}

/// Run a force-model evaluator either as the single-IC perf loop (inner loop
/// of n_samples for timing) or as the multi-IC accuracy sweep (one call per
/// case). Returns `(times_per_iter, result_vectors)`.
fn run_force_model<F>(
    params: &serde_json::Value,
    iterations: usize,
    mut eval: F,
) -> (Vec<f64>, serde_json::Value)
where
    F: FnMut(Epoch, Vector3<f64>) -> Vector3<f64>,
{
    let mut all_times = Vec::with_capacity(iterations);
    let mut first_result: Vec<Vec<f64>> = Vec::new();

    if let Some(cases) = maybe_cases(params) {
        for iter in 0..iterations {
            let start = Instant::now();
            let mut out = Vec::with_capacity(cases.len());
            for &(epc, r) in &cases {
                let a = eval(epc, r);
                out.push(vec![a[0], a[1], a[2]]);
            }
            all_times.push(start.elapsed().as_secs_f64());
            if iter == 0 {
                first_result = out;
            }
        }
    } else {
        let (epc, r, n_samples) = single_state(params);
        for iter in 0..iterations {
            let start = Instant::now();
            let mut a = Vector3::zeros();
            for _ in 0..n_samples {
                a = eval(epc, r);
            }
            all_times.push(start.elapsed().as_secs_f64());
            if iter == 0 {
                first_result = vec![vec![a[0], a[1], a[2]]];
            }
        }
    }

    (all_times, serde_json::to_value(first_result).unwrap())
}

pub fn accel_point_mass(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let r_cb = Vector3::zeros();
    run_force_model(params, iterations, |_epc, r| {
        accel_point_mass_gravity(r, r_cb, GM_EARTH)
    })
}

fn accel_spherical_harmonics_run(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let degree: usize = serde_json::from_value(params["degree"].clone()).unwrap();
    let order: usize = serde_json::from_value(params["order"].clone()).unwrap();
    let gravity_model = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();

    run_force_model(params, iterations, move |epc, r| {
        let rot = rotation_eci_to_ecef(epc);
        accel_gravity_spherical_harmonics(r, rot, &gravity_model, degree, order, ParallelMode::Auto)
    })
}

pub fn accel_spherical_harmonics_20(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    accel_spherical_harmonics_run(params, iterations)
}

pub fn accel_spherical_harmonics_80(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    accel_spherical_harmonics_run(params, iterations)
}

pub fn accel_third_body_sun(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    run_force_model(params, iterations, |epc, r| {
        accel_third_body_sun_spice(epc, r, EphemerisSource::DE440s)
    })
}

pub fn accel_third_body_moon(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    run_force_model(params, iterations, |epc, r| {
        accel_third_body_moon_spice(epc, r, EphemerisSource::DE440s)
    })
}
