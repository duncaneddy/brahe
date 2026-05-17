use brahe::frames::rotation_eci_to_ecef;
use brahe::orbit_dynamics::{
    accel_gravity_spherical_harmonics, accel_point_mass_gravity, accel_third_body_moon_de,
    accel_third_body_sun_de, GravityModel, GravityModelType,
};
use brahe::propagators::EphemerisSource;
use brahe::time::{Epoch, TimeSystem};
use brahe::GM_EARTH;
use nalgebra::Vector3;
use std::time::Instant;

#[derive(serde::Deserialize)]
struct AccelParams {
    jd: f64,
    state_eci: Vec<f64>,
    n_samples: usize,
    #[serde(default)]
    degree: usize,
    #[serde(default)]
    order: usize,
}

fn parse(params: &serde_json::Value) -> (Epoch, Vector3<f64>, AccelParams) {
    let p: AccelParams = serde_json::from_value(params.clone()).unwrap();
    let epc = Epoch::from_jd(p.jd, TimeSystem::UTC);
    let r = Vector3::new(p.state_eci[0], p.state_eci[1], p.state_eci[2]);
    (epc, r, p)
}

pub fn accel_point_mass(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let (_epc, r, p) = parse(params);
    let r_cb = Vector3::zeros();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_result = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut a = Vector3::zeros();
        for _ in 0..p.n_samples {
            a = accel_point_mass_gravity(r, r_cb, GM_EARTH);
        }
        all_times.push(start.elapsed().as_secs_f64());
        if iter == 0 {
            first_result = vec![vec![a[0], a[1], a[2]]];
        }
    }

    (all_times, serde_json::to_value(first_result).unwrap())
}

fn accel_spherical_harmonics_run(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let (epc, r, p) = parse(params);
    let gravity_model = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_result = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut a = Vector3::zeros();
        for _ in 0..p.n_samples {
            let rot = rotation_eci_to_ecef(epc);
            a = accel_gravity_spherical_harmonics(r, rot, &gravity_model, p.degree, p.order);
        }
        all_times.push(start.elapsed().as_secs_f64());
        if iter == 0 {
            first_result = vec![vec![a[0], a[1], a[2]]];
        }
    }

    (all_times, serde_json::to_value(first_result).unwrap())
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
    let (epc, r, p) = parse(params);

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_result = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut a = Vector3::zeros();
        for _ in 0..p.n_samples {
            a = accel_third_body_sun_de(epc, r, EphemerisSource::DE440s);
        }
        all_times.push(start.elapsed().as_secs_f64());
        if iter == 0 {
            first_result = vec![vec![a[0], a[1], a[2]]];
        }
    }

    (all_times, serde_json::to_value(first_result).unwrap())
}

pub fn accel_third_body_moon(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let (epc, r, p) = parse(params);

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_result = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut a = Vector3::zeros();
        for _ in 0..p.n_samples {
            a = accel_third_body_moon_de(epc, r, EphemerisSource::DE440s);
        }
        all_times.push(start.elapsed().as_secs_f64());
        if iter == 0 {
            first_result = vec![vec![a[0], a[1], a[2]]];
        }
    }

    (all_times, serde_json::to_value(first_result).unwrap())
}
