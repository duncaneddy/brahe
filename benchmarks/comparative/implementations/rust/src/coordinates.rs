use brahe::coordinates::{position_ecef_to_geodetic, position_geodetic_to_ecef};
use brahe::AngleFormat;
use nalgebra::Vector3;
use std::time::Instant;

pub fn geodetic_to_ecef(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let points: Vec<Vec<f64>> = serde_json::from_value(params["points"].clone()).unwrap();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(points.len());

        for pt in &points {
            let geod = Vector3::new(pt[0], pt[1], pt[2]);
            let ecef = position_geodetic_to_ecef(geod, AngleFormat::Degrees).unwrap();
            results.push(vec![ecef[0], ecef[1], ecef[2]]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn ecef_to_geodetic(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let points: Vec<Vec<f64>> = serde_json::from_value(params["points"].clone()).unwrap();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(points.len());

        for pt in &points {
            let ecef = Vector3::new(pt[0], pt[1], pt[2]);
            let geod = position_ecef_to_geodetic(ecef, AngleFormat::Degrees);
            results.push(vec![geod[0], geod[1], geod[2]]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}
