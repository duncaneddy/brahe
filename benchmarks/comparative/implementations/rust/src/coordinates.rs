use brahe::AngleFormat;
use brahe::coordinates::{
    EllipsoidalConversionType, position_ecef_to_geocentric, position_ecef_to_geodetic,
    position_enz_to_azel, position_geocentric_to_ecef, position_geodetic_to_ecef,
    relative_position_ecef_to_enz,
};
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

pub fn geocentric_to_ecef(
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
            let geoc = Vector3::new(pt[0], pt[1], pt[2]);
            let ecef = position_geocentric_to_ecef(geoc, AngleFormat::Degrees).unwrap();
            results.push(vec![ecef[0], ecef[1], ecef[2]]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn ecef_to_geocentric(
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
            let geoc = position_ecef_to_geocentric(ecef, AngleFormat::Degrees);
            results.push(vec![geoc[0], geoc[1], geoc[2]]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn ecef_to_azel(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    #[derive(serde::Deserialize)]
    struct Pair {
        station_ecef: Vec<f64>,
        satellite_ecef: Vec<f64>,
        #[allow(dead_code)]
        station_geodetic: Vec<f64>,
    }
    let pairs: Vec<Pair> = serde_json::from_value(params["pairs"].clone()).unwrap();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(pairs.len());

        for pair in &pairs {
            let station = Vector3::new(
                pair.station_ecef[0],
                pair.station_ecef[1],
                pair.station_ecef[2],
            );
            let satellite = Vector3::new(
                pair.satellite_ecef[0],
                pair.satellite_ecef[1],
                pair.satellite_ecef[2],
            );
            let enz = relative_position_ecef_to_enz(
                station,
                satellite,
                EllipsoidalConversionType::Geodetic,
            );
            let azel = position_enz_to_azel(enz, AngleFormat::Degrees);
            results.push(vec![azel[0], azel[1], azel[2]]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}
