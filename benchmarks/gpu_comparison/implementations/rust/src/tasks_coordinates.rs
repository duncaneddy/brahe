//! Coordinate transformation handlers for bench_gpu_rust.

use brahe::coordinates::{
    position_ecef_to_geodetic, position_enz_to_azel, position_geodetic_to_ecef,
    state_eci_to_koe, state_koe_to_eci,
};
use brahe::AngleFormat;
use brahe::math::SVector6;
use nalgebra::Vector3;
use rayon::prelude::*;
use serde::Deserialize;
use std::hint::black_box;
use std::time::Instant;

use crate::{BenchmarkInput, BenchmarkOutput, make_output};

#[derive(Deserialize)]
struct Points3 {
    points: Vec<[f64; 3]>,
}

#[derive(Deserialize)]
struct Elements6 {
    elements: Vec<[f64; 6]>,
}

#[derive(Deserialize)]
struct Vectors3 {
    vectors: Vec<[f64; 3]>,
}

pub fn geodetic_to_ecef(input: &BenchmarkInput) -> Result<BenchmarkOutput, String> {
    let p: Points3 = serde_json::from_value(input.params.clone())
        .map_err(|e| format!("params parse: {e}"))?;
    assert_eq!(p.points.len(), input.batch_size);
    let pts = p.points;

    for _ in 0..input.warmup_iterations {
        let _: Vec<Vector3<f64>> = pts.par_iter().map(|pt| {
            let g = Vector3::new(pt[0], pt[1], pt[2]);
            black_box(position_geodetic_to_ecef(g, AngleFormat::Degrees).unwrap())
        }).collect();
    }

    let mut times = Vec::with_capacity(input.iterations);
    for _ in 0..input.iterations {
        let start = Instant::now();
        let _: Vec<Vector3<f64>> = pts.par_iter().map(|pt| {
            let g = Vector3::new(pt[0], pt[1], pt[2]);
            black_box(position_geodetic_to_ecef(g, AngleFormat::Degrees).unwrap())
        }).collect();
        times.push(start.elapsed().as_secs_f64().max(1e-12));
    }
    Ok(make_output(input, times))
}

pub fn keplerian_to_cartesian(input: &BenchmarkInput) -> Result<BenchmarkOutput, String> {
    let p: Elements6 = serde_json::from_value(input.params.clone())
        .map_err(|e| format!("params parse: {e}"))?;
    assert_eq!(p.elements.len(), input.batch_size);
    let oes = p.elements;

    for _ in 0..input.warmup_iterations {
        let _: Vec<SVector6> = oes.par_iter().map(|e| {
            let v = SVector6::new(e[0], e[1], e[2], e[3], e[4], e[5]);
            black_box(state_koe_to_eci(v, AngleFormat::Degrees))
        }).collect();
    }

    let mut times = Vec::with_capacity(input.iterations);
    for _ in 0..input.iterations {
        let start = Instant::now();
        let _: Vec<SVector6> = oes.par_iter().map(|e| {
            let v = SVector6::new(e[0], e[1], e[2], e[3], e[4], e[5]);
            black_box(state_koe_to_eci(v, AngleFormat::Degrees))
        }).collect();
        times.push(start.elapsed().as_secs_f64().max(1e-12));
    }
    Ok(make_output(input, times))
}

pub fn enz_to_azel(input: &BenchmarkInput) -> Result<BenchmarkOutput, String> {
    let p: Vectors3 = serde_json::from_value(input.params.clone())
        .map_err(|e| format!("params parse: {e}"))?;
    assert_eq!(p.vectors.len(), input.batch_size);
    let vecs = p.vectors;

    for _ in 0..input.warmup_iterations {
        let _: Vec<Vector3<f64>> = vecs.par_iter().map(|v| {
            let enz = Vector3::new(v[0], v[1], v[2]);
            black_box(position_enz_to_azel(enz, AngleFormat::Degrees))
        }).collect();
    }

    let mut times = Vec::with_capacity(input.iterations);
    for _ in 0..input.iterations {
        let start = Instant::now();
        let _: Vec<Vector3<f64>> = vecs.par_iter().map(|v| {
            let enz = Vector3::new(v[0], v[1], v[2]);
            black_box(position_enz_to_azel(enz, AngleFormat::Degrees))
        }).collect();
        times.push(start.elapsed().as_secs_f64().max(1e-12));
    }
    Ok(make_output(input, times))
}
