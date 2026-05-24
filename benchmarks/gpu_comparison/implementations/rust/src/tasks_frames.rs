//! Frame transformation handlers for bench_gpu_rust.

use brahe::frames::state_gcrf_to_itrf;
use brahe::math::SVector6;
use brahe::time::{Epoch, TimeSystem};
use rayon::prelude::*;
use serde::Deserialize;
use std::hint::black_box;
use std::time::Instant;

use crate::{BenchmarkInput, BenchmarkOutput, make_output};

#[derive(Deserialize)]
struct GcrfItrfParams {
    mjd_utc: Vec<f64>,
    state_gcrf: Vec<[f64; 6]>,
}

pub fn gcrf_to_itrf(input: &BenchmarkInput) -> Result<BenchmarkOutput, String> {
    let p: GcrfItrfParams = serde_json::from_value(input.params.clone())
        .map_err(|e| format!("params parse: {e}"))?;
    assert_eq!(p.mjd_utc.len(), input.batch_size);
    assert_eq!(p.state_gcrf.len(), input.batch_size);

    let pairs: Vec<(f64, [f64; 6])> = p
        .mjd_utc
        .into_iter()
        .zip(p.state_gcrf.into_iter())
        .collect();

    for _ in 0..input.warmup_iterations {
        let _: Vec<SVector6> = pairs.par_iter().map(|(mjd, s)| {
            let e = Epoch::from_mjd(*mjd, TimeSystem::UTC);
            let v = SVector6::new(s[0], s[1], s[2], s[3], s[4], s[5]);
            black_box(state_gcrf_to_itrf(e, v))
        }).collect();
    }

    let mut times = Vec::with_capacity(input.iterations);
    for _ in 0..input.iterations {
        let start = Instant::now();
        let _: Vec<SVector6> = pairs.par_iter().map(|(mjd, s)| {
            let e = Epoch::from_mjd(*mjd, TimeSystem::UTC);
            let v = SVector6::new(s[0], s[1], s[2], s[3], s[4], s[5]);
            black_box(state_gcrf_to_itrf(e, v))
        }).collect();
        times.push(start.elapsed().as_secs_f64().max(1e-12));
    }
    Ok(make_output(input, times))
}
