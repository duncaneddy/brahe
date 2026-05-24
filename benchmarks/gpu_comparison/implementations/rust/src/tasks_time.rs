//! Time conversion handlers for bench_gpu_rust.

use brahe::time::{Epoch, TimeSystem};
use rayon::prelude::*;
use serde::Deserialize;
use std::hint::black_box;
use std::time::Instant;

use crate::{BenchmarkInput, BenchmarkOutput, make_output};

#[derive(Deserialize)]
struct UtcMjdParams {
    mjd_utc: Vec<f64>,
}

pub fn utc_mjd_to_tt_mjd(input: &BenchmarkInput) -> Result<BenchmarkOutput, String> {
    let p: UtcMjdParams = serde_json::from_value(input.params.clone())
        .map_err(|e| format!("params parse: {e}"))?;
    assert_eq!(p.mjd_utc.len(), input.batch_size);
    let mjds = p.mjd_utc;

    for _ in 0..input.warmup_iterations {
        let _: Vec<f64> = mjds.par_iter().map(|m| {
            let e = Epoch::from_mjd(*m, TimeSystem::UTC);
            black_box(e.mjd_as_time_system(TimeSystem::TT))
        }).collect();
    }

    let mut times = Vec::with_capacity(input.iterations);
    for _ in 0..input.iterations {
        let start = Instant::now();
        let _: Vec<f64> = mjds.par_iter().map(|m| {
            let e = Epoch::from_mjd(*m, TimeSystem::UTC);
            black_box(e.mjd_as_time_system(TimeSystem::TT))
        }).collect();
        times.push(start.elapsed().as_secs_f64().max(1e-12));
    }
    Ok(make_output(input, times))
}
