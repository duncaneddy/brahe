//! bench_gpu_rust — Rust subprocess for the GPU-comparison benchmark suite.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use serde::{Deserialize, Serialize};
use std::io::{self, Read};
use std::process::ExitCode;
use std::time::Instant;

mod tasks_coordinates;
mod tasks_force_model;
mod tasks_frames;
mod tasks_propagation;
mod tasks_time;

#[derive(Debug, Deserialize)]
pub struct BenchmarkInput {
    pub task: String,
    pub batch_size: usize,
    pub iterations: usize,
    pub warmup_iterations: usize,
    pub seed: u64,
    pub params: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct BenchmarkOutput {
    pub task: String,
    pub batch_size: usize,
    pub iterations: usize,
    pub times_seconds: Vec<f64>,
    pub metadata: serde_json::Value,
}

pub fn make_output(input: &BenchmarkInput, times: Vec<f64>) -> BenchmarkOutput {
    BenchmarkOutput {
        task: input.task.clone(),
        batch_size: input.batch_size,
        iterations: input.iterations,
        times_seconds: times,
        metadata: serde_json::json!({
            "brahe_version": env!("CARGO_PKG_VERSION"),
            "rayon_threads": rayon::current_num_threads(),
        }),
    }
}

fn init_eop() {
    use brahe::eop::{EOPExtrapolation, FileEOPProvider, set_global_eop_provider};
    if let Ok(path) = std::env::var("BRAHE_EOP_FILE") {
        let p = std::path::Path::new(&path);
        if let Ok(eop) = FileEOPProvider::from_file(p, true, EOPExtrapolation::Hold) {
            set_global_eop_provider(eop);
        }
    }
}

fn main() -> ExitCode {
    init_eop();

    let mut stdin = String::new();
    if let Err(e) = io::stdin().read_to_string(&mut stdin) {
        eprintln!("failed to read stdin: {e}");
        return ExitCode::from(1);
    }

    let input: BenchmarkInput = match serde_json::from_str(&stdin) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("failed to parse input JSON: {e}");
            return ExitCode::from(1);
        }
    };

    match run_task(&input) {
        Ok(output) => {
            let json = serde_json::to_string(&output).expect("serialise output");
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::from(2)
        }
    }
}

fn run_task(input: &BenchmarkInput) -> Result<BenchmarkOutput, String> {
    match input.task.as_str() {
        "ping.identity" => ping_identity(input),
        "coordinates.geodetic_to_ecef" => tasks_coordinates::geodetic_to_ecef(input),
        "coordinates.keplerian_to_cartesian" => tasks_coordinates::keplerian_to_cartesian(input),
        "coordinates.enz_to_azel" => tasks_coordinates::enz_to_azel(input),
        "propagation.sgp4_iss_sweep" => tasks_propagation::sgp4_iss_sweep(input),
        "propagation.numerical_twobody_j2" => tasks_propagation::numerical_twobody_j2(input),
        "time.utc_mjd_to_tt_mjd" => tasks_time::utc_mjd_to_tt_mjd(input),
        "frames.gcrf_to_itrf" => tasks_frames::gcrf_to_itrf(input),
        "force_model.grav_5x5" => tasks_force_model::grav_5x5(input),
        other => Err(format!("unknown task: {other}")),
    }
}

fn ping_identity(input: &BenchmarkInput) -> Result<BenchmarkOutput, String> {
    let _ = input.params.get("n");
    for _ in 0..input.warmup_iterations {
        std::hint::black_box(input.batch_size * 2);
    }
    let mut times = Vec::with_capacity(input.iterations);
    for _ in 0..input.iterations {
        let start = Instant::now();
        let _ = std::hint::black_box(input.batch_size * 2);
        times.push(start.elapsed().as_secs_f64().max(1e-12));
    }
    Ok(make_output(input, times))
}
