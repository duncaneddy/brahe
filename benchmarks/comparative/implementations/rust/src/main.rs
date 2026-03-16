//! Comparative benchmark CLI for Brahe (Rust).
//!
//! Reads JSON task input from stdin, runs the benchmark, outputs JSON to stdout.

mod coordinates;
mod orbits;

use serde::{Deserialize, Serialize};
use std::io::{self, Read};

#[derive(Debug, Deserialize)]
struct BenchmarkInput {
    task: String,
    iterations: usize,
    params: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct BenchmarkOutput {
    task: String,
    iterations: usize,
    times_seconds: Vec<f64>,
    results: serde_json::Value,
    metadata: Metadata,
}

#[derive(Debug, Serialize)]
struct Metadata {
    library: String,
    version: String,
    language: String,
}

fn main() {
    // Initialize EOP provider (zero values — sufficient for coordinate conversions)
    let eop = brahe::eop::StaticEOPProvider::from_zero();
    brahe::eop::set_global_eop_provider(eop);

    // Read JSON from stdin
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .expect("Failed to read stdin");

    let bench_input: BenchmarkInput =
        serde_json::from_str(&input).expect("Failed to parse input JSON");

    let (times, results) = match bench_input.task.as_str() {
        "coordinates.geodetic_to_ecef" => {
            coordinates::geodetic_to_ecef(&bench_input.params, bench_input.iterations)
        }
        "coordinates.ecef_to_geodetic" => {
            coordinates::ecef_to_geodetic(&bench_input.params, bench_input.iterations)
        }
        "orbits.keplerian_to_cartesian" => {
            orbits::keplerian_to_cartesian(&bench_input.params, bench_input.iterations)
        }
        "orbits.cartesian_to_keplerian" => {
            orbits::cartesian_to_keplerian(&bench_input.params, bench_input.iterations)
        }
        _ => {
            eprintln!("Unknown task: {}", bench_input.task);
            std::process::exit(1);
        }
    };

    let output = BenchmarkOutput {
        task: bench_input.task,
        iterations: bench_input.iterations,
        times_seconds: times,
        results,
        metadata: Metadata {
            library: "brahe".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            language: "rust".to_string(),
        },
    };

    let json_output = serde_json::to_string(&output).expect("Failed to serialize output");
    println!("{json_output}");
}
