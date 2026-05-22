//! Comparative benchmark CLI for Nyx + ANISE (Rust).
//!
//! Reads JSON task input from stdin, runs the benchmark, outputs JSON to stdout.
//! Mirrors the wire format of `implementations/rust/src/main.rs`.

mod attitude;
mod coordinates;
mod data;
mod force_model;
mod frames;
mod orbits;
mod propagation;
mod time_bench;

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
    anise_version: String,
}

fn main() {
    // Read JSON from stdin
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .expect("Failed to read stdin");

    let bench_input: BenchmarkInput =
        serde_json::from_str(&input).expect("Failed to parse input JSON");

    let (times, results): (Vec<f64>, serde_json::Value) = match bench_input.task.as_str() {
        "orbits.keplerian_to_cartesian" => orbits::keplerian_to_cartesian(&bench_input.params, bench_input.iterations),
        "orbits.cartesian_to_keplerian" => orbits::cartesian_to_keplerian(&bench_input.params, bench_input.iterations),
        "time.utc_to_tai"      => time_bench::utc_to_tai(&bench_input.params, bench_input.iterations),
        "time.epoch_creation"  => time_bench::epoch_creation(&bench_input.params, bench_input.iterations),
        "time.utc_to_tt"       => time_bench::utc_to_tt(&bench_input.params, bench_input.iterations),
        "time.utc_to_gps"      => time_bench::utc_to_gps(&bench_input.params, bench_input.iterations),
        "time.utc_to_ut1"      => time_bench::utc_to_ut1(&bench_input.params, bench_input.iterations),
        "attitude.quaternion_to_rotation_matrix"  => attitude::quaternion_to_rotation_matrix(&bench_input.params, bench_input.iterations),
        "attitude.rotation_matrix_to_quaternion"  => attitude::rotation_matrix_to_quaternion(&bench_input.params, bench_input.iterations),
        "attitude.quaternion_to_euler_angle"      => attitude::quaternion_to_euler_angle(&bench_input.params, bench_input.iterations),
        "attitude.euler_angle_to_quaternion"      => attitude::euler_angle_to_quaternion(&bench_input.params, bench_input.iterations),
        "coordinates.geodetic_to_ecef"            => coordinates::geodetic_to_ecef(&bench_input.params, bench_input.iterations),
        "coordinates.ecef_to_geodetic"            => coordinates::ecef_to_geodetic(&bench_input.params, bench_input.iterations),
        "coordinates.ecef_to_azel"                => coordinates::ecef_to_azel(&bench_input.params, bench_input.iterations),
        "frames.state_eci_to_ecef"                => frames::eci_to_ecef(&bench_input.params, bench_input.iterations),
        "frames.state_ecef_to_eci"                => frames::ecef_to_eci(&bench_input.params, bench_input.iterations),
        "propagation.keplerian_single"            => propagation::keplerian_single(&bench_input.params, bench_input.iterations),
        "propagation.keplerian_trajectory"        => propagation::keplerian_trajectory(&bench_input.params, bench_input.iterations),
        "propagation.sgp4_single"                 => propagation::sgp4_single(&bench_input.params, bench_input.iterations),
        "propagation.sgp4_trajectory"             => propagation::sgp4_trajectory(&bench_input.params, bench_input.iterations),
        "propagation.numerical_twobody"           => propagation::numerical_twobody(&bench_input.params, bench_input.iterations),
        "propagation.numerical_rk4_grav5x5"      => propagation::numerical_rk4_grav5x5(&bench_input.params, bench_input.iterations),
        "propagation.numerical_rk4_grav20x20_sun_moon" => propagation::numerical_rk4_grav20x20_sun_moon(&bench_input.params, bench_input.iterations),
        "propagation.numerical_rk4_grav80x80_full" => propagation::numerical_rk4_grav80x80_full(&bench_input.params, bench_input.iterations),
        "force_model.accel_spherical_harmonics_20"  => force_model::accel_spherical_harmonics_20(&bench_input.params, bench_input.iterations),
        "force_model.accel_spherical_harmonics_80"  => force_model::accel_spherical_harmonics_80(&bench_input.params, bench_input.iterations),
        "force_model.accel_third_body_sun"          => force_model::accel_third_body_sun(&bench_input.params, bench_input.iterations),
        "force_model.accel_third_body_moon"         => force_model::accel_third_body_moon(&bench_input.params, bench_input.iterations),
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
            library: "nyx".to_string(),
            version: env!("NYX_SPACE_VERSION").to_string(),
            language: "nyx".to_string(),
            anise_version: env!("ANISE_VERSION").to_string(),
        },
    };

    let json_output = serde_json::to_string(&output).expect("Failed to serialize output");
    println!("{json_output}");
}
