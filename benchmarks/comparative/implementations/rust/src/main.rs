//! Comparative benchmark CLI for Brahe (Rust).
//!
//! Reads JSON task input from stdin, runs the benchmark, outputs JSON to stdout.

// Use mimalloc to match brahe-py's allocator choice. Without this the Rust
// bench would use the macOS system allocator while the Python bench uses
// mimalloc (via brahe-py), so per-language comparisons would be biased by
// the allocator difference rather than reflecting actual library work.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod access;
mod attitude;
mod coordinates;
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
}

fn find_orekit_eop_file() -> Option<std::path::PathBuf> {
    let orekit_data = std::env::var("OREKIT_DATA").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.orekit/orekit-data")
    });
    let eop_path = std::path::PathBuf::from(orekit_data)
        .join("Earth-Orientation-Parameters")
        .join("IAU-2000")
        .join("finals2000A.all");
    if eop_path.exists() {
        Some(eop_path)
    } else {
        None
    }
}

fn find_orekit_sw_file() -> Option<std::path::PathBuf> {
    let orekit_data = std::env::var("OREKIT_DATA").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.orekit/orekit-data")
    });
    let sw_path = std::path::PathBuf::from(orekit_data)
        .join("CSSI-Space-Weather-Data")
        .join("SpaceWeather-All-v1.2.txt");
    if sw_path.exists() {
        Some(sw_path)
    } else {
        None
    }
}

fn main() {
    // Initialize EOP provider — use OreKit's real IERS data if available
    if let Some(eop_path) = find_orekit_eop_file() {
        let provider = brahe::eop::FileEOPProvider::from_standard_file(
            &eop_path,
            true,
            brahe::eop::EOPExtrapolation::Hold,
        )
        .expect("Failed to load OreKit EOP file");
        brahe::eop::set_global_eop_provider(provider);
    } else {
        // Fallback to zero values
        let eop = brahe::eop::StaticEOPProvider::from_zero();
        brahe::eop::set_global_eop_provider(eop);
    }

    // Initialize space weather from OreKit's CSSI-Space-Weather-Data file
    // when available so brahe's NRLMSISE-00 drag inputs match Orekit's
    // ``CssiSpaceWeatherData`` byte-for-byte. Falls back to brahe's
    // packaged file when the OreKit table isn't on disk.
    let sw_loaded = find_orekit_sw_file().and_then(|p| {
        brahe::space_weather::FileSpaceWeatherProvider::from_file(
            &p,
            brahe::space_weather::SpaceWeatherExtrapolation::Hold,
        )
        .ok()
    });
    if let Some(provider) = sw_loaded {
        brahe::space_weather::set_global_space_weather_provider(provider);
    } else if let Ok(provider) =
        brahe::space_weather::FileSpaceWeatherProvider::from_default_file()
    {
        brahe::space_weather::set_global_space_weather_provider(provider);
    }

    // Read JSON from stdin
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .expect("Failed to read stdin");

    let bench_input: BenchmarkInput =
        serde_json::from_str(&input).expect("Failed to parse input JSON");

    let (times, results) = match bench_input.task.as_str() {
        "attitude.quaternion_to_rotation_matrix" => {
            attitude::quaternion_to_rotation_matrix(&bench_input.params, bench_input.iterations)
        }
        "attitude.rotation_matrix_to_quaternion" => {
            attitude::rotation_matrix_to_quaternion(&bench_input.params, bench_input.iterations)
        }
        "attitude.quaternion_to_euler_angle" => {
            attitude::quaternion_to_euler_angle(&bench_input.params, bench_input.iterations)
        }
        "attitude.euler_angle_to_quaternion" => {
            attitude::euler_angle_to_quaternion(&bench_input.params, bench_input.iterations)
        }
        "frames.state_eci_to_ecef" => {
            frames::eci_to_ecef(&bench_input.params, bench_input.iterations)
        }
        "frames.state_ecef_to_eci" => {
            frames::ecef_to_eci(&bench_input.params, bench_input.iterations)
        }
        "coordinates.geodetic_to_ecef" => {
            coordinates::geodetic_to_ecef(&bench_input.params, bench_input.iterations)
        }
        "coordinates.ecef_to_geodetic" => {
            coordinates::ecef_to_geodetic(&bench_input.params, bench_input.iterations)
        }
        "coordinates.geocentric_to_ecef" => {
            coordinates::geocentric_to_ecef(&bench_input.params, bench_input.iterations)
        }
        "coordinates.ecef_to_geocentric" => {
            coordinates::ecef_to_geocentric(&bench_input.params, bench_input.iterations)
        }
        "coordinates.ecef_to_azel" => {
            coordinates::ecef_to_azel(&bench_input.params, bench_input.iterations)
        }
        "orbits.keplerian_to_cartesian" => {
            orbits::keplerian_to_cartesian(&bench_input.params, bench_input.iterations)
        }
        "orbits.cartesian_to_keplerian" => {
            orbits::cartesian_to_keplerian(&bench_input.params, bench_input.iterations)
        }
        "time.epoch_creation" => {
            time_bench::epoch_creation(&bench_input.params, bench_input.iterations)
        }
        "time.utc_to_tai" => time_bench::utc_to_tai(&bench_input.params, bench_input.iterations),
        "time.utc_to_tt" => time_bench::utc_to_tt(&bench_input.params, bench_input.iterations),
        "time.utc_to_gps" => time_bench::utc_to_gps(&bench_input.params, bench_input.iterations),
        "time.utc_to_ut1" => time_bench::utc_to_ut1(&bench_input.params, bench_input.iterations),
        "propagation.keplerian_single" => {
            propagation::keplerian_single(&bench_input.params, bench_input.iterations)
        }
        "propagation.keplerian_trajectory" => {
            propagation::keplerian_trajectory(&bench_input.params, bench_input.iterations)
        }
        "propagation.sgp4_single" => {
            propagation::sgp4_single(&bench_input.params, bench_input.iterations)
        }
        "propagation.sgp4_trajectory" => {
            propagation::sgp4_trajectory(&bench_input.params, bench_input.iterations)
        }
        "propagation.numerical_twobody" => {
            propagation::numerical_twobody(&bench_input.params, bench_input.iterations)
        }
        "propagation.numerical_rk4_grav5x5" => {
            propagation::numerical_rk4_grav5x5(&bench_input.params, bench_input.iterations)
        }
        "propagation.numerical_rk4_grav20x20_sun_moon" => {
            propagation::numerical_rk4_grav20x20_sun_moon(
                &bench_input.params,
                bench_input.iterations,
            )
        }
        "propagation.numerical_rk4_grav80x80_full" => {
            propagation::numerical_rk4_grav80x80_full(&bench_input.params, bench_input.iterations)
        }
        "force_model.accel_point_mass_gravity" => {
            force_model::accel_point_mass(&bench_input.params, bench_input.iterations)
        }
        "force_model.accel_spherical_harmonics_20" => {
            force_model::accel_spherical_harmonics_20(
                &bench_input.params,
                bench_input.iterations,
            )
        }
        "force_model.accel_spherical_harmonics_80" => {
            force_model::accel_spherical_harmonics_80(
                &bench_input.params,
                bench_input.iterations,
            )
        }
        "force_model.accel_third_body_sun" => {
            force_model::accel_third_body_sun(&bench_input.params, bench_input.iterations)
        }
        "force_model.accel_third_body_moon" => {
            force_model::accel_third_body_moon(&bench_input.params, bench_input.iterations)
        }
        "access.sgp4_access" => {
            access::sgp4_access(&bench_input.params, bench_input.iterations)
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
