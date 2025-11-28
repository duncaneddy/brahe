//! ```cargo
//! [dependencies]
//! brahe = { path = ".." }
//! serde = { version = "1.0", features = ["derive"] }
//! serde_json = "1.0"
//! chrono = { version = "0.4", features = ["serde"] }
//!
//! [profile.release]
//! opt-level = 3
//! lto = true
//! codegen-units = 1
//! ```
//!
//! Rust benchmark script for access computation.
//! Reads JSON input from stdin, computes access windows, outputs JSON to stdout.
//!
//! Usage:
//!   cd scripts && echo '{"locations": [...], ...}' | rust-script benchmark_access_rust.rs

use brahe::{
    access::{location_accesses, ElevationConstraint, PointLocation},
    eop::StaticEOPProvider,
    propagators::SGPPropagator,
    time::Epoch,
    utils::Identifiable,
};
use chrono::{DateTime, Datelike, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct InputLocation {
    lat: f64,
    lon: f64,
    alt: f64,
    name: String,
}

#[derive(Debug, Deserialize)]
struct BenchmarkInput {
    locations: Vec<InputLocation>,
    tle_line1: String,
    tle_line2: String,
    window_open: String,
    window_close: String,
    min_elevation_deg: f64,
    iterations: usize,
}

#[derive(Debug, Clone, Serialize)]
struct AccessOutput {
    location_name: String,
    aos: String,
    los: String,
    duration_s: f64,
}

#[derive(Debug, Serialize)]
struct TimingOutput {
    mean_seconds: f64,
    std_seconds: f64,
    iterations: usize,
    num_windows: usize,
}

#[derive(Debug, Serialize)]
struct LocationResult {
    location_name: String,
    lat: f64,
    lon: f64,
    accesses: Vec<AccessOutput>,
    timing: TimingOutput,
}

#[derive(Debug, Serialize)]
struct BenchmarkOutput {
    results: Vec<LocationResult>,
}

fn parse_datetime(s: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(s)
        .expect("Failed to parse datetime")
        .with_timezone(&Utc)
}

fn epoch_to_iso(epoch: &Epoch) -> String {
    let (year, month, day, hour, minute, second, nanosecond) = epoch.to_datetime();
    let dt = chrono::NaiveDate::from_ymd_opt(year as i32, month as u32, day as u32)
        .unwrap()
        .and_hms_nano_opt(
            hour as u32,
            minute as u32,
            second as u32,
            nanosecond as u32,
        )
        .unwrap();
    DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc)
        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

fn compute_accesses_for_location(
    location: &InputLocation,
    propagator: &SGPPropagator,
    epoch_start: Epoch,
    epoch_end: Epoch,
    constraint: &ElevationConstraint,
) -> Vec<AccessOutput> {
    let point = PointLocation::new(location.lon, location.lat, location.alt)
        .with_name(&location.name);

    let windows = location_accesses(&point, propagator, epoch_start, epoch_end, constraint, None, None, None)
        .expect("Failed to compute accesses");

    windows
        .iter()
        .map(|w| {
            let aos = epoch_to_iso(&w.window_open);
            let los = epoch_to_iso(&w.window_close);
            AccessOutput {
                location_name: location.name.clone(),
                aos,
                los,
                duration_s: w.duration(),
            }
        })
        .collect()
}

fn benchmark_location(
    location: &InputLocation,
    propagator: &SGPPropagator,
    epoch_start: Epoch,
    epoch_end: Epoch,
    constraint: &ElevationConstraint,
    iterations: usize,
) -> LocationResult {
    let mut times: Vec<f64> = Vec::with_capacity(iterations);
    let mut last_accesses: Vec<AccessOutput> = Vec::new();

    for _ in 0..iterations {
        let start = Instant::now();
        last_accesses = compute_accesses_for_location(
            location,
            propagator,
            epoch_start,
            epoch_end,
            constraint,
        );
        let elapsed = start.elapsed().as_secs_f64();
        times.push(elapsed);
    }

    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    let std = variance.sqrt();

    LocationResult {
        location_name: location.name.clone(),
        lat: location.lat,
        lon: location.lon,
        accesses: last_accesses.clone(),
        timing: TimingOutput {
            mean_seconds: mean,
            std_seconds: std,
            iterations,
            num_windows: last_accesses.len(),
        },
    }
}

fn main() {
    // Initialize EOP with static provider (no file I/O needed)
    let eop = StaticEOPProvider::from_zero();
    brahe::eop::set_global_eop_provider(eop);

    // Read JSON input from stdin
    let stdin = io::stdin();
    let input: String = stdin.lock().lines().map(|l| l.unwrap()).collect();
    let config: BenchmarkInput = serde_json::from_str(&input).expect("Failed to parse input JSON");

    // Parse time window
    let window_open = parse_datetime(&config.window_open);
    let window_close = parse_datetime(&config.window_close);

    let epoch_start = Epoch::from_datetime(
        window_open.year() as u32,
        window_open.month() as u8,
        window_open.day() as u8,
        window_open.hour() as u8,
        window_open.minute() as u8,
        window_open.second() as f64,
        0.0,
        brahe::time::TimeSystem::UTC,
    );

    let epoch_end = Epoch::from_datetime(
        window_close.year() as u32,
        window_close.month() as u8,
        window_close.day() as u8,
        window_close.hour() as u8,
        window_close.minute() as u8,
        window_close.second() as f64,
        0.0,
        brahe::time::TimeSystem::UTC,
    );

    // Create propagator from TLE
    let propagator = SGPPropagator::from_tle(&config.tle_line1, &config.tle_line2, 60.0)
        .expect("Failed to create propagator from TLE");

    // Create constraint
    let constraint = ElevationConstraint::new(Some(config.min_elevation_deg), None)
        .expect("Failed to create elevation constraint");

    // Benchmark each location
    let mut results: Vec<LocationResult> = Vec::new();
    for location in &config.locations {
        let result = benchmark_location(
            location,
            &propagator,
            epoch_start,
            epoch_end,
            &constraint,
            config.iterations,
        );
        results.push(result);
    }

    // Output JSON to stdout
    let output = BenchmarkOutput { results };
    let json = serde_json::to_string(&output).expect("Failed to serialize output");
    io::stdout().write_all(json.as_bytes()).unwrap();
    io::stdout().write_all(b"\n").unwrap();
}
