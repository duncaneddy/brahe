//! Earth coordinate conversions using ANISE.
//!
//! ## What ANISE provides
//!
//! ANISE's `CartesianState` (a.k.a. `Orbit`) supports geodetic conversions
//! backed by PCK ellipsoid data loaded from `MetaAlmanac::latest()`:
//!
//! - `CartesianState::try_latlongalt(lat_deg, lon_deg, height_km, epoch, frame)`
//!   → ECEF position in the body-fixed frame (used for geodetic_to_ecef).
//! - `CartesianState::latlongalt()` → `(lat_deg, lon_deg, alt_km)`
//!   (used for ecef_to_geodetic via the Heikkinen iterative algorithm).
//! - `Almanac::azimuth_elevation_range_sez(rx, tx)` computes AER in the
//!   SEZ (South-East-Zenith) frame for a transmitter/receiver pair of
//!   `Orbit` states (used for ecef_to_azel).
//!
//! ## Dropped tasks (no ANISE equivalent)
//!
//! - `geocentric_to_ecef` and `ecef_to_geocentric`: ANISE has no spherical
//!   geocentric API. Implementing these with pure trig would not exercise
//!   ANISE functionality; they are excluded from the Nyx task list per the
//!   spec's "no hand-rolled math" rule.
//!
//! ## Unit conventions
//!
//! ANISE uses km / km/s internally. The benchmark framework uses m / m/s.
//! All inputs are converted /1000 on ingress and outputs ×1000 on egress
//! so that the results are directly comparable to brahe-Rust and Orekit.
//!
//! ## Frame
//!
//! `EARTH_ITRF93` (ANISE constant) is the Earth body-fixed frame. The
//! `frame_info` call fetches the PCK ellipsoid data (semi-major axis a and
//! flattening f from the pck11.pca kernel) and populates the Frame struct
//! so that the geodetic computations have access to the WGS-84 equivalent
//! values stored in the loaded PCK.

use crate::data::almanac;
use anise::constants::frames::EARTH_ITRF93;
use anise::math::cartesian::CartesianState;
use hifitime::Epoch;
use std::time::Instant;

/// A fixed J2000 epoch used for all coordinate conversions.
///
/// Geodetic and ECEF conversions are epoch-independent in the mathematical
/// sense (they only depend on the ellipsoid shape). We pass a fixed epoch
/// to satisfy ANISE's API; it does not affect results.
fn bench_epoch() -> Epoch {
    Epoch::from_gregorian_utc_at_midnight(2024, 1, 1)
}

/// Return the ITRF93 Earth body-fixed frame with PCK ellipsoid data loaded.
///
/// Cached with OnceLock so the PCK lookup is paid once per process, not
/// once per benchmark iteration.
fn earth_itrf93_frame() -> anise::frames::Frame {
    use std::sync::OnceLock;
    static FRAME: OnceLock<anise::frames::Frame> = OnceLock::new();
    *FRAME.get_or_init(|| {
        almanac()
            .frame_info(EARTH_ITRF93)
            .expect("EARTH_ITRF93 frame must be present in pck11.pca")
    })
}

/// Geodetic (lon_deg, lat_deg, alt_m) → ECEF (x_m, y_m, z_m).
///
/// Input format per benchmark spec: `params["points"]` is a list of
/// `[lon_deg, lat_deg, alt_m]` triples.
///
/// ANISE's `try_latlongalt` takes `(latitude, longitude, height_km)` —
/// note the reversed lat/lon order vs the benchmark's [lon, lat] order.
pub fn geodetic_to_ecef(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let points: Vec<Vec<f64>> = serde_json::from_value(params["points"].clone()).unwrap();
    let frame = earth_itrf93_frame();
    let epoch = bench_epoch();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(points.len());

        for pt in &points {
            let lon_deg = pt[0];
            let lat_deg = pt[1];
            let alt_km = pt[2] / 1000.0; // m → km

            // ANISE argument order: (latitude, longitude, height_km)
            let state = CartesianState::try_latlongalt(lat_deg, lon_deg, alt_km, epoch, frame)
                .expect("geodetic → ECEF conversion must succeed");

            results.push(vec![
                state.radius_km.x * 1000.0, // km → m
                state.radius_km.y * 1000.0,
                state.radius_km.z * 1000.0,
            ]);
        }

        all_times.push(start.elapsed().as_secs_f64());
        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

/// ECEF (x_m, y_m, z_m) → geodetic (lon_deg, lat_deg, alt_m).
///
/// Input format per benchmark spec: `params["points"]` is a list of
/// `[x_m, y_m, z_m]` ECEF position triples.
///
/// ANISE's `latlongalt()` returns `(lat_deg, lon_deg, alt_km)` —
/// note the returned lat/lon order; we reorder to `[lon, lat, alt_m]`
/// to match the brahe-Rust / Orekit output format.
pub fn ecef_to_geodetic(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let points: Vec<Vec<f64>> = serde_json::from_value(params["points"].clone()).unwrap();
    let frame = earth_itrf93_frame();
    let epoch = bench_epoch();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(points.len());

        for pt in &points {
            // Build a CartesianState from ECEF km position with zero velocity.
            let state = CartesianState::new(
                pt[0] / 1000.0, // x: m → km
                pt[1] / 1000.0, // y: m → km
                pt[2] / 1000.0, // z: m → km
                0.0,            // vx (km/s)
                0.0,            // vy (km/s)
                0.0,            // vz (km/s)
                epoch,
                frame,
            );

            // latlongalt() returns (lat_deg, lon_deg, alt_km)
            let (lat_deg, lon_deg, alt_km) = state
                .latlongalt()
                .expect("ECEF → geodetic conversion must succeed");

            // Reorder to match brahe/Orekit output: [lon_deg, lat_deg, alt_m]
            results.push(vec![lon_deg, lat_deg, alt_km * 1000.0]);
        }

        all_times.push(start.elapsed().as_secs_f64());
        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

/// ECEF station + satellite positions → azimuth (deg), elevation (deg), range (m).
///
/// Input format per benchmark spec: `params["pairs"]` is a list of objects:
/// ```json
/// { "station_ecef": [x_m, y_m, z_m],
///   "satellite_ecef": [x_m, y_m, z_m],
///   "station_geodetic": [lon_deg, lat_deg, alt_m] }
/// ```
///
/// ANISE's `azimuth_elevation_range_sez(rx, tx)` accepts two `Orbit`
/// states and returns an `AzElRange` with `azimuth_deg`, `elevation_deg`,
/// and `range_km`. Both states must be in the same body-fixed frame (ITRF93)
/// so that no frame transformation is needed (and the Almanac's BPC
/// rotation kernel is not exercised here — only the PCK ellipsoid data for
/// the SEZ DCM computation).
///
/// Output: `[azimuth_deg, elevation_deg, range_m]`
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
    let frame = earth_itrf93_frame();
    let epoch = bench_epoch();
    let alm = almanac();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(pairs.len());

        for pair in &pairs {
            // Transmitter (ground station) — ECEF, zero velocity.
            let tx = CartesianState::new(
                pair.station_ecef[0] / 1000.0,
                pair.station_ecef[1] / 1000.0,
                pair.station_ecef[2] / 1000.0,
                0.0,
                0.0,
                0.0,
                epoch,
                frame,
            );

            // Receiver (satellite) — ECEF, zero velocity.
            let rx = CartesianState::new(
                pair.satellite_ecef[0] / 1000.0,
                pair.satellite_ecef[1] / 1000.0,
                pair.satellite_ecef[2] / 1000.0,
                0.0,
                0.0,
                0.0,
                epoch,
                frame,
            );

            // Compute AER in the SEZ frame of the transmitter.
            // Both states are in ITRF93 so no frame transformation is needed.
            let aer = alm
                .azimuth_elevation_range_sez(
                    rx,   // receiver (satellite)
                    tx,   // transmitter (ground station)
                    None, // no obstructing body check
                    None, // no aberration correction
                )
                .expect("AER computation must succeed");

            results.push(vec![
                aer.azimuth_deg,
                aer.elevation_deg,
                aer.range_km * 1000.0, // km → m
            ]);
        }

        all_times.push(start.elapsed().as_secs_f64());
        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}
