/*!
 * Internal testing helper functions and fixtures.
 *
 * This module is only compiled for tests (#[cfg(test)] in mod.rs).
 */

use std::env;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use nalgebra::SVector;

use crate::AngleFormat;
use crate::constants::DEG2RAD;
use crate::constants::physical::R_EARTH;
use crate::eop::*;
use crate::math::angles::oe_to_radians;
use crate::orbit_dynamics::gravity::{GravityModel, GravityModelType, set_global_gravity_model};
use crate::orbits::keplerian::{geo_sma, perigee_velocity, sun_synchronous_inclination};
use crate::space_weather::{
    FileSpaceWeatherProvider, SpaceWeatherExtrapolation, set_global_space_weather_provider,
};
use crate::utils::get_naif_cache_dir;

// =============================================================================
// Test Orbit Fixtures - Keplerian Elements
// =============================================================================
// All elements are in the format: [a, e, i, raan, argp, M]
// Units: meters for semi-major axis, angles defined in degrees internally

/// Convert orbital elements defined in degrees to requested output format
fn convert_to_output_format(
    oe_degrees: SVector<f64, 6>,
    output_format: AngleFormat,
) -> SVector<f64, 6> {
    match output_format {
        AngleFormat::Radians => oe_to_radians(oe_degrees, AngleFormat::Degrees),
        AngleFormat::Degrees => oe_degrees,
    }
}

/// Standard LEO test orbit - 500 km circular, sun-synchronous inclination
/// Elements: [a, e, i, Ω, ω, M] in meters and specified angle format
pub(crate) fn fixture_orbit_leo(angle_format: AngleFormat) -> SVector<f64, 6> {
    let a = R_EARTH + 500e3;
    let e = 0.001;
    let oe = SVector::<f64, 6>::new(
        a,                                                       // 6,878,136.3 m
        e,                                                       // Near-circular
        sun_synchronous_inclination(a, e, AngleFormat::Degrees), // Sun-synchronous inclination
        15.0,                                                    // RAAN
        30.0,                                                    // Argument of perigee
        45.0,                                                    // Mean anomaly
    );
    convert_to_output_format(oe, angle_format)
}

/// Geostationary orbit test fixture
/// Elements: [a, e, i, Ω, ω, M] in meters and specified angle format
pub(crate) fn fixture_orbit_geo(angle_format: AngleFormat) -> SVector<f64, 6> {
    let oe = SVector::<f64, 6>::new(
        geo_sma(), // GEO semi-major axis
        0.0001,    // Nearly circular
        0.0,       // Equatorial
        0.0,       // RAAN
        0.0,       // Argument of perigee
        0.0,       // Mean anomaly
    );
    convert_to_output_format(oe, angle_format)
}

/// Sun-synchronous orbit test fixture - 700 km altitude
/// Elements: [a, e, i, Ω, ω, M] in meters and specified angle format
pub(crate) fn fixture_orbit_sso(angle_format: AngleFormat) -> SVector<f64, 6> {
    let a = R_EARTH + 700e3;
    let e = 0.001;
    let oe = SVector::<f64, 6>::new(
        a,                                                       // 7,078,136.3 m
        e,                                                       // Near-circular
        sun_synchronous_inclination(a, e, AngleFormat::Degrees), // Sun-synchronous inclination
        45.0,                                                    // RAAN
        90.0,                                                    // Argument of perigee
        0.0,                                                     // Mean anomaly
    );
    convert_to_output_format(oe, angle_format)
}

/// Molniya orbit test fixture - Highly elliptical, critical inclination
/// Elements: [a, e, i, Ω, ω, M] in meters and specified angle format
pub(crate) fn fixture_orbit_molniya(angle_format: AngleFormat) -> SVector<f64, 6> {
    let oe = SVector::<f64, 6>::new(
        26600e3, // Semi-major axis
        0.74,    // High eccentricity
        63.4,    // Critical inclination
        270.0,   // RAAN
        270.0,   // Argument of perigee
        0.0,     // Mean anomaly at perigee
    );
    convert_to_output_format(oe, angle_format)
}

/// Polar orbit test fixture - 800 km altitude
/// Elements: [a, e, i, Ω, ω, M] in meters and specified angle format
pub(crate) fn fixture_orbit_polar(angle_format: AngleFormat) -> SVector<f64, 6> {
    let oe = SVector::<f64, 6>::new(
        R_EARTH + 800e3, // 7,178,136.3 m
        0.001,           // Near-circular
        90.0,            // Polar inclination
        0.0,             // RAAN
        0.0,             // Argument of perigee
        0.0,             // Mean anomaly
    );
    convert_to_output_format(oe, angle_format)
}

/// Medium eccentricity orbit test fixture
/// Elements: [a, e, i, Ω, ω, M] in meters and specified angle format
pub(crate) fn fixture_orbit_elliptical(angle_format: AngleFormat) -> SVector<f64, 6> {
    let oe = SVector::<f64, 6>::new(
        R_EARTH + 1000e3, // 7,378,136.3 m
        0.3,              // Medium eccentricity
        45.0,             // 45 degree inclination
        60.0,             // RAAN
        120.0,            // Argument of perigee
        0.0,              // Mean anomaly
    );
    convert_to_output_format(oe, angle_format)
}

/// Retrograde orbit test fixture
/// Elements: [a, e, i, Ω, ω, M] in meters and specified angle format
pub(crate) fn fixture_orbit_retrograde(angle_format: AngleFormat) -> SVector<f64, 6> {
    let oe = SVector::<f64, 6>::new(
        R_EARTH + 600e3, // 6,978,136.3 m
        0.01,            // Low eccentricity
        150.0,           // Retrograde inclination
        30.0,            // RAAN
        45.0,            // Argument of perigee
        0.0,             // Mean anomaly
    );
    convert_to_output_format(oe, angle_format)
}

/// Near-equatorial orbit test fixture
/// Elements: [a, e, i, Ω, ω, M] in meters and specified angle format
pub(crate) fn fixture_orbit_equatorial(angle_format: AngleFormat) -> SVector<f64, 6> {
    let oe = SVector::<f64, 6>::new(
        R_EARTH + 550e3, // 6,928,136.3 m
        0.005,           // Low eccentricity
        5.0,             // Near-equatorial
        0.0,             // RAAN
        0.0,             // Argument of perigee
        0.0,             // Mean anomaly
    );
    convert_to_output_format(oe, angle_format)
}

// =============================================================================
// Test State Fixtures - ECI Cartesian States
// =============================================================================
// All states are in the format: [x, y, z, vx, vy, vz]
// Units: meters for position, m/s for velocity

/// Test altitude for ECI state fixtures (650 km above Earth's surface)
pub const FIXTURE_STATE_ALTITUDE: f64 = 650e3;

/// Orbital radius for test states
pub const FIXTURE_STATE_RADIUS: f64 = R_EARTH + FIXTURE_STATE_ALTITUDE;

/// Circular orbital velocity at FIXTURE_STATE_RADIUS
/// Computed as sqrt(GM_EARTH / r) for circular orbit
pub(crate) fn fixture_circular_velocity() -> f64 {
    perigee_velocity(FIXTURE_STATE_RADIUS, 0.0)
}

// -----------------------------------------------------------------------------
// Equatorial Plane States
// -----------------------------------------------------------------------------

/// ECI state with position along +X axis, velocity along +Y axis
/// Position: [R, 0, 0], Velocity: [0, +v, 0]
pub(crate) fn fixture_state_eci_plus_x() -> SVector<f64, 6> {
    let v = fixture_circular_velocity();
    SVector::<f64, 6>::new(FIXTURE_STATE_RADIUS, 0.0, 0.0, 0.0, v, 0.0)
}

/// ECI state with position along -X axis, velocity along -Y axis
/// Position: [-R, 0, 0], Velocity: [0, -v, 0]
pub(crate) fn fixture_state_eci_minus_x() -> SVector<f64, 6> {
    let v = fixture_circular_velocity();
    SVector::<f64, 6>::new(-FIXTURE_STATE_RADIUS, 0.0, 0.0, 0.0, -v, 0.0)
}

/// ECI state with position along +Y axis, velocity along -X axis
/// Position: [0, R, 0], Velocity: [-v, 0, 0]
pub(crate) fn fixture_state_eci_plus_y() -> SVector<f64, 6> {
    let v = fixture_circular_velocity();
    SVector::<f64, 6>::new(0.0, FIXTURE_STATE_RADIUS, 0.0, -v, 0.0, 0.0)
}

/// ECI state with position along -Y axis, velocity along +X axis
/// Position: [0, -R, 0], Velocity: [+v, 0, 0]
pub(crate) fn fixture_state_eci_minus_y() -> SVector<f64, 6> {
    let v = fixture_circular_velocity();
    SVector::<f64, 6>::new(0.0, -FIXTURE_STATE_RADIUS, 0.0, v, 0.0, 0.0)
}

// -----------------------------------------------------------------------------
// North Pole States - Position at +Z, with 4 velocity directions
// -----------------------------------------------------------------------------

/// ECI state at North Pole with velocity in +X direction
/// Position: [0, 0, R], Velocity: [+v, 0, 0]
pub(crate) fn fixture_state_eci_north_vel_px() -> SVector<f64, 6> {
    let v = fixture_circular_velocity();
    SVector::<f64, 6>::new(0.0, 0.0, FIXTURE_STATE_RADIUS, v, 0.0, 0.0)
}

/// ECI state at North Pole with velocity in -X direction
/// Position: [0, 0, R], Velocity: [-v, 0, 0]
pub(crate) fn fixture_state_eci_north_vel_nx() -> SVector<f64, 6> {
    let v = fixture_circular_velocity();
    SVector::<f64, 6>::new(0.0, 0.0, FIXTURE_STATE_RADIUS, -v, 0.0, 0.0)
}

/// ECI state at North Pole with velocity in +Y direction
/// Position: [0, 0, R], Velocity: [0, +v, 0]
pub(crate) fn fixture_state_eci_north_vel_py() -> SVector<f64, 6> {
    let v = fixture_circular_velocity();
    SVector::<f64, 6>::new(0.0, 0.0, FIXTURE_STATE_RADIUS, 0.0, v, 0.0)
}

/// ECI state at North Pole with velocity in -Y direction
/// Position: [0, 0, R], Velocity: [0, -v, 0]
pub(crate) fn fixture_state_eci_north_vel_ny() -> SVector<f64, 6> {
    let v = fixture_circular_velocity();
    SVector::<f64, 6>::new(0.0, 0.0, FIXTURE_STATE_RADIUS, 0.0, -v, 0.0)
}

// -----------------------------------------------------------------------------
// South Pole States - Position at -Z, with 4 velocity directions
// -----------------------------------------------------------------------------

/// ECI state at South Pole with velocity in +X direction
/// Position: [0, 0, -R], Velocity: [+v, 0, 0]
pub(crate) fn fixture_state_eci_south_vel_px() -> SVector<f64, 6> {
    let v = fixture_circular_velocity();
    SVector::<f64, 6>::new(0.0, 0.0, -FIXTURE_STATE_RADIUS, v, 0.0, 0.0)
}

/// ECI state at South Pole with velocity in -X direction
/// Position: [0, 0, -R], Velocity: [-v, 0, 0]
pub(crate) fn fixture_state_eci_south_vel_nx() -> SVector<f64, 6> {
    let v = fixture_circular_velocity();
    SVector::<f64, 6>::new(0.0, 0.0, -FIXTURE_STATE_RADIUS, -v, 0.0, 0.0)
}

/// ECI state at South Pole with velocity in +Y direction
/// Position: [0, 0, -R], Velocity: [0, +v, 0]
pub(crate) fn fixture_state_eci_south_vel_py() -> SVector<f64, 6> {
    let v = fixture_circular_velocity();
    SVector::<f64, 6>::new(0.0, 0.0, -FIXTURE_STATE_RADIUS, 0.0, v, 0.0)
}

/// ECI state at South Pole with velocity in -Y direction
/// Position: [0, 0, -R], Velocity: [0, -v, 0]
pub(crate) fn fixture_state_eci_south_vel_ny() -> SVector<f64, 6> {
    let v = fixture_circular_velocity();
    SVector::<f64, 6>::new(0.0, 0.0, -FIXTURE_STATE_RADIUS, 0.0, -v, 0.0)
}

// =============================================================================
// Global Test Setup Functions
// =============================================================================
// These functions are only compiled for tests since they're used by other test modules.

/// Initialize global EOP provider with test data for unit testing.
///
/// Loads `test_assets/finals.all.iau2000.txt` and configures with Hold extrapolation
/// and linear interpolation. Use at the start of tests requiring EOP data for frame
/// transformations.
///
/// # Panics
/// Panics if test asset file cannot be found or loaded.
pub(crate) fn setup_global_test_eop() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let filepath = Path::new(&manifest_dir)
        .join("test_assets")
        .join("finals.all.iau2000.txt");

    let eop_extrapolation = EOPExtrapolation::Hold;
    let eop_interpolation = true;

    let eop = FileEOPProvider::from_file(&filepath, eop_interpolation, eop_extrapolation).unwrap();
    set_global_eop_provider(eop);
}

/// Initialize global EOP provider with original Brahe test data for validation tests.
///
/// Loads `test_assets/brahe_original_eop_file.txt` for comparing against original Brahe
/// implementation. Use for regression testing and validation against known results.
///
/// # Panics
/// Panics if legacy test file cannot be found or loaded.
pub(crate) fn setup_global_test_eop_original_brahe() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let filepath = Path::new(&manifest_dir)
        .join("test_assets")
        .join("brahe_original_eop_file.txt");

    let eop_extrapolation = EOPExtrapolation::Hold;
    let eop_interpolation = true;

    let eop = FileEOPProvider::from_file(&filepath, eop_interpolation, eop_extrapolation).unwrap();
    set_global_eop_provider(eop);
}

/// Initialize global gravity model with EGM2008_120 for orbit dynamics tests.
///
/// Sets up EGM2008 (degree/order 120) as the global gravity model. Use at the start
/// of tests requiring high-fidelity gravity for orbit propagation or perturbation analysis.
pub(crate) fn setup_global_test_gravity_model() {
    let gravity_model = GravityModel::from_model_type(&GravityModelType::EGM2008_120).unwrap();
    set_global_gravity_model(gravity_model);
}

/// Initialize the global SPICE kernel registry with the DE440s test asset.
///
/// Copies `test_assets/de440s.bsp` into the NAIF cache directory (if present)
/// and loads it into the global registry. Safe to call from multiple tests;
/// initialization runs once.
///
/// # Panics
/// Panics if the test asset exists but cannot be copied or loaded.
pub(crate) fn setup_global_test_spice() {
    static SPICE_INIT: OnceLock<()> = OnceLock::new();

    SPICE_INIT.get_or_init(|| {
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let test_asset_path = Path::new(&manifest_dir)
            .join("test_assets")
            .join("de440s.bsp");

        if !test_asset_path.exists() {
            return;
        }

        let cache_dir = get_naif_cache_dir().expect("Failed to get NAIF cache dir");
        let cache_path = Path::new(&cache_dir).join("de440s.bsp");
        fs::copy(&test_asset_path, &cache_path).expect("Failed to copy test asset to cache");

        crate::spice::load_spice_kernel("de440s").expect("Failed to load DE440s test kernel");
    });
}

/// Build a minimal little-endian `DAF/SPK` kernel with one constant-position
/// Type 2 segment per `(target, center, x_km)` entry, all covering ET
/// `[-1e10, 1e10]` (so any modern epoch is in range). Each segment's position
/// is the constant `x_km` on the x-axis (y = z = 0), so its velocity is
/// exactly zero.
///
/// Seed the returned bytes into a `BRAHE_CACHE`-redirected cache under a
/// kernel's [`crate::spice::SPICEKernel::filename`] to make the download layer
/// return it without touching the network, for offline registry/positions
/// tests. The single record directory spans the full coverage so any epoch in
/// range resolves.
pub(crate) fn synthetic_spk_kernel_bytes(segments: &[(i32, i32, f64)]) -> Vec<u8> {
    let start_et = -1.0e10_f64;
    let end_et = 1.0e10_f64;
    let mid = (start_et + end_et) / 2.0;
    let radius = (end_et - start_et) / 2.0;
    let intlen = end_et - start_et;
    let rsize = 8usize; // 2 + 3 * (degree 1 + 1)
    let seg_words = 12usize; // 8-word record + 4-word trailer
    let data_words_start = 385usize; // (record 4 - 1) * 128 + 1
    let ss = 5usize; // ND(2) + ceil(NI(6)/2)(3)

    let last_word = data_words_start - 1 + segments.len() * seg_words;
    let n_records = (last_word * 8).div_ceil(1024).max(4);
    let mut file = vec![0u8; n_records * 1024];

    file[..8].copy_from_slice(b"DAF/SPK ");
    file[8..12].copy_from_slice(&2i32.to_le_bytes()); // ND
    file[12..16].copy_from_slice(&6i32.to_le_bytes()); // NI
    file[76..80].copy_from_slice(&2i32.to_le_bytes()); // FWARD -> record 2
    file[80..84].copy_from_slice(&2i32.to_le_bytes()); // BWARD
    file[84..88].copy_from_slice(&500i32.to_le_bytes()); // FREE
    file[88..96].copy_from_slice(b"LTL-IEEE");

    // Summary record (record 2): NEXT=0, PREV=0, NSUM=segments.len()
    let rec = 1024;
    file[rec + 16..rec + 24].copy_from_slice(&(segments.len() as f64).to_le_bytes());
    for (i, (target, center, x_km)) in segments.iter().enumerate() {
        let s_off = rec + (3 + i * ss) * 8;
        file[s_off..s_off + 8].copy_from_slice(&start_et.to_le_bytes());
        file[s_off + 8..s_off + 16].copy_from_slice(&end_et.to_le_bytes());
        let start_addr = (data_words_start + i * seg_words) as i32;
        let end_addr = start_addr + seg_words as i32 - 1;
        for (k, v) in [*target, *center, 1, 2, start_addr, end_addr]
            .iter()
            .enumerate()
        {
            let off = s_off + 16 + k * 4;
            file[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }

        // Name record (record 3): spaces, one name slot per summary.
        let name_off = 2048 + i * (8 * ss);
        for b in &mut file[name_off..name_off + 8 * ss] {
            *b = b' ';
        }

        // Data: [MID, RADIUS, x0, x1, y0, y1, z0, z1, INIT, INTLEN, RSIZE, N]
        let data = [
            mid,
            radius,
            *x_km,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            start_et,
            intlen,
            rsize as f64,
            1.0,
        ];
        let d_off = (start_addr as usize - 1) * 8;
        for (j, v) in data.iter().enumerate() {
            file[d_off + j * 8..d_off + j * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
    }
    file
}

/// Build a minimal little-endian `DAF/PCK` kernel with one Type 2 segment for
/// `frame_id` relative to frame 1, covering ET `[0, 1000]` with linear Euler
/// angles. For offline tests needing a PCK-typed kernel seeded into a
/// `BRAHE_CACHE`-redirected cache.
pub(crate) fn synthetic_pck_kernel_bytes(frame_id: i32) -> Vec<u8> {
    let rsize = 8usize; // 2 + 3 * (degree 1 + 1)
    let data: [f64; 12] = [
        500.0,
        500.0, // MID, RADIUS
        0.1,
        0.2, // phi
        0.3,
        0.0, // delta
        0.4,
        0.5, // w
        0.0,
        1000.0,
        rsize as f64,
        1.0, // INIT, INTLEN, RSIZE, N
    ];

    let mut file = vec![0u8; 4 * 1024];
    file[..8].copy_from_slice(b"DAF/PCK ");
    file[8..12].copy_from_slice(&2i32.to_le_bytes()); // ND
    file[12..16].copy_from_slice(&5i32.to_le_bytes()); // NI
    file[76..80].copy_from_slice(&2i32.to_le_bytes()); // FWARD -> record 2
    file[80..84].copy_from_slice(&2i32.to_le_bytes()); // BWARD
    file[84..88].copy_from_slice(&500i32.to_le_bytes()); // FREE
    file[88..96].copy_from_slice(b"LTL-IEEE");

    let rec = 1024;
    file[rec + 16..rec + 24].copy_from_slice(&1f64.to_le_bytes()); // NSUM
    file[rec + 24..rec + 32].copy_from_slice(&0f64.to_le_bytes()); // start_et
    file[rec + 32..rec + 40].copy_from_slice(&1000f64.to_le_bytes()); // end_et
    let start_addr = 385i32;
    let end_addr = start_addr + data.len() as i32 - 1;
    for (i, v) in [frame_id, 1, 2, start_addr, end_addr].iter().enumerate() {
        let off = rec + 40 + i * 4;
        file[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
    for b in &mut file[2048..2048 + 40] {
        *b = b' ';
    }
    for (i, v) in data.iter().enumerate() {
        let off = 3 * 1024 + i * 8;
        file[off..off + 8].copy_from_slice(&v.to_le_bytes());
    }
    file
}

/// Scoped redirect of the `BRAHE_CACHE` environment variable to a fresh
/// temporary directory, restoring the previous value on drop. Lets offline
/// tests seed synthetic kernels under `$BRAHE_CACHE/naif/<filename>` so the
/// NAIF download layer returns them from cache without a network fetch.
///
/// `BRAHE_CACHE` is process-global, so every test using this must be
/// `#[serial]`.
pub(crate) struct CacheRedirect {
    _dir: tempfile::TempDir,
    naif_dir: std::path::PathBuf,
    prev: Option<String>,
}

impl CacheRedirect {
    /// Redirect `BRAHE_CACHE` to a new tempdir and create its `naif`
    /// subdirectory.
    pub(crate) fn new() -> Self {
        let prev = env::var("BRAHE_CACHE").ok();
        let dir = tempfile::tempdir().unwrap();
        // SAFETY: single-threaded within a #[serial] test; no other thread
        // reads the environment concurrently.
        unsafe {
            env::set_var("BRAHE_CACHE", dir.path());
        }
        let naif_dir = dir.path().join("naif");
        fs::create_dir_all(&naif_dir).unwrap();
        CacheRedirect {
            _dir: dir,
            naif_dir,
            prev,
        }
    }

    /// Write `bytes` into the redirected NAIF cache under `filename` (the
    /// kernel's [`crate::spice::SPICEKernel::filename`]).
    pub(crate) fn seed(&self, filename: &str, bytes: &[u8]) {
        fs::write(self.naif_dir.join(filename), bytes).unwrap();
    }

    /// Copy the committed real de440s test asset into the redirected cache, if
    /// present. Because `BRAHE_CACHE` is process-global, a concurrent test may
    /// (re)load de440s while this redirect is active; seeding the real kernel
    /// keeps that read valid instead of forcing a network download.
    pub(crate) fn seed_real_de440s(&self) {
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("test_assets")
            .join("de440s.bsp");
        if src.exists() {
            fs::copy(&src, self.naif_dir.join("de440s.bsp")).unwrap();
        }
    }
}

impl Drop for CacheRedirect {
    fn drop(&mut self) {
        // SAFETY: single-threaded within a #[serial] test.
        unsafe {
            match &self.prev {
                Some(v) => env::set_var("BRAHE_CACHE", v),
                None => env::remove_var("BRAHE_CACHE"),
            }
        }
    }
}

/// Initialize global space weather provider with test data for unit testing.
///
/// Loads `test_assets/sw19571001.txt` and configures with Hold extrapolation.
/// Use at the start of tests requiring space weather data for atmospheric
/// density calculations or other space environment models.
///
/// # Panics
/// Panics if test asset file cannot be found or loaded.
pub(crate) fn setup_global_test_space_weather() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let filepath = Path::new(&manifest_dir)
        .join("test_assets")
        .join("sw19571001.txt");

    let sw =
        FileSpaceWeatherProvider::from_file(&filepath, SpaceWeatherExtrapolation::Hold).unwrap();
    set_global_space_weather_provider(sw);
}

/// Get the path to the space weather test file for unit testing.
///
/// Returns the path to `test_assets/sw19571001.txt`. Use when tests need to
/// work with the file directly (e.g., caching provider tests that need to
/// copy data to a temp directory).
///
/// # Panics
/// Panics if CARGO_MANIFEST_DIR is not set.
pub(crate) fn get_test_space_weather_filepath() -> std::path::PathBuf {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest_dir)
        .join("test_assets")
        .join("sw19571001.txt")
}

#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::AngleFormat;
    use crate::constants::physical::GM_EARTH;
    use crate::orbits::keplerian::sun_synchronous_inclination;
    use approx::assert_abs_diff_eq;

    // =========================================================================
    // Orbital Fixture Tests
    // =========================================================================

    #[test]
    fn test_fixture_orbit_leo_elements() {
        // Verify LEO orbit fixture values in radians
        let leo = fixture_orbit_leo(AngleFormat::Radians);
        assert_abs_diff_eq!(leo[0], R_EARTH + 500e3, epsilon = 1.0);
        assert_abs_diff_eq!(leo[1], 0.001, epsilon = 1e-6);
        // Verify sun-synchronous inclination is calculated correctly (~97.4 deg for 500 km)
        let expected_inc =
            sun_synchronous_inclination(R_EARTH + 500e3, 0.001, AngleFormat::Radians);
        assert_abs_diff_eq!(leo[2], expected_inc, epsilon = 1e-6);

        // Verify degrees output
        let leo_deg = fixture_orbit_leo(AngleFormat::Degrees);
        assert_abs_diff_eq!(leo_deg[0], R_EARTH + 500e3, epsilon = 1.0);
        assert_abs_diff_eq!(leo_deg[3], 15.0, epsilon = 1e-6); // RAAN in degrees
        assert_abs_diff_eq!(leo_deg[4], 30.0, epsilon = 1e-6); // Arg perigee in degrees
        assert_abs_diff_eq!(leo_deg[5], 45.0, epsilon = 1e-6); // Mean anomaly in degrees
    }

    #[test]
    fn test_fixture_orbit_geo_elements() {
        // Verify GEO orbit fixture values
        let geo = fixture_orbit_geo(AngleFormat::Radians);
        assert_abs_diff_eq!(geo[0], geo_sma(), epsilon = 1.0);
        assert!(geo[2] < 1e-6); // Equatorial
    }

    #[test]
    fn test_fixture_orbit_sso_elements() {
        // Verify SSO orbit fixture values
        let sso = fixture_orbit_sso(AngleFormat::Radians);
        assert_abs_diff_eq!(sso[0], R_EARTH + 700e3, epsilon = 1.0);
        assert_abs_diff_eq!(sso[1], 0.001, epsilon = 1e-6);
        // Verify sun-synchronous inclination is calculated correctly
        let expected_inc =
            sun_synchronous_inclination(R_EARTH + 700e3, 0.001, AngleFormat::Radians);
        assert_abs_diff_eq!(sso[2], expected_inc, epsilon = 1e-6);
    }

    #[test]
    fn test_fixture_orbit_molniya_elements() {
        // Verify Molniya orbit fixture values
        let molniya = fixture_orbit_molniya(AngleFormat::Radians);
        assert_abs_diff_eq!(molniya[0], 26600e3, epsilon = 1.0);
        assert_abs_diff_eq!(molniya[1], 0.74, epsilon = 1e-6);
        assert_abs_diff_eq!(molniya[2], 63.4 * DEG2RAD, epsilon = 1e-6);

        // Verify degrees output
        let molniya_deg = fixture_orbit_molniya(AngleFormat::Degrees);
        assert_abs_diff_eq!(molniya_deg[2], 63.4, epsilon = 1e-6); // Critical inclination
        assert_abs_diff_eq!(molniya_deg[3], 270.0, epsilon = 1e-6); // RAAN
    }

    #[test]
    fn test_fixture_orbit_polar_elements() {
        // Verify polar orbit fixture values in radians
        let polar = fixture_orbit_polar(AngleFormat::Radians);
        assert_abs_diff_eq!(polar[0], R_EARTH + 800e3, epsilon = 1.0);
        assert_abs_diff_eq!(polar[1], 0.001, epsilon = 1e-6);
        assert_abs_diff_eq!(polar[2], 90.0 * DEG2RAD, epsilon = 1e-6); // Polar inclination

        // Verify degrees output
        let polar_deg = fixture_orbit_polar(AngleFormat::Degrees);
        assert_abs_diff_eq!(polar_deg[0], R_EARTH + 800e3, epsilon = 1.0);
        assert_abs_diff_eq!(polar_deg[2], 90.0, epsilon = 1e-6); // Polar inclination
        assert_abs_diff_eq!(polar_deg[3], 0.0, epsilon = 1e-6); // RAAN
        assert_abs_diff_eq!(polar_deg[4], 0.0, epsilon = 1e-6); // Arg perigee
        assert_abs_diff_eq!(polar_deg[5], 0.0, epsilon = 1e-6); // Mean anomaly
    }

    #[test]
    fn test_fixture_orbit_elliptical_elements() {
        // Verify elliptical orbit fixture values in radians
        let elliptical = fixture_orbit_elliptical(AngleFormat::Radians);
        assert_abs_diff_eq!(elliptical[0], R_EARTH + 1000e3, epsilon = 1.0);
        assert_abs_diff_eq!(elliptical[1], 0.3, epsilon = 1e-6); // Medium eccentricity
        assert_abs_diff_eq!(elliptical[2], 45.0 * DEG2RAD, epsilon = 1e-6);

        // Verify degrees output
        let elliptical_deg = fixture_orbit_elliptical(AngleFormat::Degrees);
        assert_abs_diff_eq!(elliptical_deg[0], R_EARTH + 1000e3, epsilon = 1.0);
        assert_abs_diff_eq!(elliptical_deg[2], 45.0, epsilon = 1e-6); // Inclination
        assert_abs_diff_eq!(elliptical_deg[3], 60.0, epsilon = 1e-6); // RAAN
        assert_abs_diff_eq!(elliptical_deg[4], 120.0, epsilon = 1e-6); // Arg perigee
        assert_abs_diff_eq!(elliptical_deg[5], 0.0, epsilon = 1e-6); // Mean anomaly
    }

    #[test]
    fn test_fixture_orbit_retrograde_elements() {
        // Verify retrograde orbit fixture values in radians
        let retrograde = fixture_orbit_retrograde(AngleFormat::Radians);
        assert_abs_diff_eq!(retrograde[0], R_EARTH + 600e3, epsilon = 1.0);
        assert_abs_diff_eq!(retrograde[1], 0.01, epsilon = 1e-6);
        assert_abs_diff_eq!(retrograde[2], 150.0 * DEG2RAD, epsilon = 1e-6); // Retrograde inclination

        // Verify degrees output
        let retrograde_deg = fixture_orbit_retrograde(AngleFormat::Degrees);
        assert_abs_diff_eq!(retrograde_deg[0], R_EARTH + 600e3, epsilon = 1.0);
        assert_abs_diff_eq!(retrograde_deg[2], 150.0, epsilon = 1e-6); // Retrograde inclination
        assert_abs_diff_eq!(retrograde_deg[3], 30.0, epsilon = 1e-6); // RAAN
        assert_abs_diff_eq!(retrograde_deg[4], 45.0, epsilon = 1e-6); // Arg perigee
        assert_abs_diff_eq!(retrograde_deg[5], 0.0, epsilon = 1e-6); // Mean anomaly
    }

    #[test]
    fn test_fixture_orbit_equatorial_elements() {
        // Verify equatorial orbit fixture values in radians
        let equatorial = fixture_orbit_equatorial(AngleFormat::Radians);
        assert_abs_diff_eq!(equatorial[0], R_EARTH + 550e3, epsilon = 1.0);
        assert_abs_diff_eq!(equatorial[1], 0.005, epsilon = 1e-6);
        assert_abs_diff_eq!(equatorial[2], 5.0 * DEG2RAD, epsilon = 1e-6); // Near-equatorial

        // Verify degrees output
        let equatorial_deg = fixture_orbit_equatorial(AngleFormat::Degrees);
        assert_abs_diff_eq!(equatorial_deg[0], R_EARTH + 550e3, epsilon = 1.0);
        assert_abs_diff_eq!(equatorial_deg[2], 5.0, epsilon = 1e-6); // Near-equatorial
        assert_abs_diff_eq!(equatorial_deg[3], 0.0, epsilon = 1e-6); // RAAN
        assert_abs_diff_eq!(equatorial_deg[4], 0.0, epsilon = 1e-6); // Arg perigee
        assert_abs_diff_eq!(equatorial_deg[5], 0.0, epsilon = 1e-6); // Mean anomaly
    }

    #[test]
    fn test_fixture_circular_velocity_calculation() {
        // Verify circular velocity is reasonable (should be ~7.5 km/s at LEO)
        let v = fixture_circular_velocity();
        assert!(
            v > 7000.0 && v < 8000.0,
            "Velocity {} out of expected range",
            v
        );

        // Verify against vis-viva equation: v = sqrt(GM/r)
        let expected = (GM_EARTH / FIXTURE_STATE_RADIUS).sqrt();
        assert_abs_diff_eq!(v, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_eci_state_equatorial_plus_x() {
        let state = fixture_state_eci_plus_x();
        let v = fixture_circular_velocity();

        // Position along +X
        assert_abs_diff_eq!(state[0], FIXTURE_STATE_RADIUS, epsilon = 1e-6);
        assert_abs_diff_eq!(state[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(state[2], 0.0, epsilon = 1e-6);

        // Velocity along +Y
        assert_abs_diff_eq!(state[3], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(state[4], v, epsilon = 1e-6);
        assert_abs_diff_eq!(state[5], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_eci_state_equatorial_minus_x() {
        let state = fixture_state_eci_minus_x();
        let v = fixture_circular_velocity();

        // Position along -X
        assert_abs_diff_eq!(state[0], -FIXTURE_STATE_RADIUS, epsilon = 1e-6);
        assert_abs_diff_eq!(state[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(state[2], 0.0, epsilon = 1e-6);

        // Velocity along -Y
        assert_abs_diff_eq!(state[3], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(state[4], -v, epsilon = 1e-6);
        assert_abs_diff_eq!(state[5], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_eci_state_north_pole() {
        let state = fixture_state_eci_north_vel_px();
        let v = fixture_circular_velocity();

        // Position at North Pole (+Z)
        assert_abs_diff_eq!(state[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(state[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(state[2], FIXTURE_STATE_RADIUS, epsilon = 1e-6);

        // Velocity along +X
        assert_abs_diff_eq!(state[3], v, epsilon = 1e-6);
        assert_abs_diff_eq!(state[4], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(state[5], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_eci_state_south_pole() {
        let state = fixture_state_eci_south_vel_ny();
        let v = fixture_circular_velocity();

        // Position at South Pole (-Z)
        assert_abs_diff_eq!(state[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(state[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(state[2], -FIXTURE_STATE_RADIUS, epsilon = 1e-6);

        // Velocity along -Y
        assert_abs_diff_eq!(state[3], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(state[4], -v, epsilon = 1e-6);
        assert_abs_diff_eq!(state[5], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_all_state_velocities_have_correct_magnitude() {
        // All test states should have the same velocity magnitude
        let expected_v = fixture_circular_velocity();

        let states = vec![
            fixture_state_eci_plus_x(),
            fixture_state_eci_minus_x(),
            fixture_state_eci_plus_y(),
            fixture_state_eci_minus_y(),
            fixture_state_eci_north_vel_px(),
            fixture_state_eci_north_vel_nx(),
            fixture_state_eci_north_vel_py(),
            fixture_state_eci_north_vel_ny(),
            fixture_state_eci_south_vel_px(),
            fixture_state_eci_south_vel_nx(),
            fixture_state_eci_south_vel_py(),
            fixture_state_eci_south_vel_ny(),
        ];

        for state in states {
            let v_mag = (state[3].powi(2) + state[4].powi(2) + state[5].powi(2)).sqrt();
            assert_abs_diff_eq!(v_mag, expected_v, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_all_state_positions_have_correct_radius() {
        // All test states should have the same position magnitude
        let states = vec![
            fixture_state_eci_plus_x(),
            fixture_state_eci_minus_x(),
            fixture_state_eci_plus_y(),
            fixture_state_eci_minus_y(),
            fixture_state_eci_north_vel_px(),
            fixture_state_eci_north_vel_nx(),
            fixture_state_eci_north_vel_py(),
            fixture_state_eci_north_vel_ny(),
            fixture_state_eci_south_vel_px(),
            fixture_state_eci_south_vel_nx(),
            fixture_state_eci_south_vel_py(),
            fixture_state_eci_south_vel_ny(),
        ];

        for state in states {
            let r_mag = (state[0].powi(2) + state[1].powi(2) + state[2].powi(2)).sqrt();
            assert_abs_diff_eq!(r_mag, FIXTURE_STATE_RADIUS, epsilon = 1e-6);
        }
    }

    // =========================================================================
    // Global Setup Function Tests
    // =========================================================================

    #[test]
    fn test_setup_global_test_eop() {
        // Test that setup_global_test_eop runs without panicking
        setup_global_test_eop();

        // Verify EOP is initialized by checking we can get values
        let result = get_global_ut1_utc(58849.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_setup_global_test_eop_original_brahe() {
        // Test that setup with original brahe EOP file works
        setup_global_test_eop_original_brahe();

        // Verify EOP is initialized
        let result = get_global_ut1_utc(58849.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_setup_global_test_gravity_model() {
        // Test that gravity model setup works
        setup_global_test_gravity_model();

        // This function has side effects but doesn't return anything to verify
        // The fact that it doesn't panic is the test
    }

    #[test]
    fn test_setup_global_test_space_weather() {
        use crate::space_weather::get_global_kp;

        // Test that setup_global_test_space_weather runs without panicking
        setup_global_test_space_weather();

        // Verify space weather is initialized by checking we can get values
        let result = get_global_kp(60000.0);
        assert!(result.is_ok());
        let kp = result.unwrap();
        assert!((0.0..=9.0).contains(&kp));
    }

    #[test]
    fn test_get_test_space_weather_filepath() {
        let filepath = get_test_space_weather_filepath();
        assert!(filepath.exists());
        assert!(filepath.ends_with("sw19571001.txt"));
    }
}
