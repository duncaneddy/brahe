/*!
 * Internal testing helper functions and fixtures.
 *
 * This module is only compiled for tests (#[cfg(test)] in mod.rs).
 */

use std::env;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use anise::prelude::{Almanac, SPK};
use nalgebra::SVector;

use crate::AngleFormat;
use crate::constants::DEG2RAD;
use crate::constants::physical::R_EARTH;
use crate::eop::*;
use crate::math::angles::oe_to_radians;
use crate::orbit_dynamics::ephemerides::set_global_almanac;
use crate::orbit_dynamics::gravity::{DefaultGravityModel, GravityModel, set_global_gravity_model};
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

/// Initialize global gravity model with EGM2008_360 for orbit dynamics tests.
///
/// Sets up EGM2008 (degree/order 360) as the global gravity model. Use at the start
/// of tests requiring high-fidelity gravity for orbit propagation or perturbation analysis.
pub(crate) fn setup_global_test_gravity_model() {
    let gravity_model = GravityModel::from_default(DefaultGravityModel::EGM2008_360);
    set_global_gravity_model(gravity_model);
}

/// Initialize global ANISE Almanac with DE440s kernel for ephemeris tests.
///
/// Copies `test_assets/de440s.bsp` to the NAIF cache directory and loads it as the
/// global Almanac. This avoids network downloads during CI tests while providing
/// the same DE440s ephemeris data for testing high-precision sun/moon positions.
///
/// Use at the start of tests requiring DE440s ephemeris (`sun_position_de440s()`,
/// `moon_position_de440s()`). If the test asset doesn't exist, this function does
/// nothing (allows running tests without the large kernel file).
///
/// # Panics
/// Panics if the test asset exists but cannot be copied or loaded.
pub(crate) fn setup_global_test_almanac() {
    // Use OnceLock to ensure single initialization across parallel tests
    static ALMANAC_INIT: OnceLock<()> = OnceLock::new();

    ALMANAC_INIT.get_or_init(|| {
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let test_asset_path = Path::new(&manifest_dir)
            .join("test_assets")
            .join("de440s.bsp");

        // Only proceed if test asset exists (it might not in local dev)
        if !test_asset_path.exists() {
            return;
        }

        let cache_dir = get_naif_cache_dir().expect("Failed to get NAIF cache dir");
        let cache_path = Path::new(&cache_dir).join("de440s.bsp");

        // Copy test asset to cache (overwrite if exists to ensure we have latest)
        fs::copy(&test_asset_path, &cache_path).expect("Failed to copy test asset to cache");

        // Load SPK and create Almanac context
        let cache_path_str = cache_path
            .to_str()
            .expect("Failed to convert path to string");
        let spk = SPK::load(cache_path_str).expect("Failed to load DE440s test kernel");
        let almanac = Almanac::from_spk(spk);

        // Set as global
        set_global_almanac(almanac);
    });
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
