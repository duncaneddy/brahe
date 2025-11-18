/*!
 * Internal testing helper functions
 */

use std::env;
use std::fs;
use std::path::Path;

use crate::eop::*;
use crate::orbit_dynamics::ephemerides::set_global_almanac;
use crate::orbit_dynamics::gravity::{DefaultGravityModel, GravityModel, set_global_gravity_model};
use crate::utils::get_naif_cache_dir;
use anise::prelude::{Almanac, SPK};

/// Initialize global EOP provider with test data for unit testing.
///
/// Loads `test_assets/finals.all.iau2000.txt` and configures with Hold extrapolation
/// and linear interpolation. Use at the start of tests requiring EOP data for frame
/// transformations.
///
/// # Panics
/// Panics if test asset file cannot be found or loaded.
pub fn setup_global_test_eop() {
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
pub fn setup_global_test_eop_original_brahe() {
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
pub fn setup_global_test_gravity_model() {
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
pub fn setup_global_test_almanac() {
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

    // Copy test asset to cache if not already there
    if !cache_path.exists() {
        fs::copy(&test_asset_path, &cache_path).expect("Failed to copy test asset to cache");
    }

    // Load SPK and create Almanac context
    let cache_path_str = cache_path
        .to_str()
        .expect("Failed to convert path to string");
    let spk = SPK::load(cache_path_str).expect("Failed to load DE440s test kernel");
    let almanac = Almanac::from_spk(spk);

    // Set as global
    set_global_almanac(almanac);
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

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
}
