/*!
 * Internal testing helper functions
 */

use std::env;
use std::path::Path;

use crate::eop::*;
use crate::orbit_dynamics::gravity::{DefaultGravityModel, GravityModel, set_global_gravity_model};

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

pub fn setup_global_test_gravity_model() {
    let gravity_model = GravityModel::from_default(DefaultGravityModel::EGM2008_360);
    set_global_gravity_model(gravity_model);
}

#[cfg(test)]
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
