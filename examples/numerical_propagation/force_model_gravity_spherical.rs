//! Configuring spherical harmonic gravity with different model sources.
//! Shows packaged models (EGM2008, GGM05S, JGM3) and custom file loading.

use brahe as bh;
use bh::GravityModelType;

fn main() {
    // ==========================================================================
    // Packaged Gravity Models
    // ==========================================================================

    // EGM2008 - High-fidelity NGA model (360x360 max)
    let _gravity_egm2008 = bh::GravityConfiguration::SphericalHarmonic {
        source: bh::GravityModelSource::ModelType(GravityModelType::EGM2008_360),
        degree: 20,
        order: 20,
    };

    // GGM05S - GRACE mission model (180x180 max)
    let _gravity_ggm05s = bh::GravityConfiguration::SphericalHarmonic {
        source: bh::GravityModelSource::ModelType(GravityModelType::GGM05S),
        degree: 20,
        order: 20,
    };

    // JGM3 - Legacy model, fast computation (70x70 max)
    let _gravity_jgm3 = bh::GravityConfiguration::SphericalHarmonic {
        source: bh::GravityModelSource::ModelType(GravityModelType::JGM3),
        degree: 20,
        order: 20,
    };

    // ==========================================================================
    // Custom Model from File
    // ==========================================================================

    // Load custom gravity model from GFC format file
    // GravityModelType::from_file validates the path exists
    let custom_model_type =
        GravityModelType::from_file("data/gravity_models/EGM2008_360.gfc").unwrap();
    let _gravity_custom = bh::GravityConfiguration::SphericalHarmonic {
        source: bh::GravityModelSource::ModelType(custom_model_type),
        degree: 20,
        order: 20,
    };
}
