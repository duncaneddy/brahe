//! Configure a force model with an ICGEM-sourced spherical-harmonic gravity field.
//!
//! `GravityModelType::ICGEMModel { body, name }` slots into the same
//! `GravityModelSource::ModelType(...)` slot as the packaged JGM3 / GGM05S /
//! EGM2008_120 variants. The .gfc file is downloaded into `$BRAHE_CACHE/icgem/`
//! on first use of the resulting `GravityModel`.
//!
//! FLAGS = ["NETWORK"]

use bh::GravityModelType;
use bh::datasets::icgem::ICGEMBody;
use brahe as bh;

fn main() {
    // Reference an ICGEM Earth model. Use
    // `bh::datasets::icgem::list_icgem_models(ICGEMBody::Earth)` to discover
    // the full catalog. Append "-<DEGREE>" to pin a specific variant.
    let grav_type = GravityModelType::ICGEMModel {
        body: ICGEMBody::Earth,
        name: "JGM3".to_string(),
    };

    let _force_config = bh::ForceModelConfig {
        central_body: bh::CentralBody::Earth,
        gravity: bh::GravityConfiguration::SphericalHarmonic {
            source: bh::GravityModelSource::ModelType(grav_type),
            degree: 20,
            order: 20,
            parallel: bh::orbit_dynamics::ParallelMode::Auto,
        },
        drag: None,
        srp: None,
        third_body: None,
        relativity: false,
        mass: None,
        frame_transform: bh::FrameTransformationModel::default(),
        tides: None,
    };

    println!("Built ForceModelConfig with ICGEM gravity source: Earth/JGM3 (20x20)");
}
