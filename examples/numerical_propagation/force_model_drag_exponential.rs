//! Configuring atmospheric drag with exponential model.
//! Simple analytical model for quick estimates.

use brahe as bh;
use bh::GravityModelType;

fn main() {

    let drag_config = bh::DragConfiguration {
        model: bh::AtmosphericModel::Exponential {
            scale_height: 53000.0, // Scale height H in meters (53 km for ~300 km altitude)
            rho0: 1.225e-11,       // Reference density at h0 in kg/m^3
            h0: 300000.0,          // Reference altitude in meters (300 km)
        },
        area: bh::ParameterSource::ParameterIndex(1),
        cd: bh::ParameterSource::ParameterIndex(2),
    };

    // Create force model with exponential drag
    let _force_config = bh::ForceModelConfig {
        gravity: bh::GravityConfiguration::SphericalHarmonic {
            source: bh::GravityModelSource::ModelType(GravityModelType::EGM2008_360),
            degree: 20,
            order: 20,
        },
        drag: Some(drag_config),
        srp: None,
        third_body: None,
        relativity: false,
        mass: Some(bh::ParameterSource::ParameterIndex(0)),
    };
}
