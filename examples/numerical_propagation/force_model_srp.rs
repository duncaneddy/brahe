//! Configuring solar radiation pressure with different eclipse models.
//! Shows how to configure SRP for different accuracy requirements.

use brahe as bh;

fn main() {
    // Solar Radiation Pressure configuration
    // Parameters:
    // - area: Cross-sectional area facing the Sun (m^2)
    // - cr: Coefficient of reflectivity (1.0=absorbing to 2.0=perfectly reflecting)
    // - eclipse_model: How to handle Earth's shadow

    // Option 1: No eclipse model (always illuminated)
    // Fast but inaccurate during eclipse periods
    let _srp_no_eclipse = bh::SolarRadiationPressureConfiguration {
        area: bh::ParameterSource::ParameterIndex(3), // srp_area from params[3]
        cr: bh::ParameterSource::ParameterIndex(4),   // Cr from params[4]
        eclipse_model: bh::EclipseModel::None,
    };

    // Option 2: Cylindrical shadow model
    // Simple and fast, sharp shadow boundary (no penumbra)
    let _srp_cylindrical = bh::SolarRadiationPressureConfiguration {
        area: bh::ParameterSource::ParameterIndex(3),
        cr: bh::ParameterSource::ParameterIndex(4),
        eclipse_model: bh::EclipseModel::Cylindrical,
    };

    // Option 3: Conical shadow model (recommended)
    // Accounts for penumbra and umbra regions
    let _srp_conical = bh::SolarRadiationPressureConfiguration {
        area: bh::ParameterSource::ParameterIndex(3),
        cr: bh::ParameterSource::ParameterIndex(4),
        eclipse_model: bh::EclipseModel::Conical,
    };
}
