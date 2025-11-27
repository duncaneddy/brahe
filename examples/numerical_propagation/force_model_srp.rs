//! Configuring solar radiation pressure with different eclipse models.
//! Shows how to configure SRP for different accuracy requirements.

use brahe as bh;
use bh::GravityModelType;

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
    let srp_conical = bh::SolarRadiationPressureConfiguration {
        area: bh::ParameterSource::ParameterIndex(3),
        cr: bh::ParameterSource::ParameterIndex(4),
        eclipse_model: bh::EclipseModel::Conical,
    };

    // Create force model with conical SRP (most common)
    let force_config = bh::ForceModelConfiguration {
        gravity: bh::GravityConfiguration::SphericalHarmonic {
            source: bh::GravityModelSource::ModelType(GravityModelType::EGM2008_360),
            degree: 8,
            order: 8,
        },
        drag: None,
        srp: Some(srp_conical),
        third_body: None,
        relativity: false,
        mass: Some(bh::ParameterSource::ParameterIndex(0)),
    };

    println!("Solar Radiation Pressure Configuration:");
    println!("  Requires params: {}", force_config.requires_params());

    println!("\nEclipse Models:");
    println!("  None: SRP always applied, fast but inaccurate in shadow");
    println!("  Cylindrical: Sharp shadow boundary, simple and fast");
    println!("  Conical: Penumbra + umbra, most accurate (recommended)");

    println!("\nParameter vector layout (default):");
    println!("  params[3] = srp_area (m^2)");
    println!("  params[4] = Cr (dimensionless, 1.0-2.0)");

    println!("\nTypical Cr values:");
    println!("  - Absorbing surface (black body): 1.0");
    println!("  - Typical spacecraft: 1.2-1.5");
    println!("  - Highly reflective (solar sail): 1.8-2.0");

    println!("\nWhen SRP is significant:");
    println!("  - High altitude orbits (GEO, MEO)");
    println!("  - High area-to-mass ratio spacecraft");
    println!("  - Solar sails");
    println!("  - When drag is negligible");

    // Validate - SRP config requires parameters
    assert!(force_config.requires_params());

    println!("\nExample validated successfully!");
}
