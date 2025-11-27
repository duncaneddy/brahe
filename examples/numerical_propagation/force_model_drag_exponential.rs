//! Configuring atmospheric drag with exponential model.
//! Simple analytical model for quick estimates.

use brahe as bh;
use bh::GravityModelType;

fn main() {
    // Exponential atmosphere model
    // Density varies as: rho(h) = rho0 * exp(-(h - h0) / H)
    // - Very fast computation
    // - Good for rough estimates and educational purposes
    // - Does not account for latitude, solar activity, or time variations

    // Typical scale heights for different altitude regimes:
    // 150 km: ~22 km
    // 200 km: ~37 km
    // 300 km: ~53 km
    // 400 km: ~59 km
    // 500 km: ~70 km

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
    let force_config = bh::ForceModelConfiguration {
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

    println!("Exponential Atmosphere Model:");
    println!("  Formula: rho(h) = rho0 * exp(-(h - h0) / H)");
    println!("  Scale height (H): 53,000 m");
    println!("  Reference density (rho0): 1.225e-11 kg/m^3");
    println!("  Reference altitude (h0): 300,000 m");
    println!("  Requires params: {}", force_config.requires_params());

    println!("\nTypical scale heights by altitude:");
    println!("  150 km: ~22 km");
    println!("  200 km: ~37 km");
    println!("  300 km: ~53 km");
    println!("  400 km: ~59 km");
    println!("  500 km: ~70 km");

    println!("\nWhen to use exponential model:");
    println!("  - Quick analytical estimates");
    println!("  - Educational/teaching purposes");
    println!("  - When computation speed is critical");
    println!("  - Not recommended for precision applications");

    // Validate - drag config requires parameters
    assert!(force_config.requires_params());

    println!("\nExample validated successfully!");
}
