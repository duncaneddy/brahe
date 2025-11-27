//! Using ParameterSource::Value for fixed parameter values.
//! Parameters that don't change during propagation.

use brahe as bh;
use bh::GravityModelType;

fn main() {
    // ParameterSource::Value creates a fixed constant parameter
    // Use when the parameter doesn't change and doesn't need to be estimated

    // Example: Fixed drag configuration
    // Mass, drag area, and Cd are all constant
    let drag_config = bh::DragConfiguration {
        model: bh::AtmosphericModel::HarrisPriester,
        area: bh::ParameterSource::Value(10.0), // Fixed 10 m^2 drag area
        cd: bh::ParameterSource::Value(2.2),    // Fixed Cd of 2.2
    };

    // Example: Fixed SRP configuration
    let srp_config = bh::SolarRadiationPressureConfiguration {
        area: bh::ParameterSource::Value(15.0), // Fixed 15 m^2 SRP area
        cr: bh::ParameterSource::Value(1.3),    // Fixed Cr of 1.3
        eclipse_model: bh::EclipseModel::Conical,
    };

    // Create force model with all fixed parameters
    let force_config = bh::ForceModelConfiguration {
        gravity: bh::GravityConfiguration::SphericalHarmonic {
            source: bh::GravityModelSource::ModelType(GravityModelType::EGM2008_360),
            degree: 20,
            order: 20,
        },
        drag: Some(drag_config),
        srp: Some(srp_config),
        third_body: Some(bh::ThirdBodyConfiguration {
            ephemeris_source: bh::EphemerisSource::LowPrecision,
            bodies: vec![bh::ThirdBody::Sun, bh::ThirdBody::Moon],
        }),
        relativity: false,
        mass: Some(bh::ParameterSource::Value(500.0)), // Fixed 500 kg mass
    };

    println!("Using ParameterSource::Value for Fixed Parameters:");
    println!("  Mass: 500.0 kg (fixed)");
    println!("  Drag area: 10.0 m^2 (fixed)");
    println!("  Cd: 2.2 (fixed)");
    println!("  SRP area: 15.0 m^2 (fixed)");
    println!("  Cr: 1.3 (fixed)");

    // With all fixed values, no parameter vector is required
    println!(
        "\nRequires parameter vector: {}",
        force_config.requires_params()
    );

    println!("\nWhen to use ParameterSource::Value:");
    println!("  - Parameter is well-known and constant");
    println!("  - Not performing parameter estimation");
    println!("  - Simpler propagator setup (no params array needed)");

    println!("\nWhen to use ParameterSource::ParameterIndex instead:");
    println!("  - Parameter may be varied or estimated");
    println!("  - Running batch studies with different values");
    println!("  - Performing orbit determination");
    println!("  - Need to update parameters during simulation");

    // Validate - all fixed values means no params required
    assert!(!force_config.requires_params());

    println!("\nExample validated successfully!");
}
