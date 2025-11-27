//! Using ParameterSource::ParameterIndex for parameter vector values.
//! Parameters that can be varied or estimated during propagation.

use brahe as bh;
use bh::GravityModelType;

fn main() {
    // ParameterSource::ParameterIndex references a value in the parameter vector
    // Use when parameters may change or need to be estimated

    // Default parameter layout:
    // Index 0: mass (kg)
    // Index 1: drag_area (m^2)
    // Index 2: Cd (dimensionless)
    // Index 3: srp_area (m^2)
    // Index 4: Cr (dimensionless)

    let drag_config = bh::DragConfiguration {
        model: bh::AtmosphericModel::HarrisPriester,
        area: bh::ParameterSource::ParameterIndex(1), // params[1] = drag_area
        cd: bh::ParameterSource::ParameterIndex(2),   // params[2] = Cd
    };

    let srp_config = bh::SolarRadiationPressureConfiguration {
        area: bh::ParameterSource::ParameterIndex(3), // params[3] = srp_area
        cr: bh::ParameterSource::ParameterIndex(4),   // params[4] = Cr
        eclipse_model: bh::EclipseModel::Conical,
    };

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
        mass: Some(bh::ParameterSource::ParameterIndex(0)), // params[0] = mass
    };

    println!("Using ParameterSource::ParameterIndex for Variable Parameters:");
    println!("\nDefault parameter layout:");
    println!("  params[0] = mass (kg)");
    println!("  params[1] = drag_area (m^2)");
    println!("  params[2] = Cd (dimensionless)");
    println!("  params[3] = srp_area (m^2)");
    println!("  params[4] = Cr (dimensionless)");

    println!(
        "\nRequires parameter vector: {}",
        force_config.requires_params()
    );

    // Example parameter vector
    println!("\nExample parameter vector:");
    println!("  [500.0, 10.0, 2.2, 10.0, 1.3]");
    println!("  mass=500kg, drag_area=10m^2, Cd=2.2, srp_area=10m^2, Cr=1.3");

    // Custom parameter layout example
    println!("\nCustom parameter layout example:");
    println!("  You can map parameters to any indices:");
    let _custom_drag = bh::DragConfiguration {
        model: bh::AtmosphericModel::HarrisPriester,
        area: bh::ParameterSource::ParameterIndex(5),  // Custom index
        cd: bh::ParameterSource::ParameterIndex(10),   // Custom index
    };
    println!("  Drag area from params[5], Cd from params[10]");

    println!("\nWhen to use ParameterSource::ParameterIndex:");
    println!("  - Parameters may be estimated (orbit determination)");
    println!("  - Running Monte Carlo or batch studies");
    println!("  - Sensitivity analysis");
    println!("  - Dynamic parameter updates during simulation");

    // Validate - indexed parameters require parameter vector
    assert!(force_config.requires_params());

    println!("\nExample validated successfully!");
}
