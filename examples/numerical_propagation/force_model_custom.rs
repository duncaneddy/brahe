//! Building a fully custom force model configuration from scratch.
//! Shows how to combine different components for specific mission requirements.

use brahe as bh;
use bh::GravityModelType;

fn main() {
    // Example: Custom configuration for a LEO Earth observation satellite
    // Requirements:
    // - High-precision gravity (mission involves precise pointing)
    // - Atmospheric drag (significant at 400 km)
    // - No SRP (not critical for this mission)
    // - Sun/Moon third-body (long-term accuracy)
    // - Fixed spacecraft properties (well-characterized)

    // Configure high-resolution gravity
    let gravity_config = bh::GravityConfiguration::SphericalHarmonic {
        source: bh::GravityModelSource::ModelType(GravityModelType::EGM2008_360),
        degree: 36, // Higher than default for precise positioning
        order: 36,
    };

    // Configure NRLMSISE-00 drag (best for LEO precision)
    let drag_config = bh::DragConfiguration {
        model: bh::AtmosphericModel::NRLMSISE00,
        area: bh::ParameterSource::Value(5.0), // 5 m^2 cross-section
        cd: bh::ParameterSource::Value(2.3),   // Measured Cd
    };

    // Configure third-body with DE440s (high precision)
    let third_body_config = bh::ThirdBodyConfiguration {
        ephemeris_source: bh::EphemerisSource::DE440s,
        bodies: vec![bh::ThirdBody::Sun, bh::ThirdBody::Moon],
    };

    // Assemble the complete force model
    let custom_force_config = bh::ForceModelConfiguration {
        gravity: gravity_config,
        drag: Some(drag_config),
        srp: None, // SRP disabled for this mission
        third_body: Some(third_body_config),
        relativity: false, // Not needed for km-level accuracy
        mass: Some(bh::ParameterSource::Value(800.0)), // 800 kg spacecraft
    };

    println!("Custom Force Model Configuration:");
    println!("\nGravity:");
    println!("  Model: EGM2008 36x36 spherical harmonics");
    println!("  Purpose: High-precision positioning");

    println!("\nAtmospheric Drag:");
    println!("  Model: NRLMSISE-00");
    println!("  Area: 5.0 m^2 (fixed)");
    println!("  Cd: 2.3 (fixed)");

    println!("\nThird-Body:");
    println!("  Ephemeris: DE440s (high precision)");
    println!("  Bodies: Sun, Moon");

    println!("\nOther Settings:");
    println!("  SRP: Disabled");
    println!("  Relativity: Disabled");
    println!("  Mass: 800.0 kg (fixed)");

    println!(
        "\nRequires parameter vector: {}",
        custom_force_config.requires_params()
    );

    // Example 2: Mixed fixed and variable parameters
    println!("\n{}", "=".repeat(50));
    println!("Example 2: Mixed Fixed and Variable Parameters");
    println!("{}", "=".repeat(50));

    // Some parameters fixed, others from parameter vector
    let mixed_drag = bh::DragConfiguration {
        model: bh::AtmosphericModel::HarrisPriester,
        area: bh::ParameterSource::ParameterIndex(0), // Variable area (being estimated)
        cd: bh::ParameterSource::Value(2.2),          // Fixed Cd (well-known)
    };

    let mixed_srp = bh::SolarRadiationPressureConfiguration {
        area: bh::ParameterSource::ParameterIndex(0), // Same area as drag (realistic)
        cr: bh::ParameterSource::Value(1.4),          // Fixed Cr
        eclipse_model: bh::EclipseModel::Conical,
    };

    let mixed_force_config = bh::ForceModelConfiguration {
        gravity: bh::GravityConfiguration::SphericalHarmonic {
            source: bh::GravityModelSource::ModelType(GravityModelType::EGM2008_360),
            degree: 20,
            order: 20,
        },
        drag: Some(mixed_drag),
        srp: Some(mixed_srp),
        third_body: None,
        relativity: false,
        mass: Some(bh::ParameterSource::Value(500.0)), // Fixed mass
    };

    println!("\nMixed parameter configuration:");
    println!("  Fixed: mass=500kg, Cd=2.2, Cr=1.4");
    println!("  Variable: area (params[0]) - shared by drag and SRP");
    println!(
        "  Requires parameter vector: {}",
        mixed_force_config.requires_params()
    );

    // Validate
    assert!(!custom_force_config.requires_params()); // All fixed
    assert!(mixed_force_config.requires_params()); // Has variable params

    println!("\nExample validated successfully!");
}
