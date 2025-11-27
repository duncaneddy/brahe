//! Configuring atmospheric drag with Harris-Priester model.
//! Fast atmospheric model accounting for diurnal density variations.

use brahe as bh;
use bh::GravityModelType;

fn main() {
    // Harris-Priester atmospheric drag configuration
    // - Valid for altitudes 100-1000 km
    // - Accounts for latitude-dependent diurnal bulge
    // - Does not require space weather data (F10.7, Ap)
    // - Good balance of speed and accuracy

    // Using parameter indices (default layout)
    let drag_config = bh::DragConfiguration {
        model: bh::AtmosphericModel::HarrisPriester,
        area: bh::ParameterSource::ParameterIndex(1), // drag_area from params[1]
        cd: bh::ParameterSource::ParameterIndex(2),   // Cd from params[2]
    };

    // Create force model with Harris-Priester drag
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
        mass: Some(bh::ParameterSource::ParameterIndex(0)), // mass from params[0]
    };

    println!("Harris-Priester Drag Configuration:");
    println!("  Atmospheric model: Harris-Priester");
    println!("  Valid altitude range: 100-1000 km");
    println!("  Space weather required: No");
    println!("  Requires params: {}", force_config.requires_params());

    println!("\nParameter vector layout (default):");
    println!("  params[0] = mass (kg)");
    println!("  params[1] = drag_area (m^2)");
    println!("  params[2] = Cd (dimensionless, typically 2.0-2.5)");

    // Typical drag coefficient values:
    println!("\nTypical Cd values:");
    println!("  - Spherical satellite: 2.0-2.2");
    println!("  - Flat plate normal to flow: 2.2-2.4");
    println!("  - Complex spacecraft: 2.0-2.5");

    // Validate - drag config requires parameters
    assert!(force_config.requires_params());

    println!("\nExample validated successfully!");
}
