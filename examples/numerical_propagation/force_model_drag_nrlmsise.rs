//! Configuring atmospheric drag with NRLMSISE-00 model.
//! High-fidelity atmospheric model for precision applications.

use brahe as bh;
use bh::GravityModelType;

fn main() {
    // NRLMSISE-00 atmospheric drag configuration
    // - Naval Research Laboratory Mass Spectrometer and Incoherent Scatter Radar
    // - High-fidelity empirical model
    // - Valid from ground to thermospheric heights
    // - Uses space weather data (F10.7, Ap) when available
    // - More computationally expensive than Harris-Priester

    let drag_config = bh::DragConfiguration {
        model: bh::AtmosphericModel::NRLMSISE00,
        area: bh::ParameterSource::ParameterIndex(1), // drag_area from params[1]
        cd: bh::ParameterSource::ParameterIndex(2),   // Cd from params[2]
    };

    // Create force model with NRLMSISE-00 drag
    let force_config = bh::ForceModelConfiguration {
        gravity: bh::GravityConfiguration::SphericalHarmonic {
            source: bh::GravityModelSource::ModelType(GravityModelType::EGM2008_360),
            degree: 30,
            order: 30,
        },
        drag: Some(drag_config),
        srp: None,
        third_body: None,
        relativity: false,
        mass: Some(bh::ParameterSource::ParameterIndex(0)),
    };

    println!("NRLMSISE-00 Drag Configuration:");
    println!("  Atmospheric model: NRLMSISE-00");
    println!("  Valid altitude range: Ground to thermosphere");
    println!("  Space weather data: Uses F10.7 and Ap indices");
    println!("  Accuracy: Higher fidelity than Harris-Priester");
    println!("  Requires params: {}", force_config.requires_params());

    println!("\nWhen to use NRLMSISE-00:");
    println!("  - Precision orbit determination");
    println!("  - Low Earth Orbit operations");
    println!("  - Atmospheric research applications");
    println!("  - When space weather effects are important");

    println!("\nComparison with Harris-Priester:");
    println!("  Harris-Priester: Fast, no space weather, 100-1000 km");
    println!("  NRLMSISE-00: Slower, uses space weather, all altitudes");

    // Validate - drag config requires parameters
    assert!(force_config.requires_params());

    println!("\nExample validated successfully!");
}
