//! Configuring spherical harmonic gravity with custom degree and order.
//! Shows how to use different gravity model resolutions.

use brahe as bh;
use bh::GravityModelType;

fn main() {
    // Spherical harmonic gravity with custom degree/order
    // Higher degree/order = more accurate but slower
    // EGM2008 model supports up to degree 360

    // Low-resolution for fast computation (e.g., GEO)
    let gravity_low = bh::GravityConfiguration::SphericalHarmonic {
        source: bh::GravityModelSource::ModelType(GravityModelType::EGM2008_360),
        degree: 8,
        order: 8,
    };

    // Medium resolution for general use (e.g., default)
    let _gravity_medium = bh::GravityConfiguration::SphericalHarmonic {
        source: bh::GravityModelSource::ModelType(GravityModelType::EGM2008_360),
        degree: 20,
        order: 20,
    };

    // High resolution for precision applications (e.g., LEO)
    let gravity_high = bh::GravityConfiguration::SphericalHarmonic {
        source: bh::GravityModelSource::ModelType(GravityModelType::EGM2008_360),
        degree: 70,
        order: 70,
    };

    // Create force models with different gravity resolutions
    let force_config_low = bh::ForceModelConfiguration {
        gravity: gravity_low,
        drag: None,
        srp: None,
        third_body: None,
        relativity: false,
        mass: None,
    };

    let force_config_high = bh::ForceModelConfiguration {
        gravity: gravity_high,
        drag: None,
        srp: None,
        third_body: None,
        relativity: false,
        mass: None,
    };

    println!("Spherical Harmonic Gravity Configurations:");
    println!("\n  Low resolution (8x8 - GEO applications):");
    println!("    Requires params: {}", force_config_low.requires_params());

    println!("\n  High resolution (70x70 - precision LEO):");
    println!("    Requires params: {}", force_config_high.requires_params());

    println!("\nRecommended degree/order by orbit type:");
    println!("  - High altitude (GEO): 4x4 to 8x8");
    println!("  - General mission analysis: 20x20 to 36x36");
    println!("  - LEO precision: 20x20 to 36x36");
    println!("  - Geodesy/POD: 70x70+");

    // Validate - gravity-only configs don't require parameters
    assert!(!force_config_low.requires_params());
    assert!(!force_config_high.requires_params());

    println!("\nExample validated successfully!");
}
