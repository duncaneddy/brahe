//! Configuring point mass gravity for the force model.
//! Simplest gravity configuration with no perturbations.

use brahe as bh;

fn main() {
    // Point mass gravity configuration
    // Uses only central body gravity (mu/r^2)
    // No spherical harmonics, J2, or higher-order terms
    let force_config = bh::ForceModelConfiguration {
        gravity: bh::GravityConfiguration::PointMass,
        drag: None,
        srp: None,
        third_body: None,
        relativity: false,
        mass: None,
    };

    // Check configuration
    println!("Point Mass Gravity Configuration:");
    println!("  Gravity type: PointMass");
    println!("  Requires params: {}", force_config.requires_params());
    println!("  Drag enabled: {}", force_config.drag.is_some());
    println!("  SRP enabled: {}", force_config.srp.is_some());
    println!("  Third-body enabled: {}", force_config.third_body.is_some());
    println!("  Relativity enabled: {}", force_config.relativity);

    // This is equivalent to using the two_body_gravity() preset
    let preset_config = bh::ForceModelConfiguration::two_body_gravity();
    println!("\nEquivalent preset: ForceModelConfiguration::two_body_gravity()");
    println!("  Preset requires params: {}", preset_config.requires_params());

    // Validate - point mass should not require parameters
    assert!(!force_config.requires_params());
    assert!(!preset_config.requires_params());

    println!("\nExample validated successfully!");
}
