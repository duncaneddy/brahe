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

    // This is equivalent to using the two_body_gravity() preset
    let preset_config = bh::ForceModelConfiguration::two_body_gravity();
    println!("\nEquivalent preset: ForceModelConfiguration::two_body_gravity()");
    println!("  Preset requires params: {}", preset_config.requires_params());
}
