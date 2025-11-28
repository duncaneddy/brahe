//! Configuring third-body perturbations with different ephemeris sources.
//! Shows how to include Sun, Moon, and planetary gravitational effects.

use brahe as bh;
use bh::GravityModelType;

fn main() {
    // Third-body perturbations configuration
    // Gravitational attraction from other celestial bodies

    // Option 1: Low-precision analytical ephemerides
    // Fast but less accurate (~km level errors for Sun/Moon)
    // Only Sun and Moon are available
    let _third_body_low = bh::ThirdBodyConfiguration {
        ephemeris_source: bh::EphemerisSource::LowPrecision,
        bodies: vec![bh::ThirdBody::Sun, bh::ThirdBody::Moon],
    };

    // Option 2: DE440s high-precision ephemerides (recommended)
    // Uses JPL Development Ephemeris 440 (small bodies version)
    // ~m level accuracy, valid 1550-2650 CE
    // All planets available, ~17 MB file
    let third_body_de440s = bh::ThirdBodyConfiguration {
        ephemeris_source: bh::EphemerisSource::DE440s,
        bodies: vec![bh::ThirdBody::Sun, bh::ThirdBody::Moon],
    };

    // Option 3: DE440 full-precision ephemerides
    // Highest accuracy (~mm level), valid 13200 BCE-17191 CE
    // All planets available, ~114 MB file
    let _third_body_de440 = bh::ThirdBodyConfiguration {
        ephemeris_source: bh::EphemerisSource::DE440,
        bodies: vec![bh::ThirdBody::Sun, bh::ThirdBody::Moon],
    };

    // Option 4: Include all major planets (high-fidelity)
    let _third_body_all_planets = bh::ThirdBodyConfiguration {
        ephemeris_source: bh::EphemerisSource::DE440s,
        bodies: vec![
            bh::ThirdBody::Sun,
            bh::ThirdBody::Moon,
            bh::ThirdBody::Mercury,
            bh::ThirdBody::Venus,
            bh::ThirdBody::Mars,
            bh::ThirdBody::Jupiter,
            bh::ThirdBody::Saturn,
            bh::ThirdBody::Uranus,
            bh::ThirdBody::Neptune,
        ],
    };

    // Create force model with Sun/Moon perturbations (common case)
    let _force_config = bh::ForceModelConfig {
        gravity: bh::GravityConfiguration::SphericalHarmonic {
            source: bh::GravityModelSource::ModelType(GravityModelType::EGM2008_360),
            degree: 20,
            order: 20,
        },
        drag: None,
        srp: None,
        third_body: Some(third_body_de440s),
        relativity: false,
        mass: None,
    };
}
