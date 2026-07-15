//! Configuring third-body perturbations with different ephemeris sources.
//! Shows how to include Sun, Moon, and planetary gravitational effects.

use bh::GravityModelType;
use brahe as bh;

fn main() {
    // Third-body perturbations configuration: one entry per perturbing body,
    // each carrying its own ephemeris source and gravity model (point-mass
    // by default).

    // Option 1: Low-precision analytical ephemerides
    // Fast but less accurate (~km level errors for Sun/Moon)
    // Only Sun and Moon are available
    let _third_bodies_low = vec![
        bh::ThirdBodyConfiguration {
            ephemeris_source: bh::EphemerisSource::LowPrecision,
            ..bh::ThirdBody::Sun.into()
        },
        bh::ThirdBodyConfiguration {
            ephemeris_source: bh::EphemerisSource::LowPrecision,
            ..bh::ThirdBody::Moon.into()
        },
    ];

    // Option 2: DE440s high-precision ephemerides (recommended)
    // Uses JPL Development Ephemeris 440 (small bodies version)
    // ~m level accuracy, valid 1550-2650 CE
    // All planets available, ~17 MB file. DE440s is the default source, so
    // bare bodies convert directly into point-mass entries.
    let third_bodies_de440s: Vec<bh::ThirdBodyConfiguration> =
        vec![bh::ThirdBody::Sun.into(), bh::ThirdBody::Moon.into()];

    // Option 3: DE440 full-precision ephemerides
    // Highest accuracy (~mm level), valid 13200 BCE-17191 CE
    // All planets available, ~114 MB file
    let _third_bodies_de440 = vec![
        bh::ThirdBodyConfiguration {
            ephemeris_source: bh::EphemerisSource::DE440,
            ..bh::ThirdBody::Sun.into()
        },
        bh::ThirdBodyConfiguration {
            ephemeris_source: bh::EphemerisSource::DE440,
            ..bh::ThirdBody::Moon.into()
        },
    ];

    // Option 4: Include all major planets (high-fidelity). The *Barycenter
    // variants use the planetary-system barycenters with system GMs — the
    // classical third-body formulation, resolvable from the DE kernel alone.
    let _third_bodies_all_planets: Vec<bh::ThirdBodyConfiguration> = vec![
        bh::ThirdBody::Sun.into(),
        bh::ThirdBody::Moon.into(),
        bh::ThirdBody::Mercury.into(),
        bh::ThirdBody::Venus.into(),
        bh::ThirdBody::MarsBarycenter.into(),
        bh::ThirdBody::JupiterBarycenter.into(),
        bh::ThirdBody::SaturnBarycenter.into(),
        bh::ThirdBody::UranusBarycenter.into(),
        bh::ThirdBody::NeptuneBarycenter.into(),
    ];

    // Create force model with Sun/Moon perturbations (common case)
    let _force_config = bh::ForceModelConfig {
        central_body: bh::CentralBody::Earth,
        gravity: bh::GravityConfiguration::SphericalHarmonic {
            source: bh::GravityModelSource::ModelType(GravityModelType::EGM2008_360),
            degree: 20,
            order: 20,
            parallel: bh::orbit_dynamics::ParallelMode::Auto,
        },
        drag: None,
        srp: None,
        third_bodies: Some(third_bodies_de440s),
        relativity: false,
        mass: None,
        frame_transform: bh::FrameTransformationModel::default(),
        tides: None,
    };
}
