//! Force model configuration with permanent tide + static and time-varying solid Earth tides.

use brahe as bh;

fn main() {
    // Permanent tide + both the static and time-varying (frequency-dependent) parts
    // of the solid Earth tide correction. `frequency_dependent: true` adds the
    // tidal-line refinements (IERS Tables 6.5a/b/c) on top of the static correction.
    let tides = bh::TidesConfiguration {
        permanent: bh::PermanentTideConfig::Auto,
        solid: Some(bh::SolidTideConfig {
            frequency_dependent: true,
            pole_tide: false,
        }),
        ocean: None,
    };

    let mut force_config = bh::ForceModelConfig::earth_gravity();
    force_config.tides = Some(tides);

    // `force_config` is now ready to hand to a NumericalOrbitPropagator.
    assert!(force_config.tides.is_some());
}
