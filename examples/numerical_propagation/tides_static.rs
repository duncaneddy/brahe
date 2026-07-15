//! Force model configuration with permanent tide + the static solid Earth tide.

use brahe as bh;

fn main() {
    // Permanent tide + the static (frequency-independent) solid Earth tide correction.
    // `frequency_dependent: false` keeps only the always-on static part.
    let tides = bh::TidesConfiguration {
        permanent: bh::PermanentTideConfig::Auto,
        solid: Some(bh::SolidTideConfig {
            frequency_dependent: false,
            pole_tide: false,
        }),
        ocean: None,
    };

    let mut force_config = bh::ForceModelConfig::earth_gravity();
    force_config.tides = Some(tides);

    // `force_config` is now ready to hand to a NumericalOrbitPropagator.
    assert!(force_config.tides.is_some());
}
