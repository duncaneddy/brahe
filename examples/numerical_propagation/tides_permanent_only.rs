//! Force model configuration with the permanent-tide correction only.

use brahe as bh;

fn main() {
    // Permanent-tide correction only: normalize the loaded model's C̄20 to the
    // conventional tide-free system, but add no time-varying solid-tide accelerations.
    let tides = bh::TidesConfiguration {
        ephemeris_source: bh::EphemerisSource::LowPrecision,
        permanent: bh::PermanentTideConfig::Auto,
        solid: None,
        ocean: None,
    };

    let mut force_config = bh::ForceModelConfig::earth_gravity();
    force_config.tides = Some(tides);

    // `force_config` is now ready to hand to a NumericalOrbitPropagator.
    assert!(force_config.tides.is_some());
}
