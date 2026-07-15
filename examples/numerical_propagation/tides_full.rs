//! Force model configuration with the full tide model: solid Earth tides
//! (static + time-varying + pole tide) plus FES2004 ocean tides (admittance +
//! ocean pole tide). This mirrors the tide configuration used internally by
//! `ForceModelConfig::high_fidelity()`.

use brahe as bh;

fn main() {
    // Solid Earth tides: static + time-varying (frequency-dependent)
    // corrections, plus the solid Earth pole tide.
    let solid = bh::SolidTideConfig {
        frequency_dependent: true,
        pole_tide: true,
    };

    // Ocean tides: FES2004 to degree/order 30, admittance-completed, plus the
    // ocean pole tide. Building this configuration does not touch the
    // network; the FES2004 coefficient file (~3.7 MB) is downloaded once
    // into `$BRAHE_CACHE/tides/` the first time a propagator with ocean
    // tides enabled is constructed.
    let ocean = bh::OceanTideConfig {
        degree: 30,
        order: 30,
        include_admittance: true,
        pole_tide: true,
    };

    let tides = bh::TidesConfiguration {
        permanent: bh::PermanentTideConfig::Auto,
        solid: Some(solid),
        ocean: Some(ocean),
    };

    let mut force_config = bh::ForceModelConfig::earth_gravity();
    force_config.tides = Some(tides);

    // `force_config` is now ready to hand to a NumericalOrbitPropagator.
    assert!(force_config.tides.is_some());
    assert!(force_config.tides.unwrap().ocean.is_some());
}
