//! Tidal corrections to the geopotential: solid Earth tides, the solid Earth
//! and ocean pole tides, and FES2004 ocean tides (30x30, admittance-complete).
//! Propagates one LEO orbit with tides ON and OFF, and reports the peak
//! position difference to show the tidal perturbation magnitude.
//!
//! FLAGS = ["NETWORK"]
//!
//! Enabling ocean tides downloads a one-time IERS coefficient file (~3.7 MB)
//! into `$BRAHE_CACHE/tides/` the first time a propagator with ocean tides
//! enabled is constructed.

use bh::traits::DStatePropagator;
use brahe as bh;
use nalgebra as na;
use std::f64::consts::PI;

fn make_propagator(
    epoch: bh::Epoch,
    state: na::DVector<f64>,
    tides: Option<bh::TidesConfiguration>,
) -> bh::DNumericalOrbitPropagator {
    // Gravity-only: no drag/SRP so both propagators are directly comparable.
    let mut force_config = bh::ForceModelConfig::earth_gravity();
    // Add Sun+Moon third-body perturbations. Note: tidal accelerations use their
    // own internal low-precision ephemeris and do NOT depend on this setting.
    force_config.third_body = Some(vec![bh::ThirdBody::Sun.into(), bh::ThirdBody::Moon.into()]);
    force_config.tides = tides;

    bh::DNumericalOrbitPropagator::new(
        epoch,
        state,
        bh::NumericalPropagationConfig::default(),
        force_config,
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

fn main() {
    // EOP is required for the ITRF frame transformations used inside the tidal model.
    bh::initialize_eop().unwrap();
    bh::initialize_sw().unwrap();

    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // LEO satellite: 500 km altitude, slightly elliptical, sun-synchronous inclination.
    let oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 500e3, // semi-major axis (m)
        0.001,               // eccentricity
        97.8,                // inclination (deg)
        15.0,                // RAAN (deg)
        30.0,                // argument of perigee (deg)
        45.0,                // mean anomaly (deg)
    );
    let state0 = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);
    let state_dv = na::DVector::from_column_slice(state0.as_slice());

    // Propagate one full orbital period with and without solid Earth tides.
    let sma = bh::R_EARTH + 500e3;
    let period = 2.0 * PI * (sma.powi(3) / bh::GM_EARTH).sqrt();
    let n_steps = 90; // check every ~minute
    let dt = period / n_steps as f64;

    // Tides-ON configuration: IERS Step 1 + Step 2 (frequency-dependent) solid
    // Earth tides plus the solid Earth pole tide, FES2004 ocean tides (30x30,
    // admittance-completed) plus the ocean pole tide, and Auto permanent-tide
    // handling (converts the model's C̄20 to conventional tide-free).
    let tides_on = bh::TidesConfiguration {
        ephemeris_source: bh::EphemerisSource::LowPrecision,
        permanent: bh::PermanentTideConfig::Auto,
        solid: Some(bh::SolidTideConfig {
            frequency_dependent: true,
            pole_tide: true,
        }),
        ocean: Some(bh::OceanTideConfig {
            degree: 30,
            order: 30,
            include_admittance: true,
            pole_tide: true,
        }),
    };

    let mut prop_on = make_propagator(epoch, state_dv.clone(), Some(tides_on));
    let mut prop_off = make_propagator(epoch, state_dv.clone(), None);

    let mut max_diff_m: f64 = 0.0;
    let mut t = epoch;
    for _ in 0..n_steps {
        t = t + dt;
        prop_on.propagate_to(t).unwrap();
        prop_off.propagate_to(t).unwrap();

        let pos_on = prop_on.current_state().fixed_rows::<3>(0).into_owned();
        let pos_off = prop_off.current_state().fixed_rows::<3>(0).into_owned();
        let diff = (pos_on - pos_off).norm();
        if diff > max_diff_m {
            max_diff_m = diff;
        }
    }

    println!("Tidal corrections example");
    println!("  Orbital period:               {:.1} min", period / 60.0);
    println!(
        "  Max tidal position difference: {:.3} m  ({:.3} km)",
        max_diff_m,
        max_diff_m / 1e3
    );
    assert!(
        max_diff_m > 0.0 && max_diff_m < 1000.0,
        "Unexpected tidal difference: {max_diff_m} m"
    );
    println!("Example validated successfully!");
}
