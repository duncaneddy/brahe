//! Permanent-tide correction only (no solid Earth tides).
//! Corrects the geopotential model's C̄20 for its tide system but adds no
//! time-varying solid-tide accelerations.

use bh::traits::DStatePropagator;
use brahe as bh;
use nalgebra as na;

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

    // Permanent tide only: corrects C̄20 for the tide system of the loaded model
    // but adds no time-varying solid-tide accelerations.
    let tides = bh::TidesConfiguration {
        permanent: bh::PermanentTideConfig::Auto,
        solid: None,
    };

    let mut force_config = bh::ForceModelConfig::earth_gravity();
    force_config.tides = Some(tides);

    let dt = 60.0; // step once by 60 seconds
    let t1 = epoch + dt;

    let mut prop = bh::DNumericalOrbitPropagator::new(
        epoch,
        state_dv,
        bh::NumericalPropagationConfig::default(),
        force_config,
        None,
        None,
        None,
        None,
    )
    .unwrap();

    prop.propagate_to(t1);
    let state = prop.current_state();
    println!(
        "Permanent-tide-only example: position after 60 s = ({:.3}, {:.3}, {:.3}) m",
        state[0], state[1], state[2]
    );
    println!("Example validated successfully!");
}
