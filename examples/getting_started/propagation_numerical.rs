#![allow(unused_imports)]
use brahe as bh;
use bh::traits::DStatePropagator;
use nalgebra as na;

fn main() {
    // Initialize EOP and space weather data (required for NRLMSISE-00 drag model)
    bh::initialize_eop().unwrap();
    bh::initialize_sw().unwrap();

    // Create initial epoch
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Define orbital elements: [a, e, i, raan, argp, M] in SI units
    // LEO satellite: 500 km altitude, near-circular, sun-synchronous inclination
    let oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 500e3,
        0.001,
        97.8,
        15.0,
        30.0,
        45.0,
    );
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // Parameters: [mass, drag_area, Cd, srp_area, Cr]
    let params = na::DVector::from_vec(vec![500.0, 2.0, 2.2, 2.0, 1.3]);

    // Create propagator with default configuration
    let mut prop = bh::DNumericalOrbitPropagator::builder(
        epoch,
        na::DVector::from_column_slice(state.as_slice()),
        bh::ForceModelConfig::default(),
    )
    .params(params)
    .build()
    .unwrap();

    // Propagate for 1 hour
    prop.propagate_to(epoch + 3600.0).unwrap();

    // Get final state
    let final_epoch = prop.current_epoch();
    let final_state = prop.current_state();

    println!("Initial epoch: {}", epoch);
    println!("Final epoch:   {}", final_epoch);
    println!(
        "Position (km): [{:.3}, {:.3}, {:.3}]",
        final_state[0] / 1e3,
        final_state[1] / 1e3,
        final_state[2] / 1e3
    );
    println!(
        "Velocity (m/s): [{:.3}, {:.3}, {:.3}]",
        final_state[3], final_state[4], final_state[5]
    );
}