//! Basic numerical orbit propagation using NumericalOrbitPropagator.
//! Demonstrates creating a propagator and propagating to a target time.

use brahe as bh;
use bh::traits::DStatePropagator;
use nalgebra as na;

fn main() {
    // Initialize EOP data
    bh::initialize_eop().unwrap();

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
    let mut prop = bh::DNumericalOrbitPropagator::new(
        epoch,
        na::DVector::from_column_slice(state.as_slice()),
        bh::NumericalPropagationConfig::default(),
        bh::ForceModelConfiguration::default(),
        Some(params),
        None,  // No additional dynamics
        None,  // No control input
        None,  // No initial covariance
    )
    .unwrap();

    // Propagate for 1 hour
    prop.propagate_to(epoch + 3600.0);

    // Get final state
    let final_epoch = prop.current_epoch();
    let final_state = prop.current_state();

    // Validate propagation completed
    assert_eq!(final_epoch, epoch + 3600.0);
    assert_eq!(final_state.len(), 6);
    assert!(final_state.fixed_rows::<3>(0).norm() > bh::R_EARTH);

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
    println!("Example validated successfully!");
}
