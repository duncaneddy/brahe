//! Numerical propagation with trajectory history access.
//! Demonstrates accessing stored trajectory states and interpolation.

use brahe as bh;
use bh::traits::{DStatePropagator, DOrbitStateProvider, DStateProvider, Trajectory};
use nalgebra as na;

fn main() {
    // Initialize EOP data
    bh::initialize_eop().unwrap();

    // Create initial epoch
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Define orbital elements: LEO satellite
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

    // Create propagator
    let mut prop = bh::DNumericalOrbitPropagator::new(
        epoch,
        na::DVector::from_column_slice(state.as_slice()),
        bh::NumericalPropagationConfig::default(),
        bh::ForceModelConfiguration::default(),
        Some(params),
        None,
        None,
        None,
    )
    .unwrap();

    // Propagate for 2 hours
    prop.propagate_to(epoch + 7200.0);

    // Access trajectory
    let trajectory = prop.trajectory();

    // Get trajectory length (number of stored points)
    let num_points = trajectory.len();
    println!("Trajectory points stored: {}", num_points);

    // Get state at intermediate time using interpolation
    let mid_time = epoch + 3600.0; // 1 hour in
    let mid_state = prop.state(mid_time).unwrap();
    println!(
        "State at t+1h: position norm = {:.3} km",
        mid_state.fixed_rows::<3>(0).norm() / 1e3
    );

    // Get states at multiple times
    let times = vec![1800.0, 3600.0, 5400.0, 7200.0];
    for t in &times {
        let s = prop.state(epoch + *t).unwrap();
        let alt = s.fixed_rows::<3>(0).norm() - bh::R_EARTH;
        println!("  t+{}s: altitude = {:.3} km", t, alt / 1e3);
    }

    // Get state in different frames
    let ecef_state = prop.state_ecef(mid_time).unwrap();
    let koe = prop.state_koe(mid_time, bh::AngleFormat::Degrees).unwrap();
    println!(
        "\nECEF position (km): [{:.3}, {:.3}, {:.3}]",
        ecef_state[0] / 1e3,
        ecef_state[1] / 1e3,
        ecef_state[2] / 1e3
    );
    println!(
        "Keplerian elements: a={:.3} km, e={:.6}, i={:.2} deg",
        koe[0] / 1e3,
        koe[1],
        koe[2]
    );

    // Validate
    assert!(num_points > 1);
    assert_eq!(mid_state.len(), 6);
    assert!(mid_state.fixed_rows::<3>(0).norm() > bh::R_EARTH);

    println!("\nExample validated successfully!");
}
