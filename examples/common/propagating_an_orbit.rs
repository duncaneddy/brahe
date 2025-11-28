//! This example demonstrates how to propagate a Keplerian orbit using the Brahe library.

#[allow(unused_imports)]
use brahe as bh;
use brahe::{Epoch, R_EARTH, KeplerianPropagator, AngleFormat};
use brahe::traits::{SStatePropagator, Trajectory};
use nalgebra::Vector6;

fn main() {
    // Define the initial Keplerian elements
    let a = R_EARTH + 700e3;  // Semi-major axis: 700 km altitude
    let e = 0.001;            // Eccentricity
    let i = 98.7;             // Inclination in degrees
    let raan = 15.0;          // Right Ascension of Ascending Node in degrees
    let argp = 30.0;          // Argument of Perigee in degrees
    let mean_anomaly = 75.0;  // Mean Anomaly at epoch in degrees

    let initial_state = Vector6::new(a, e, i, raan, argp, mean_anomaly);

    // Define the epoch time
    let epoch = Epoch::now();

    // Create the Keplerian Orbit Propagator
    let dt = 60.0;  // Time step in seconds
    let mut propagator = KeplerianPropagator::from_keplerian(
        epoch,
        initial_state,
        AngleFormat::Degrees,
        dt
    );

    // Propagate the orbit for 3 time steps
    propagator.propagate_steps(3);

    // States are stored as a Trajectory object
    assert_eq!(propagator.trajectory.len(), 4);  // Initial state + 3 propagated states

    // Convert trajectory to ECI coordinates
    let eci_trajectory = propagator.trajectory.to_eci();

    // Iterate over all stored states
    for i in 0..eci_trajectory.len() {
        let epoch = eci_trajectory.epochs[i];
        let state = eci_trajectory.states[i].clone();
        println!(
            "Epoch: {}, Position (ECI): {:.2} km, {:.2} km, {:.2} km",
            epoch,
            state[0] / 1e3,
            state[1] / 1e3,
            state[2] / 1e3
        );
    }
    // Output (will vary based on current time):
    // Epoch: 2025-10-24 22:14:56.707 UTC, Position (ECI): -1514.38 km, -1475.59 km, 6753.03 km
    // Epoch: 2025-10-24 22:15:56.707 UTC, Position (ECI): -1935.70 km, -1568.01 km, 6623.80 km
    // Epoch: 2025-10-24 22:16:56.707 UTC, Position (ECI): -2349.19 km, -1654.08 km, 6467.76 km
    // Epoch: 2025-10-24 22:17:56.707 UTC, Position (ECI): -2753.17 km, -1733.46 km, 6285.55 km

    // Propagate for 7 days
    let end_epoch = epoch + 86400.0 * 7.0;  // 7 days later
    propagator.propagate_to(end_epoch);

    // Confirm the final epoch is close to expected time
    let time_diff = (propagator.current_epoch() - end_epoch).abs();
    assert!(time_diff < 1.0e-6, "Final epoch should be within 1 second of target");
    println!("Propagation complete. Final epoch: {}", propagator.current_epoch());
    // Output (will vary based on current time):
    // Propagation complete. Final epoch: 2025-10-31 22:18:40.413 UTC
}
