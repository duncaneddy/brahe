//! Step KeplerianPropagator forward one step at a time

#[allow(unused_imports)]
use brahe as bh;
use bh::traits::{OrbitPropagator, Trajectory};
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create propagator
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let elements = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let mut prop = bh::KeplerianPropagator::from_keplerian(
        epoch, elements, bh::AngleFormat::Degrees, 60.0
    );

    // Take one step (60 seconds)
    prop.step();
    println!("After 1 step: {}", prop.current_epoch());
    // After 1 step: 2024-01-01 00:01:00.000 UTC

    // Step by custom duration (120 seconds)
    prop.step_by(120.0);
    println!("After custom step: {}", prop.current_epoch());
    // After custom step: 2024-01-01 00:03:00.000 UTC

    // Trajectory now contains 3 states (initial + 2 steps)
    println!("Trajectory length: {}", prop.trajectory.len());
    // Trajectory length: 3
}
