//! Propagate KeplerianPropagator forward multiple steps at once

#[allow(unused_imports)]
use brahe as bh;
use bh::traits::{OrbitPropagator, Trajectory};
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let elements = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let mut prop = bh::KeplerianPropagator::from_keplerian(
        epoch, elements, bh::AngleFormat::Degrees, 60.0
    );

    // Take 10 steps (10 × 60 = 600 seconds)
    prop.propagate_steps(10);
    println!("After 10 steps: {:.1} seconds elapsed",
             prop.current_epoch() - epoch);
    // After 10 steps: 600.0 seconds elapsed
    println!("Trajectory length: {}", prop.trajectory.len());
    // Trajectory length: 11
}
