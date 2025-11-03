//! Manage KeplerianPropagator trajectory memory with eviction policies

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

    // Keep only 100 most recent states
    prop.set_eviction_policy_max_size(100).unwrap();

    // Propagate many steps
    prop.propagate_steps(500);
    println!("Trajectory length: {}", prop.trajectory.len());  // Will be 100
    // Trajectory length: 100

    // Alternative: Keep only states within 1 hour of current time
    prop.reset();
    prop.set_eviction_policy_max_age(3600.0).unwrap();  // 3600 seconds = 1 hour
    prop.propagate_steps(500);
    println!("Trajectory length after age policy: {}", prop.trajectory.len());
    // Trajectory length after age policy: 61
}
