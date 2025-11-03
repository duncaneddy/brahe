//! Reset KeplerianPropagator to initial conditions

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

    // Propagate forward
    prop.propagate_steps(100);
    println!("After propagation: {} states", prop.trajectory.len());
    // After propagation: 101 states

    // Reset to initial conditions
    prop.reset();
    println!("After reset: {} states", prop.trajectory.len());
    // After reset: 1 states
    println!("Current epoch: {}", prop.current_epoch());
    // Current epoch: 2024-01-01 00:00:00.000 UTC
}
