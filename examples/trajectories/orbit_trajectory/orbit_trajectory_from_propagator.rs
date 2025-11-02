//! Create OrbitTrajectory through orbit propagation

#[allow(unused_imports)]
use brahe as bh;
use bh::time::Epoch;
use bh::traits::{Trajectory, OrbitPropagator};
use bh::{KeplerianPropagator, R_EARTH, AngleFormat};
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define orbital elements
    let oe = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 0.001, 97.8_f64.to_radians(),
        15.0_f64.to_radians(), 30.0_f64.to_radians(), 0.0
    );

    // Create epoch and propagator
    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::time::TimeSystem::UTC);
    let mut propagator = KeplerianPropagator::from_keplerian(
        epoch, oe, AngleFormat::Radians, 60.0
    );

    // Propagate for several steps
    propagator.propagate_steps(10);

    // Access the trajectory
    let traj = &propagator.trajectory;
    println!("Trajectory length: {}", traj.len());  // Output: 11
    println!("Frame: {}", traj.frame);  // Output: ECI
    println!("Representation: {}", traj.representation);  // Output: Keplerian
}
