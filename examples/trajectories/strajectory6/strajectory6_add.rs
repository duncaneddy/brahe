//! Add states to a 6D trajectory

#[allow(unused_imports)]
use brahe as bh;
use brahe::traits::Trajectory;
use bh::constants::R_EARTH;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create empty trajectory
    let mut traj = bh::STrajectory6::new();

    // Add states
    let epoch0 = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::TimeSystem::UTC);
    let state0 = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0
    );
    traj.add(epoch0, state0);

    let epoch1 = epoch0 + 60.0;
    let state1 = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 456000.0, 0.0, -7600.0, 0.0, 0.0
    );
    traj.add(epoch1, state1);

    println!("Trajectory length: {}", traj.len());
    // Trajectory length: 2
}
