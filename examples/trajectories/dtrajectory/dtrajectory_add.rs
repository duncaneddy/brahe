//! Add states to a DTrajectory one at a time

#[allow(unused_imports)]
use brahe as bh;
use bh::time::Epoch;
use bh::trajectories::DTrajectory;
use bh::traits::Trajectory;
use bh::constants::R_EARTH;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create empty trajectory
    let mut traj = DTrajectory::new(6);

    // Add states
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::time::TimeSystem::UTC);
    let state0 = na::DVector::from_vec(vec![
        R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0
    ]);
    traj.add(epoch0, state0);

    println!("Trajectory length: {}", traj.len());
    // Trajectory length: 1

    let epoch1 = epoch0 + 60.0;
    let state1 = na::DVector::from_vec(vec![
        R_EARTH + 500e3, 456000.0, 0.0, -7600.0, 0.0, 0.0
    ]);
    traj.add(epoch1, state1);

    println!("Trajectory length: {}", traj.len());
    // Trajectory length: 2
}
