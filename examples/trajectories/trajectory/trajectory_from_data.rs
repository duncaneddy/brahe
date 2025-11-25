//! Create DTrajectory from existing epochs and states

#[allow(unused_imports)]
use brahe as bh;
use bh::time::Epoch;
use bh::trajectories::DTrajectory;
use bh::traits::Trajectory;
use bh::constants::R_EARTH;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create epochs
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::time::TimeSystem::UTC);
    let epoch1 = epoch0 + 60.0;  // 1 minute later
    let epoch2 = epoch0 + 120.0;  // 2 minutes later

    // Create states
    let state0 = na::DVector::from_vec(vec![
        R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0
    ]);
    let state1 = na::DVector::from_vec(vec![
        R_EARTH + 500e3, 456000.0, 0.0, -7600.0, 0.0, 0.0
    ]);
    let state2 = na::DVector::from_vec(vec![
        R_EARTH + 500e3, 0.0, 0.0, 0.0, -7600.0, 0.0
    ]);

    // Create trajectory from data
    let epochs = vec![epoch0, epoch1, epoch2];
    let states = vec![state0, state1, state2];
    let traj = DTrajectory::from_data(epochs, states).unwrap();

    println!("Trajectory length: {}", traj.len());
    // Output: 3
}
