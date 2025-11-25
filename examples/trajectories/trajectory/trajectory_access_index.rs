//! Retrieve states and epochs by their index

#[allow(unused_imports)]
use brahe as bh;
use bh::time::Epoch;
use bh::trajectories::DTrajectory;
use bh::traits::Trajectory;
use bh::constants::R_EARTH;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create and populate trajectory
    let mut traj = DTrajectory::new(6);
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::time::TimeSystem::UTC);
    let state0 = na::DVector::from_vec(vec![
        R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0
    ]);
    traj.add(epoch0, state0);

    let epoch1 = epoch0 + 60.0;
    let state1 = na::DVector::from_vec(vec![
        R_EARTH + 600e3, 456000.0, 0.0, -7600.0, 0.0, 0.0
    ]);
    traj.add(epoch1, state1);

    let epoch2 = epoch0 + 120.0;
    let state2 = na::DVector::from_vec(vec![
        R_EARTH + 700e3, 0.0, 0.0, 0.0, -7600.0, 0.0
    ]);
    traj.add(epoch2, state2);

    // Access by index
    let retrieved_epoch = traj.epoch_at_idx(1).unwrap();
    let retrieved_state = traj.state_at_idx(1).unwrap();

    println!("Epoch: {}", retrieved_epoch);
    println!("Altitude: {:.2} m", retrieved_state[0] - R_EARTH);
}

// Output:
// Epoch: 2024-01-01 00:01:00.000 UTC
// Altitude: 600000.00 m