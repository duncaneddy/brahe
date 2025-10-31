//! Access trajectory states and epochs by index

#[allow(unused_imports)]
use brahe as bh;
use brahe::traits::Trajectory;
use bh::constants::R_EARTH;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create and populate trajectory
    let mut traj = bh::STrajectory6::new();
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

    let epoch2 = epoch1 + 60.0;
    let state2 = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 912000.0, 0.0, 0.0, -7600.0, 0.0
    );
    traj.add(epoch2, state2);

    // Access by index
    let retrieved_epoch = traj.epoch_at_idx(1).unwrap();
    let retrieved_state = traj.state_at_idx(1).unwrap();

    println!("Epoch: {}", retrieved_epoch);
    println!("Position: [{:.2}, {:.2}, {:.2}] m",
        retrieved_state[0], retrieved_state[1], retrieved_state[2]);
    println!("Velocity: [{:.2}, {:.2}, {:.2}] m/s",
        retrieved_state[3], retrieved_state[4], retrieved_state[5]);
}

// Output:
// Epoch: 2024-01-01 00:01:00.000 UTC
// Position: [6878136.30, 456000.00, 0.00] m
// Velocity: [-7600.00, 0.00, 0.00] m/s