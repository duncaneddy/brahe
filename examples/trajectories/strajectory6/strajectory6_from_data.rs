//! Create a 6D trajectory from existing data

#[allow(unused_imports)]
use brahe as bh;
use brahe::traits::Trajectory;
use bh::constants::R_EARTH;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create epochs
    let epoch0 = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::TimeSystem::UTC);
    let epoch1 = epoch0 + 60.0;
    let epoch2 = epoch0 + 120.0;

    // Create 6D states
    let state0 = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0
    );
    let state1 = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 456000.0, 0.0, -7600.0, 0.0, 0.0
    );
    let state2 = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 0.0, 0.0, 0.0, -7600.0, 0.0
    );

    // Create trajectory from data
    let epochs = vec![epoch0, epoch1, epoch2];
    let states = vec![state0, state1, state2];
    let traj = bh::STrajectory6::from_data(epochs, states).unwrap();

    println!("Trajectory length: {}", traj.len());
    // Trajectory length: 3
}
