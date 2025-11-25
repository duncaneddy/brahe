//! Create SOrbitTrajectory from existing epochs and orbital states

#[allow(unused_imports)]
use brahe as bh;
use bh::time::Epoch;
use bh::trajectories::SOrbitTrajectory;
use bh::trajectories::traits::{OrbitFrame, OrbitRepresentation, OrbitalTrajectory};
use bh::traits::Trajectory;
use bh::constants::R_EARTH;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create epochs
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::time::TimeSystem::UTC);
    let epoch1 = epoch0 + 60.0;
    let epoch2 = epoch0 + 120.0;

    // Create Cartesian states
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
    let traj = SOrbitTrajectory::from_orbital_data(
        epochs,
        states,
        OrbitFrame::ECI,
        OrbitRepresentation::Cartesian,
        None, // Angle Format
        None  // No covariances
    );

    println!("Trajectory length: {}", traj.len());  
    // Trajectory length: 3
}
