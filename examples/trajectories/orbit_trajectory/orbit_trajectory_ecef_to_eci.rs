//! Convert trajectory from ECEF to ECI frame

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

    // Create trajectory in ECEF frame
    let mut traj_ecef = SOrbitTrajectory::new(
        OrbitFrame::ECEF,
        OrbitRepresentation::Cartesian,
        None
    );

    // Add dummy states in ECEF
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::time::TimeSystem::UTC);
    for i in 0..3 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state_ecef = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 7600.0
        );
        traj_ecef.add(epoch, state_ecef);
    }

    println!("Original frame: {:?}", traj_ecef.frame);

    // Convert to ECI
    let traj_eci = traj_ecef.to_eci();

    println!("Converted frame: {:?}", traj_eci.frame);
    println!("Trajectory length: {}", traj_eci.len());

    // Iterate over converted states
    for (epoch, state_eci) in &traj_eci {
        let pos_mag = state_eci.fixed_rows::<3>(0).norm();
        let vel_mag = state_eci.fixed_rows::<3>(3).norm();
        println!("Epoch: {}", epoch);
        println!("  Position magnitude: {:.2} km", pos_mag / 1e3);
        println!("  Velocity magnitude: {:.2} m/s", vel_mag);
    }
}
