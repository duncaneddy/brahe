//! Verify round-trip frame conversion consistency (ECI -> ECEF -> ECI)

#[allow(unused_imports)]
use brahe as bh;
use bh::time::Epoch;
use bh::trajectories::OrbitTrajectory;
use bh::trajectories::traits::{OrbitFrame, OrbitRepresentation, OrbitalTrajectory};
use bh::traits::Trajectory;
use bh::constants::R_EARTH;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create trajectory in ECI
    let mut traj_eci_original = OrbitTrajectory::new(
        OrbitFrame::ECI,
        OrbitRepresentation::Cartesian,
        None
    );

    // Add a state
    let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0,
        bh::time::TimeSystem::UTC);
    let state_original = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0
    );
    traj_eci_original.add(epoch, state_original);

    // Convert to ECEF and back to ECI
    let traj_ecef = traj_eci_original.to_ecef();
    let traj_eci_roundtrip = traj_ecef.to_eci();

    // Compare original and round-trip states
    let (_, state_roundtrip) = traj_eci_roundtrip.first().unwrap();
    let diff = state_original - state_roundtrip;

    println!("Position difference: {:.6e} m",
        diff.fixed_rows::<3>(0).norm());
    println!("Velocity difference: {:.6e} m/s",
        diff.fixed_rows::<3>(3).norm());
    // Expected: Very small differences (numerical precision)
}

// Output:
// Position difference: 2.499882e-10 m
// Velocity difference: 1.829382e-12 m/s