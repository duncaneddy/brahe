//! Convert trajectory from ECI to ECEF frame

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

    // Create trajectory in ECI frame
    let mut traj_eci = SOrbitTrajectory::new(
        OrbitFrame::ECI,
        OrbitRepresentation::Cartesian,
        None
    );

    // Add states in ECI
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::time::TimeSystem::UTC);
    for i in 0..5 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state_eci = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3, (i as f64) * 100e3, 0.0, 0.0, 7600.0, 0.0
        );
        traj_eci.add(epoch, state_eci);
    }

    println!("Original frame: {:?}", traj_eci.frame);
    println!("Original representation: {:?}", traj_eci.representation);

    // Convert all states in trajectory to ECEF
    let traj_ecef = traj_eci.to_ecef();

    println!("\nConverted frame: {:?}", traj_ecef.frame);
    println!("Converted representation: {:?}", traj_ecef.representation);
    println!("Same number of states: {}", traj_ecef.len());

    // Compare first states
    let state_eci_first = traj_eci.state_at_idx(0).unwrap();
    let state_ecef_first = traj_ecef.state_at_idx(0).unwrap();
    println!("\nFirst state ECI: [{}, {}, {}] m",
        state_eci_first[0], state_eci_first[1], state_eci_first[2]
    );
    println!("First state ECEF: [{}, {}, {}] m",
        state_ecef_first[0], state_ecef_first[1], state_ecef_first[2]
    );
}

// Output:
// Original frame: OrbitFrame(Earth-Centered Inertial)
// Original representation: OrbitRepresentation(Cartesian)

// Converted frame: OrbitFrame(Earth-Centered Earth-Fixed)
// Converted representation: OrbitRepresentation(Cartesian)
// Same number of states: 5

// First state ECI: [6878136.3, 0, 0] m
// First state ECEF: [-1176064.0596141217, -6776826.507241379, 15961.82358860613] m