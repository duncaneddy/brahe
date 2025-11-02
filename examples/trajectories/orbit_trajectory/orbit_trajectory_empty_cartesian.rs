//! Create empty OrbitTrajectory in Cartesian representation

#[allow(unused_imports)]
use brahe as bh;
use bh::trajectories::OrbitTrajectory;
use bh::trajectories::traits::{OrbitFrame, OrbitRepresentation};

fn main() {
    bh::initialize_eop().unwrap();

    // Create trajectory in ECI frame, Cartesian representation
    let traj_eci = OrbitTrajectory::new(
        OrbitFrame::ECI,
        OrbitRepresentation::Cartesian,
        None
    );
    println!("Frame (Display): {}", traj_eci.frame);
    println!("Frame (Debug): {:?}", traj_eci.frame);
    println!("Representation (Display): {}", traj_eci.representation);
    println!("Representation (Debug): {:?}", traj_eci.representation);

    // Create trajectory in ECEF frame, Cartesian representation
    let traj_ecef = OrbitTrajectory::new(
        OrbitFrame::ECEF,
        OrbitRepresentation::Cartesian,
        None
    );
    println!("Frame (Display): {}", traj_ecef.frame);
    println!("Frame (Debug): {:?}", traj_ecef.frame);
}

// Output:
// Frame (Display): ECI
// Frame (Debug): OrbitFrame(Earth-Centered Inertial)
// Representation (Display): OrbitRepresentation(Cartesian)
// Representation (Debug): Cartesian
// Frame (Display): ECEF
// Frame (Debug): OrbitFrame(Earth-Centered Earth-Fixed)