//! Create empty SOrbitTrajectory in Cartesian representation

#[allow(unused_imports)]
use brahe as bh;
use bh::trajectories::SOrbitTrajectory;
use bh::frames::CelestialFrame;
use bh::trajectories::traits::OrbitRepresentation;

fn main() {
    bh::initialize_eop().unwrap();

    // Create trajectory in ECI frame, Cartesian representation
    let traj_eci = SOrbitTrajectory::new(
        CelestialFrame::ECI,
        OrbitRepresentation::Cartesian,
        None
    ).unwrap();
    println!("Frame (Display): {}", traj_eci.frame);
    println!("Frame (Debug): {:?}", traj_eci.frame);
    println!("Representation (Display): {}", traj_eci.representation);
    println!("Representation (Debug): {:?}", traj_eci.representation);

    // Create trajectory in ECEF frame, Cartesian representation
    let traj_ecef = SOrbitTrajectory::new(
        CelestialFrame::ECEF,
        OrbitRepresentation::Cartesian,
        None
    ).unwrap();
    println!("Frame (Display): {}", traj_ecef.frame);
    println!("Frame (Debug): {:?}", traj_ecef.frame);
}

