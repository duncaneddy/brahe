//! Create empty SOrbitTrajectory in Keplerian representation

#[allow(unused_imports)]
use brahe as bh;
use bh::trajectories::SOrbitTrajectory;
use bh::frames::CelestialFrame;
use bh::trajectories::traits::OrbitRepresentation;
use bh::AngleFormat;

fn main() {
    bh::initialize_eop().unwrap();

    // Create trajectory in ECI frame, Keplerian representation with radians
    let _traj_kep_rad = SOrbitTrajectory::new(
        CelestialFrame::ECI,
        OrbitRepresentation::Keplerian,
        Some(AngleFormat::Radians)
    ).unwrap();

    // Create trajectory in ECI frame, Keplerian representation with degrees
    let _traj_kep_deg = SOrbitTrajectory::new(
        CelestialFrame::ECI,
        OrbitRepresentation::Keplerian,
        Some(AngleFormat::Degrees)
    ).unwrap();
}

