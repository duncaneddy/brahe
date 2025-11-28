//! Create empty SOrbitTrajectory in Keplerian representation

#[allow(unused_imports)]
use brahe as bh;
use bh::trajectories::SOrbitTrajectory;
use bh::trajectories::traits::{OrbitFrame, OrbitRepresentation};
use bh::AngleFormat;

fn main() {
    bh::initialize_eop().unwrap();

    // Create trajectory in ECI frame, Keplerian representation with radians
    let _traj_kep_rad = SOrbitTrajectory::new(
        OrbitFrame::ECI,
        OrbitRepresentation::Keplerian,
        Some(AngleFormat::Radians)
    );

    // Create trajectory in ECI frame, Keplerian representation with degrees
    let _traj_kep_deg = SOrbitTrajectory::new(
        OrbitFrame::ECI,
        OrbitRepresentation::Keplerian,
        Some(AngleFormat::Degrees)
    );
}
