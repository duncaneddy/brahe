//! Create empty OrbitTrajectory in Keplerian representation

#[allow(unused_imports)]
use brahe as bh;
use bh::trajectories::OrbitTrajectory;
use bh::trajectories::traits::{OrbitFrame, OrbitRepresentation};
use bh::AngleFormat;

fn main() {
    bh::initialize_eop().unwrap();

    // Create trajectory in ECI frame, Keplerian representation with radians
    let traj_kep_rad = OrbitTrajectory::new(
        OrbitFrame::ECI,
        OrbitRepresentation::Keplerian,
        Some(AngleFormat::Radians)
    );

    // Create trajectory in ECI frame, Keplerian representation with degrees
    let traj_kep_deg = OrbitTrajectory::new(
        OrbitFrame::ECI,
        OrbitRepresentation::Keplerian,
        Some(AngleFormat::Degrees)
    );
}
