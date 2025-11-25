//! Chain multiple frame and representation conversions

#[allow(unused_imports)]
use brahe as bh;
use bh::time::Epoch;
use bh::trajectories::SOrbitTrajectory;
use bh::trajectories::traits::{OrbitFrame, OrbitRepresentation, OrbitalTrajectory};
use bh::traits::Trajectory;
use bh::{state_osculating_to_cartesian, R_EARTH, AngleFormat};
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Start with ECI Cartesian trajectory
    let mut traj_eci_cart = SOrbitTrajectory::new(
        OrbitFrame::ECI,
        OrbitRepresentation::Cartesian,
        None
    );

    // Add states
    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::time::TimeSystem::UTC);
    let oe = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 0.001, 0.9, 1.0, 0.5, 0.0
    );
    let state_cart = state_osculating_to_cartesian(oe, AngleFormat::Radians);
    traj_eci_cart.add(epoch, state_cart);

    println!("Original:");
    println!("  Frame: {:?}", traj_eci_cart.frame);
    println!("  Representation: {:?}", traj_eci_cart.representation);

    // Convert to ECEF frame (stays Cartesian)
    let traj_ecef_cart = traj_eci_cart.to_ecef();
    println!("\nAfter to_ecef():");
    println!("  Frame: {:?}", traj_ecef_cart.frame);
    println!("  Representation: {:?}", traj_ecef_cart.representation);

    // Convert back to ECI
    let traj_eci_cart2 = traj_ecef_cart.to_eci();
    println!("\nAfter to_eci():");
    println!("  Frame: {:?}", traj_eci_cart2.frame);
    println!("  Representation: {:?}", traj_eci_cart2.representation);

    // Convert to Keplerian (in ECI frame)
    let traj_eci_kep = traj_eci_cart2.to_keplerian(AngleFormat::Radians);
    println!("\nAfter to_keplerian():");
    println!("  Frame: {:?}", traj_eci_kep.frame);
    println!("  Representation: {:?}", traj_eci_kep.representation);
}
