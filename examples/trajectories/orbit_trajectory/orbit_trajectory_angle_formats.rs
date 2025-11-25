//! Convert to Keplerian with different angle formats (radians vs degrees)

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

    // Create trajectory in ECI Cartesian
    let mut traj_cart = SOrbitTrajectory::new(
        OrbitFrame::ECI,
        OrbitRepresentation::Cartesian,
        None
    );

    // Add a state
    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::time::TimeSystem::UTC);
    let oe = na::SVector::<f64, 6>::new(
        R_EARTH + 500e3, 0.001, 0.9, 1.0, 0.5, 0.0
    );
    let state_cart = state_osculating_to_cartesian(oe, AngleFormat::Radians);
    traj_cart.add(epoch, state_cart);

    // Convert to Keplerian with radians
    let traj_kep_rad = traj_cart.to_keplerian(AngleFormat::Radians);
    let (_, oe_rad) = traj_kep_rad.first().unwrap();

    // Convert to Keplerian with degrees
    let traj_kep_deg = traj_cart.to_keplerian(AngleFormat::Degrees);
    let (_, oe_deg) = traj_kep_deg.first().unwrap();

    println!("Radians version:");
    println!("  Inclination: {:.6} rad = {:.2}°", oe_rad[2], oe_rad[2].to_degrees());

    println!("\nDegrees version:");
    println!("  Inclination: {:.2}°", oe_deg[2]);
}

// Output:
// Radians version:
//   Inclination: 0.900000 rad = 51.57°

// Degrees version:
//   Inclination: 51.57°