//! Convert trajectory from Cartesian to Keplerian representation

#[allow(unused_imports)]
use brahe as bh;
use bh::time::Epoch;
use bh::trajectories::SOrbitTrajectory;
use bh::trajectories::traits::{OrbitFrame, OrbitRepresentation, OrbitalTrajectory};
use bh::traits::Trajectory;
use bh::{state_koe_to_eci, R_EARTH, AngleFormat};
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create trajectory in ECI Cartesian
    let mut traj_cart = SOrbitTrajectory::new(
        OrbitFrame::ECI,
        OrbitRepresentation::Cartesian,
        None
    );

    // Add Cartesian states
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::time::TimeSystem::UTC);
    for i in 0..3 {
        let epoch = epoch0 + (i as f64) * 300.0;
        let oe = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, (i as f64) * 10.0
        );
        let state_cart = state_koe_to_eci(oe, AngleFormat::Degrees);
        traj_cart.add(epoch, state_cart);
    }

    println!("Original representation: {:?}", traj_cart.representation);

    // Convert to Keplerian with radians
    let traj_kep = traj_cart.to_keplerian(AngleFormat::Degrees);

    println!("Converted representation: {:?}", traj_kep.representation);
    println!("Angle format: {:?}", traj_kep.angle_format);

    // Examine Keplerian elements
    for (epoch, oe) in &traj_kep {
        println!("\nEpoch: {}", epoch);
        println!("  Semi-major axis: {:.2} km", oe[0] / 1e3);
        println!("  Eccentricity: {:.6}", oe[1]);
        println!("  Inclination: {:.2}°", oe[2]);
        println!("  RAAN: {:.2}°", oe[3]);
        println!("  Argument of perigee: {:.2}°", oe[4]);
        println!("  Mean Anomaly: {:.2}°", oe[5]);
    }
}

