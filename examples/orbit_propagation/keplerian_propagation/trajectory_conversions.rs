//! Convert KeplerianPropagator trajectory to different reference frames

#[allow(unused_imports)]
use brahe as bh;
use bh::traits::{SStatePropagator, Trajectory};
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();  // Required for ECEF conversions

    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let elements = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let mut prop = bh::KeplerianPropagator::from_keplerian(
        epoch, elements, bh::AngleFormat::Degrees, 60.0
    );
    prop.propagate_steps(10);

    // Convert entire trajectory to different frames
    let traj_eci = prop.trajectory.to_eci();       // ECI Cartesian
    let traj_ecef = prop.trajectory.to_ecef();     // ECEF Cartesian
    let traj_kep = prop.trajectory.to_keplerian(bh::AngleFormat::Radians);

    println!("ECI trajectory: {} states", traj_eci.len());
    // ECI trajectory: 11 states
    println!("ECEF trajectory: {} states", traj_ecef.len());
    // ECEF trajectory: 11 states
    println!("Keplerian trajectory: {} states", traj_kep.len());
    // Keplerian trajectory: 11 states
}
