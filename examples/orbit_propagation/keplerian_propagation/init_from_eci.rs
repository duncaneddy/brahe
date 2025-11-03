//! Initialize KeplerianPropagator from ECI Cartesian state vector

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Define Cartesian state in ECI frame [x, y, z, vx, vy, vz]
    // Convert from Keplerian elements for this example
    let elements = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let state_eci = bh::state_osculating_to_cartesian(elements, bh::AngleFormat::Degrees);

    // Create propagator from ECI state
    let _prop = bh::KeplerianPropagator::from_eci(epoch, state_eci, 60.0);

    println!("Initial position magnitude: {:.1} km",
             state_eci.fixed_rows::<3>(0).norm() / 1e3);
    // Initial position magnitude: 6873.3 km
}
