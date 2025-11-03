//! Initialize KeplerianPropagator from ECEF Cartesian state vector

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();  // Required for ECEF ↔ ECI transformations

    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Get state in ECI, then convert to ECEF for demonstration
    let elements = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let state_eci = bh::state_osculating_to_cartesian(elements, bh::AngleFormat::Degrees);
    let state_ecef = bh::state_eci_to_ecef(epoch, state_eci);

    // Create propagator from ECEF state
    let prop = bh::KeplerianPropagator::from_ecef(epoch, state_ecef, 60.0);

    println!("ECEF position magnitude: {:.1} km",
             state_ecef.fixed_rows::<3>(0).norm() / 1e3);
    // ECEF position magnitude: 6873.3 km
}
