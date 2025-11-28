//! Query KeplerianPropagator state at arbitrary epochs without building trajectory

#[allow(unused_imports)]
use brahe as bh;
use bh::utils::{DStateProvider, DOrbitStateProvider};
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();  // Required for frame transformations

    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let elements = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let prop = bh::KeplerianPropagator::from_keplerian(
        epoch, elements, bh::AngleFormat::Degrees, 60.0
    );

    // Query state 1 hour later (doesn't add to trajectory)
    let query_epoch = epoch + 3600.0;
    let state_native = prop.state(query_epoch).unwrap();       // Native format of propagator internal state  (Keplerian)
    let state_eci = prop.state_eci(query_epoch).unwrap();      // ECI Cartesian
    let state_ecef = prop.state_ecef(query_epoch).unwrap();    // ECEF Cartesian
    let _state_kep = prop.state_koe(
        query_epoch, bh::AngleFormat::Degrees
    ).unwrap();

    println!("Native state (Keplerian): a={:.1} km", state_native[0] / 1e3);
    // Native state (Keplerian): a=6878.1 km
    println!("ECI position magnitude: {:.1} km",
             state_eci.fixed_rows::<3>(0).norm() / 1e3);
    // ECI position magnitude: 6877.7 km
    println!("ECEF position magnitude: {:.1} km",
             state_ecef.fixed_rows::<3>(0).norm() / 1e3);
    // ECEF position magnitude: 6877.7 km
}
