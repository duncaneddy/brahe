//! Constructing an SGPPropagator from OMM elements using the builder API.
//!
//! The builder takes the eight required OMM inputs -- epoch, mean_motion,
//! eccentricity, inclination, raan, arg_of_pericenter, mean_anomaly, and
//! norad_id -- directly as arguments to `builder()`. Optional inputs such
//! as object_name and bstar default when omitted and are set through
//! chained setters.

use brahe as bh;
use brahe::traits::SStatePropagator;

fn main() {
    bh::initialize_eop().unwrap();

    // ISS OMM mean elements
    let epoch = bh::Epoch::from_datetime(2025, 11, 29, 20, 1, 44.058144, 0.0, bh::TimeSystem::UTC);
    let prop = bh::SGPPropagator::builder(
        epoch,
        15.49193835, // mean_motion (rev/day)
        0.0003723,   // eccentricity
        51.6312,     // inclination (degrees)
        206.3646,    // raan (degrees)
        184.1118,    // arg_of_pericenter (degrees)
        175.9840,    // mean_anomaly (degrees)
        25544,       // norad_id
    )
    .object_name("ISS (ZARYA)")
    .bstar(0.15237e-3)
    .build()
    .unwrap();

    println!("NORAD ID: {}", prop.norad_id);
    println!(
        "Position magnitude: {:.1} km",
        prop.initial_state().fixed_rows::<3>(0).norm() / 1e3
    );
}
