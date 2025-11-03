//! Initialize KeplerianPropagator from Keplerian orbital elements

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define initial epoch
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Define Keplerian elements [a, e, i, Ω, ω, M]
    let elements = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 500e3,  // Semi-major axis (m)
        0.001,                // Eccentricity
        97.8,                 // Inclination (degrees)
        15.0,                 // RAAN (degrees)
        30.0,                 // Argument of perigee (degrees)
        45.0                  // Mean anomaly (degrees)
    );

    // Create propagator with 60-second step size
    let _prop = bh::KeplerianPropagator::from_keplerian(
        epoch, elements, bh::AngleFormat::Degrees, 60.0
    );

    println!("Orbital period: {:.1} seconds", bh::orbital_period(elements[0]));
    // Orbital period: 5677.0 seconds
}
