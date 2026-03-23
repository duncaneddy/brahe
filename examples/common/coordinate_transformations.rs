//! This example demonstrates how to work with coordinate transformations
//! using the Brahe library. It shows how you can convert between Keplerian elements and
//! the Earth-Centered Earth-Fixed (ECEF) coordinate system and vice versa.

#[allow(unused_imports)]
use brahe as bh;
use brahe::{Epoch, TimeSystem, R_EARTH, state_koe_to_eci,
            state_eci_to_ecef, state_ecef_to_eci, state_eci_to_koe, AngleFormat};
use nalgebra::Vector6;

fn main() {
    // Initialize EOP
    bh::initialize_eop().unwrap();

    // Define orbital elements
    let a = R_EARTH + 700e3;    // Semi-major axis in meters (700 km altitude)
    let e = 0.001;              // Eccentricity
    let i = 98.7;               // Inclination in degrees
    let raan = 15.0;            // Right Ascension of Ascending Node in degrees
    let arg_periapsis = 30.0;   // Argument of Periapsis in degrees
    let mean_anomaly = 45.0;    // Mean Anomaly in degrees

    // Create a state vector from orbital elements
    let state_kep = Vector6::new(a, e, i, raan, arg_periapsis, mean_anomaly);

    // Convert Keplerian state to ECI coordinates
    let state_eci = state_koe_to_eci(state_kep, AngleFormat::Degrees);
    println!("ECI Coordinates: {:?}", state_eci);
    let epoch = Epoch::from_datetime(2024, 6, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

    // Convert ECI coordinates to ECEF coordinates at the given epoch
    let state_ecef = state_eci_to_ecef(epoch, state_eci);
    println!("ECEF Coordinates: {:?}", state_ecef);
    let state_eci_2 = state_ecef_to_eci(epoch, state_ecef);
    println!("Recovered ECI Coordinates: {:?}", state_eci_2);
    let state_kep_2 = state_eci_to_koe(state_eci_2, AngleFormat::Degrees);
    println!("Recovered Keplerian Elements: {:?}", state_kep_2);
}

