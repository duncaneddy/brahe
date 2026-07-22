#![allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    // Initialize EOP
    bh::initialize_eop().unwrap();

    // Initialize a Keplerian state
    // Define orbital elements [a, e, i, Ω, ω, M] in meters and degrees
    // LEO satellite: 500 km altitude, 97.8° inclination (approx sun-synchronous)
    let oe_deg = na::vector![
        bh::R_EARTH + 500e3,  // Semi-major axis (m)
        0.01,                  // Eccentricity
        97.8,                  // Inclination (deg)
        15.0,                  // Right ascension of ascending node (deg)
        30.0,                  // Argument of periapsis (deg)
        45.0,                  // Mean anomaly (deg)
    ];

    // Convert orbital elements to Cartesian state using degrees
    let x_deg = bh::state_koe_to_eci(oe_deg, bh::AngleFormat::Degrees);

    // Convert back to degrees
    let oe_deg_2 = bh::state_eci_to_koe(x_deg, bh::AngleFormat::Degrees);

    println!("Original Keplerian elements:");
    for (i, elem) in oe_deg.iter().enumerate() {
        println!("  [{i}]: {elem:.3}");
    }
    println!("Converted Cartesian state:");
    for (i, elem) in x_deg.iter().enumerate() {
        println!("  [{i}]: {elem:.3}");
    }
    println!("Back to Keplerian elements:");
    for (i, elem) in oe_deg_2.iter().enumerate() {
        println!("  [{i}]: {elem:.3}");
    }
}

