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

    // Orbital period
    let period = bh::orbital_period(oe_deg[0]);
    println!("Orbital period: {:.3} minutes", period/60.0);
}

