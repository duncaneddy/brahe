//! Convert between Keplerian orbital elements and Cartesian state vectors

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    bh::initialize_eop().unwrap();

    // Define orbital elements [a, e, i, Ω, ω, M] in meters and radians
    // LEO satellite: 500 km altitude, 45° inclination
    let oe_rad = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 500e3,  // Semi-major axis (m)
        0.01,                 // Eccentricity
        PI/4.0,               // Inclination (rad)
        PI/8.0,               // Right ascension of ascending node (rad)
        PI/2.0,               // Argument of periapsis (rad)
        3.0*PI/4.0            // Mean anomaly (rad)
    );

    // Convert orbital elements to Cartesian state using radians
    let state = bh::state_koe_to_eci(oe_rad, bh::AngleFormat::Radians);

    println!("Cartesian state [x, y, z, vx, vy, vz] (m, m/s):");
    println!("Position: [{:.3}, {:.3}, {:.3}]", state[0], state[1], state[2]);
    println!("Velocity: [{:.6}, {:.6}, {:.6}]", state[3], state[4], state[5]);
    // Cartesian state  (m, m/s):
    // Position: [-3117582.037, -5092452.343, -3511765.495]
    // Velocity: [6408.435846, -1407.501408, -3752.763969]
}
