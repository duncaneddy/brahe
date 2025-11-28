//! Convert between Keplerian orbital elements and Cartesian state vectors

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define orbital elements [a, e, i, Ω, ω, M] in meters and degrees
    // LEO satellite: 500 km altitude, 97.8° inclination (~sun-synchronous)
    let oe_deg = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 500e3,  // Semi-major axis (m)
        0.01,                 // Eccentricity
        97.8,                 // Inclination (deg)
        15.0,                 // Right ascension of ascending node (deg)
        30.0,                 // Argument of periapsis (deg)
        45.0                  // Mean anomaly (deg)
    );

    // Convert orbital elements to Cartesian state using degrees
    let state = bh::state_koe_to_eci(oe_deg, bh::AngleFormat::Degrees);

    println!("Cartesian state [x, y, z, vx, vy, vz] (m, m/s):");
    println!("Position: [{:.3}, {:.3}, {:.3}]", state[0], state[1], state[2]);
    println!("Velocity: [{:.6}, {:.6}, {:.6}]", state[3], state[4], state[5]);
    // Cartesian state  (m, m/s):
    // Position: [1848964.106, -434937.468, 6560410.530]
    // Velocity: [-7098.379734, -2173.344867, 1913.333385]
}
