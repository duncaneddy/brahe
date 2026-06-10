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

    // Calculate perigee velocity
    let v_perigee = bh::perigee_velocity(oe_deg[0], oe_deg[1]);
    println!("Perigee velocity: {:.3} m/s", v_perigee);

    // Calculate apogee velocity
    let v_apogee = bh::apogee_velocity(oe_deg[0], oe_deg[1]);
    println!("Apogee velocity: {:.3} m/s", v_apogee);

}

