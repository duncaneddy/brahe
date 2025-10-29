//! Compute orbital period for Earth-orbiting satellites.
//!
//! This example demonstrates computing the orbital period from semi-major axis
//! for both Earth-specific and general gravitational parameter cases.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Define orbit parameters
    let a = bh::constants::R_EARTH + 500.0e3; // Semi-major axis (m) - LEO orbit at 500 km altitude

    // Compute orbital period for Earth orbit (uses GM_EARTH internally)
    let period_earth = bh::orbits::orbital_period(a);
    println!("Orbital period (Earth): {:.3} s", period_earth);
    println!("Orbital period (Earth): {:.3} min", period_earth / 60.0);

    // Compute orbital period for general body (explicit GM)
    let period_general = bh::orbits::orbital_period_general(a, bh::constants::GM_EARTH);
    println!("Orbital period (general): {:.3} s", period_general);

    // Verify they match
    println!("Difference: {:.2e} s", (period_earth - period_general).abs());

    // Example with approximate GEO altitude
    let a_geo = bh::constants::R_EARTH + 35786e3;
    let period_geo = bh::orbits::orbital_period(a_geo);
    println!("\nGEO orbital period: {:.3} hours", period_geo / 3600.0);

    // Expected output:
    // Orbital period (Earth): 5676.977 s
    // Orbital period (Earth): 94.616 min
    // Orbital period (general): 5676.977 s
    // Difference: 0.00e0 s

    // GEO orbital period: 23.934 hours
}
