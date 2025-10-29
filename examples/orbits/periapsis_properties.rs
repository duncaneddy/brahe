//! Compute periapsis properties for an orbit.
//!
//! This example demonstrates computing periapsis velocity and distance
//! for a given orbit, including Earth-specific perigee functions.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Define orbit parameters
    let a = bh::constants::R_EARTH + 500.0e3; // Semi-major axis (m)
    let e = 0.01; // Eccentricity

    // Compute periapsis velocity (generic)
    let periapsis_velocity = bh::orbits::periapsis_velocity(a, e, bh::constants::GM_EARTH);
    println!("Periapsis velocity: {:.3} m/s", periapsis_velocity);

    // Compute as a perigee velocity (Earth-specific)
    let perigee_velocity = bh::orbits::perigee_velocity(a, e);
    println!("Perigee velocity:   {:.3} m/s", perigee_velocity);

    // Compute periapsis distance
    let periapsis_distance = bh::orbits::periapsis_distance(a, e);
    println!("Periapsis distance: {:.3} m", periapsis_distance);

    // Expected output:
    // Periapsis velocity: 7689.119 m/s
    // Perigee velocity:   7689.119 m/s
    // Periapsis distance: 6809354.937 m
}
