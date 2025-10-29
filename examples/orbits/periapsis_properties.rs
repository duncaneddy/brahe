//! Compute periapsis properties for an orbit.
//!
//! This example demonstrates computing periapsis velocity, distance, and altitude
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
    println!("Periapsis distance: {:.3} km", periapsis_distance / 1e3);

    // Compute periapsis altitude (generic)
    let periapsis_altitude = bh::orbits::periapsis_altitude(a, e, bh::constants::R_EARTH);
    println!("Periapsis altitude: {:.3} km", periapsis_altitude / 1e3);

    // Compute as a perigee altitude (Earth-specific)
    let perigee_altitude = bh::orbits::perigee_altitude(a, e);
    println!("Perigee altitude:   {:.3} km", perigee_altitude / 1e3);

    // Expected output:
    // Periapsis velocity: 7689.119 m/s
    // Perigee velocity:   7689.119 m/s
    // Periapsis distance: 6809.355 km
    // Periapsis altitude: 431.219 km
    // Perigee altitude:   431.219 km
}
