//! Compute apoapsis properties for an orbit.
//!
//! This example demonstrates computing apoapsis velocity, distance, and altitude
//! for a given orbit, including Earth-specific apogee functions.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Define orbit parameters
    let a = bh::constants::R_EARTH + 500.0e3; // Semi-major axis (m)
    let e = 0.01; // Eccentricity

    // Compute apoapsis velocity (generic)
    let apoapsis_velocity = bh::orbits::apoapsis_velocity(a, e, bh::constants::GM_EARTH);
    println!("Apoapsis velocity: {:.3} m/s", apoapsis_velocity);

    // Compute as an apogee velocity (Earth-specific)
    let apogee_velocity = bh::orbits::apogee_velocity(a, e);
    println!("Apogee velocity:   {:.3} m/s", apogee_velocity);

    // Compute apoapsis distance
    let apoapsis_distance = bh::orbits::apoapsis_distance(a, e);
    println!("Apoapsis distance: {:.3} km", apoapsis_distance / 1e3);

    // Compute apoapsis altitude (generic)
    let apoapsis_altitude = bh::orbits::apoapsis_altitude(a, e, bh::constants::R_EARTH);
    println!("Apoapsis altitude: {:.3} km", apoapsis_altitude / 1e3);

    // Compute as an apogee altitude (Earth-specific)
    let apogee_altitude = bh::orbits::apogee_altitude(a, e);
    println!("Apogee altitude:   {:.3} km", apogee_altitude / 1e3);

    // Expected output:
    // Apoapsis velocity: 7536.859 m/s
    // Apogee velocity:   7536.859 m/s
    // Apoapsis distance: 6946.918 km
    // Apoapsis altitude: 568.781 km
    // Apogee altitude:   568.781 km
}
