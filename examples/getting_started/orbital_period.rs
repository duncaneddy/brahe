//! Calculate the orbital period of a satellite in low Earth orbit.
//!
//! Demonstrates:
//! - Using physical constants (R_EARTH)
//! - Calculating orbital period from semi-major axis
//! - Basic orbital mechanics calculations

use brahe::constants::physical::*;
use brahe::orbits::keplerian::*;

fn main() {
    // Define the semi-major axis of a low Earth orbit (in meters)
    let a = R_EARTH + 400e3; // 400 km altitude

    // Calculate the orbital period using Kepler's third law
    let t = orbital_period(a);

    println!("Orbital Period: {:.2} minutes", t / 60.0);
}
