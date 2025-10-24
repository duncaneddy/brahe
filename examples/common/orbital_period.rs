//! This example demonstrates how to calculate the orbital period of a satellite
//! given its semi-major axis using the Brahe library.

#[allow(unused_imports)]
use brahe::{R_EARTH, orbital_period};

fn main() {
    // Define the semi-major axis of a low Earth orbit (in meters)
    let semi_major_axis = R_EARTH + 400e3; // 400 km altitude

    // Calculate the orbital period
    let period = orbital_period(semi_major_axis); 

    println!("Orbital Period: {:.2} minutes", period / 60.0);
    // Outputs:
    // Orbital Period: 92.56 minutes
}
