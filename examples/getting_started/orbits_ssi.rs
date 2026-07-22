#![allow(unused_imports)]
use brahe as bh;

fn main() {
    // Initialize EOP
    bh::initialize_eop().unwrap();

    // Define semi-major axis and eccentricity
    let a = bh::R_EARTH + 500.0e3;   // Semi-major axis (m)
    let e = 0.01;                   // Eccentricity

    // Compute sun-synchronous inclination
    let i_ssi = bh::sun_synchronous_inclination(a, e, bh::AngleFormat::Degrees);
    println!("Sun-synchronous inclination: {:.3} degrees", i_ssi);
}

