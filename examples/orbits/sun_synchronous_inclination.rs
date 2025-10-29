//! Compute sun-synchronous inclination for an orbit.
//!
//! This example demonstrates computing the required inclination for a
//! sun-synchronous orbit at various altitudes and eccentricities. Sun-synchronous
//! orbits maintain a constant angle relative to the Sun, useful for Earth
//! observation missions requiring consistent lighting conditions.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Example 1: Typical sun-synchronous LEO at 800 km altitude
    let a_leo = bh::constants::R_EARTH + 800e3; // Semi-major axis
    let e_leo = 0.0; // Circular orbit

    let inc_leo_deg = bh::orbits::sun_synchronous_inclination(a_leo, e_leo, bh::constants::AngleFormat::Degrees);
    let inc_leo_rad = bh::orbits::sun_synchronous_inclination(a_leo, e_leo, bh::constants::AngleFormat::Radians);

    println!("Sun-synchronous LEO (800 km, circular):");
    println!("  Inclination: {:.3} degrees", inc_leo_deg);
    println!("  Inclination: {:.6} radians", inc_leo_rad);

    // Example 2: Different altitudes
    let altitudes = [500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]; // km
    println!("\nSun-synchronous inclination vs altitude (circular orbits):");
    for alt_km in altitudes.iter() {
        let a = bh::constants::R_EARTH + alt_km * 1e3;
        let inc = bh::orbits::sun_synchronous_inclination(a, 0.0, bh::constants::AngleFormat::Degrees);
        println!("  {:4} km: {:.3} deg", *alt_km as i32, inc);
    }

    // Example 3: Effect of eccentricity
    let a_fixed = bh::constants::R_EARTH + 700e3;
    let eccentricities = [0.0, 0.001, 0.005, 0.01, 0.02];

    println!("\nSun-synchronous inclination vs eccentricity (700 km orbit):");
    for e in eccentricities.iter() {
        let inc = bh::orbits::sun_synchronous_inclination(a_fixed, *e, bh::constants::AngleFormat::Degrees);
        println!("  e = {:.3}: {:.3} deg", e, inc);
    }

    // Example 4: Practical mission example (Landsat-like)
    let a_landsat = bh::constants::R_EARTH + 705e3;
    let e_landsat = 0.0001;
    let inc_landsat = bh::orbits::sun_synchronous_inclination(a_landsat, e_landsat, bh::constants::AngleFormat::Degrees);

    println!("\nLandsat-like orbit (705 km, nearly circular):");
    println!("  Inclination: {:.3} deg", inc_landsat);
    println!("  Period: {:.3} min", bh::orbits::orbital_period(a_landsat) / 60.0);

    // Expected output:
    // Sun-synchronous LEO (800 km, circular):
    //   Inclination: 98.603 degrees
    //   Inclination: 1.720948 radians

    // Sun-synchronous inclination vs altitude (circular orbits):
    //    500 km: 97.402 deg
    //    600 km: 97.788 deg
    //    700 km: 98.188 deg
    //    800 km: 98.603 deg
    //    900 km: 99.033 deg
    //   1000 km: 99.479 deg

    // Sun-synchronous inclination vs eccentricity (700 km orbit):
    //   e = 0.000: 98.188 deg
    //   e = 0.001: 98.188 deg
    //   e = 0.005: 98.187 deg
    //   e = 0.010: 98.186 deg
    //   e = 0.020: 98.181 deg

    // Landsat-like orbit (705 km, nearly circular):
    //   Inclination: 98.208 deg
    //   Period: 98.878 min
}
