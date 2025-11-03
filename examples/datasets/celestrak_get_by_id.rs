//! Get a single satellite TLE by NORAD ID from CelesTrak.
//!
//! This example demonstrates the cache-efficient pattern: providing the group name
//! allows brahe to use cached group data rather than making a new API request.
//!
//! FLAGS = ["CI-ONLY"]

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Get ISS TLE by NORAD ID
    // The group hint ("stations") allows brahe to check cached data first
    let (name, line1, line2) = bh::datasets::celestrak::get_tle_by_id(25544, Some("stations")).unwrap();

    // Parse TLE data to get epoch and orbital elements
    let (epoch, oe) = bh::keplerian_elements_from_tle(&line1, &line2).unwrap();

    println!("ISS TLE:");
    println!("  Name: {}", name);
    println!("  Epoch: {}", epoch);
    println!("  Inclination: {:.2}°", oe[2]);
    println!("  RAAN: {:.2}°", oe[3]);
    println!("  Eccentricity: {:.6}", oe[1]);

    // Expected output:
    // ISS TLE:
    //   Name: ISS (ZARYA)
    //   Epoch: 2025-11-02 10:09:34.283 UTC
    //   Inclination: 51.63°
    //   RAAN: 342.07°
    //   Eccentricity: 0.000497
}
