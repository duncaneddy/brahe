//! Parse a Two-Line Element (TLE) set to extract orbital elements.
//!
//! This example demonstrates how to extract the epoch and Keplerian orbital
//! elements from a TLE set.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // ISS TLE (NORAD ID 25544)
    let line1 = "1 25544U 98067A   25302.48953544  .00013618  00000-0  24977-3 0  9995";
    let line2 = "2 25544  51.6347   1.5519 0004808 353.3325   6.7599 15.49579513535999";

    // Parse TLE to extract epoch and orbital elements
    let (epoch, elements) = bh::keplerian_elements_from_tle(line1, line2).unwrap();

    // Extract individual orbital elements
    let a = elements[0];  // Semi-major axis (m)
    let e = elements[1];  // Eccentricity
    let i = elements[2];  // Inclination (deg)
    let raan = elements[3];  // Right Ascension of Ascending Node (deg)
    let argp = elements[4];  // Argument of Periapsis (deg)
    let mean_anom = elements[5];  // Mean Anomaly (deg)

    println!("ISS Orbital Elements (Epoch: {})", epoch);
    println!("  Semi-major axis: {:.3} km", a / 1000.0);
    println!("  Eccentricity: {:.6}", e);
    println!("  Inclination: {:.4} deg", i);
    println!("  RAAN: {:.4} deg", raan);
    println!("  Arg of Perigee: {:.4} deg", argp);
    println!("  Mean Anomaly: {:.4} deg", mean_anom);

    // Expected output:
    // ISS Orbital Elements (Epoch: 2025-10-29T11:44:55.766182400 UTC)
    //   Semi-major axis: 6795.445 km
    //   Eccentricity: 0.000481
    //   Inclination: 51.6347 deg
    //   RAAN: 1.5519 deg
    //   Arg of Perigee: 353.3325 deg
    //   Mean Anomaly: 6.7599 deg
}
