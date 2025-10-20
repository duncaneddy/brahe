//! Parse and validate TLE format.
//!
//! Demonstrates:
//! - Validating TLE line format
//! - Extracting epoch from TLE
//! - Converting TLE to Keplerian elements

use brahe::eop::*;
use brahe::orbits::tle::*;

fn main() {
    // Initialize EOP (required for TLE operations)
    let eop = StaticEOPProvider::from_zero();
    set_global_eop_provider(eop);

    // Valid ISS TLE
    let line1 = "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997";
    let line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003";

    // Validate TLE format
    let is_valid = validate_tle_lines(line1, line2);
    println!("TLE valid: {}", is_valid);

    // Extract epoch and convert to Keplerian elements
    let (epoch, keplerian) = keplerian_elements_from_tle(line1, line2).unwrap();
    println!("Epoch: {}", epoch);
    println!("Keplerian elements [a, e, i, raan, argp, M]:");
    println!("  a (m): {:.3}", keplerian[0]);
    println!("  e: {:.6}", keplerian[1]);
    println!("  i (rad): {:.6}", keplerian[2]);
}
