//! FLAGS = [IGNORE]
//! Get ephemeris data from CelesTrak.
//!
//! Demonstrates:
//! - Getting GNSS satellite TLE data

use brahe::datasets::celestrak;

fn main() {
    println!("Get GNSS Ephemeris from CelesTrak");
    println!("{}", "=".repeat(60));

    // Get ephemeris data for GNSS constellation
    let ephemeris = celestrak::get_ephemeris("gnss").unwrap();
    println!("\nRetrieved {} GNSS satellites", ephemeris.len());

    // Display first few satellites
    for (i, (name, line1, line2)) in ephemeris.iter().take(3).enumerate() {
        println!("\nSatellite {}: {}", i + 1, name);
        println!("  {}", line1);
        println!("  {}", line2);
    }

    println!("\n{}", "=".repeat(60));
}
