//! Get TLE data for a satellite group from CelesTrak.
//!
//! This example demonstrates the most efficient way to download TLE data:
//! getting entire groups rather than individual satellites.
//!
//! FLAGS = ["CI-ONLY"]

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Download TLE data for the Starlink group
    // This fetches all Starlink satellites in one request
    let tles = bh::datasets::celestrak::get_tles("starlink").unwrap();

    println!("Downloaded {} Starlink TLEs", tles.len());

    // Each TLE is a tuple of (name, line1, line2)
    let (name, line1, line2) = &tles[0];
    println!("\nFirst TLE:");
    println!("  Name: {}", name);
    println!("  Line 1: {}", line1);
    println!("  Line 2: {}", line2);

    // Expected output:
    // Downloaded 8647 Starlink TLEs

    // First TLE:
    //   Name: STARLINK-1008
    //   Line 1: 1 44714U 19074B   25306.45157821  .00002551  00000+0  19011-3 0  9997
    //   Line 2: 2 44714  53.0544  37.8105 0001365  79.2826 280.8316 15.06391189329573
}
