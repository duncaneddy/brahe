//! FLAGS = [IGNORE]
//! Download and save CelesTrak ephemeris data.
//!
//! Demonstrates:
//! - Downloading ephemeris to a file

use brahe::datasets::celestrak;

fn main() {
    println!("Download CelesTrak Ephemeris");
    println!("{}", "=".repeat(60));

    // Download and save ephemeris
    let output_file = "/tmp/gnss_satellites.json";
    celestrak::download_ephemeris(
        "gnss",
        output_file,
        "3le",
        "json",
    )
    .unwrap();
    println!("\nSaved GNSS ephemeris to: {}", output_file);

    println!("\n{}", "=".repeat(60));
}
