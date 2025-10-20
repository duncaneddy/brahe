//! Network coverage analysis for groundstation datasets.
//!
//! Demonstrates:
//! - Loading all providers
//! - Analyzing by latitude band
//! - Finding stations by capability

use brahe::datasets::groundstations;
use brahe::access::AccessibleLocation;

fn main() {
    println!("Groundstation Network Coverage Analysis");
    println!("{}", "=".repeat(60));

    // Load all providers
    let all_stations = groundstations::load_all_groundstations().unwrap();
    println!("\nTotal stations across all providers: {}", all_stations.len());

    // Analyze by latitude band
    let arctic: Vec<_> = all_stations.iter().filter(|s| s.lat() > 66.5).collect();
    let temperate: Vec<_> = all_stations.iter().filter(|s| s.lat() >= -66.5 && s.lat() <= 66.5).collect();
    let antarctic: Vec<_> = all_stations.iter().filter(|s| s.lat() < -66.5).collect();

    println!("\nLatitude distribution:");
    println!("  Arctic stations (>66.5°N): {}", arctic.len());
    println!("  Temperate stations: {}", temperate.len());
    println!("  Antarctic stations (<66.5°S): {}", antarctic.len());

    // Find stations by capability (check if "X" appears in frequency_bands array)
    let x_band_stations: Vec<_> = all_stations.iter()
        .filter(|s| {
            if let Some(bands) = s.properties().get("frequency_bands") {
                bands.to_string().contains("X")
            } else {
                false
            }
        })
        .collect();

    println!("\nCapability analysis:");
    println!("  X-band capable stations: {}", x_band_stations.len());

    println!("\n{}", "=".repeat(60));
    println!("Coverage analysis complete!");
}
