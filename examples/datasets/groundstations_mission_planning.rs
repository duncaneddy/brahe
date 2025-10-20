//! Mission planning with groundstation networks.
//!
//! Demonstrates:
//! - Evaluating multiple providers
//! - Filtering by capability requirements
//! - Assessing geographic distribution

use brahe::datasets::groundstations;
use brahe::access::AccessibleLocation;

fn main() {
    println!("Groundstation Mission Planning");
    println!("{}", "=".repeat(60));

    // Requirements
    let required_bands = vec!["S", "X"];
    println!("\nMission requirements:");
    println!("  Required bands: {}", required_bands.join(", "));
    println!("  Preferred: Arctic coverage");

    // Evaluate providers
    let providers = groundstations::list_providers();
    println!("\n{}", "-".repeat(60));
    println!("Provider Evaluation:");
    println!("{}", "-".repeat(60));

    for provider in providers {
        let stations = groundstations::load_groundstations(&provider).unwrap();

        // Filter by capability (check if all required bands appear in frequency_bands array)
        let capable: Vec<_> = stations.iter()
            .filter(|s| {
                if let Some(bands) = s.properties().get("frequency_bands") {
                    let bands_str = bands.to_string();
                    required_bands.iter().all(|band| bands_str.contains(band))
                } else {
                    false
                }
            })
            .collect();

        // Check geographic distribution
        let arctic_count = capable.iter().filter(|s| s.lat() > 60.0).count();

        println!("\n{}", provider.to_uppercase());
        println!("  Total stations: {}", stations.len());
        println!("  Capable stations: {}", capable.len());
        println!("  Arctic coverage (>60°N): {}", arctic_count);
    }

    println!("\n{}", "=".repeat(60));
    println!("Mission planning analysis complete!");
}
