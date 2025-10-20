//! Combining multiple groundstation networks.
//!
//! Demonstrates:
//! - Loading multiple providers
//! - Merging networks into a single vector
//! - Using combined networks for analysis

use brahe::datasets::groundstations;

fn main() {
    println!("Combining Groundstation Networks");
    println!("{}", "=".repeat(60));

    // Load multiple providers
    let primary = groundstations::load_groundstations("ksat").unwrap();
    let backup = groundstations::load_groundstations("ssc").unwrap();

    println!("\nPrimary network (KSAT): {} stations", primary.len());
    println!("Backup network (SSC): {} stations", backup.len());

    // Combine into single network
    let mut combined = primary.clone();
    combined.extend(backup);

    println!("Combined network: {} stations", combined.len());

    // Analyze combined coverage
    let arctic_count = combined.iter().filter(|s| s.lat() > 66.5).count();
    println!("Arctic coverage: {} stations", arctic_count);

    println!("\n{}", "=".repeat(60));
    println!("Network combination complete!");
}
