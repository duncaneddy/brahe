//! Using CachingEOPProvider for automatic refresh management.
//!
//! Demonstrates:
//! - Creating caching provider with age limits
//! - Manual refresh workflow
//! - File age monitoring
//! - Use cases for caching provider

use brahe::eop::*;
use std::path::Path;
use std::env;
use std::fs;

fn main() {
    // Create temporary directory
    let temp_dir = env::temp_dir();
    let eop_file = temp_dir.join("finals.all.iau2000.txt");

    // Download initial EOP file
    println!("Downloading initial EOP file...");
    download_standard_eop_file(eop_file.to_str().unwrap()).unwrap();

    // Create provider that refreshes files older than 7 days
    let provider = CachingEOPProvider::new(
        &eop_file,  // Use &Path directly
        EOPType::StandardBulletinA,
        7 * 86400,  // max_age_seconds: 7 days
        false,      // auto_refresh: Manual refresh only
        true,       // interpolate
        EOPExtrapolation::Hold
    ).unwrap();
    set_global_eop_provider(provider.clone());

    println!("Caching EOP provider initialized");
    println!("Max age: 7 days");
    println!("Auto-refresh: False (manual control)");

    // Check file age
    let age_seconds = provider.file_age();
    let age_days = age_seconds / 86400.0;
    println!("Current file age: {:.1} days", age_days);

    // In real application, would call provider.refresh() periodically
    println!("\nUse case: Long-running services, production systems");

    // Cleanup
    fs::remove_file(eop_file).ok();
}
