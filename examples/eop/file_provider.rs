//! Using FileEOPProvider with downloaded EOP data.
//!
//! Demonstrates:
//! - Downloading EOP files from IERS
//! - Loading EOP from file
//! - Setting global file-based provider
//! - Interpolation and extrapolation options

use brahe::eop::*;
use std::path::Path;
use std::fs;
use std::env;

fn main() {
    // Create temporary directory for EOP data
    let temp_dir = env::temp_dir();
    let eop_file = temp_dir.join("finals.all.iau2000.txt");

    // Download latest EOP file
    println!("Downloading EOP file...");
    download_standard_eop_file(eop_file.to_str().unwrap()).unwrap();
    println!("Downloaded to {:?}", eop_file);

    // Load from file with interpolation
    let provider = FileEOPProvider::from_file(
        &eop_file,  // Use &Path directly
        true,  // interpolate
        EOPExtrapolation::Hold
    ).unwrap();
    set_global_eop_provider(provider);

    println!("File EOP provider initialized");
    println!("Use case: Production applications with current EOP data");

    // Cleanup
    fs::remove_file(eop_file).ok();
}
