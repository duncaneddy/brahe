//! EOP file type comparison: Standard vs C04 formats.
//!
//! Demonstrates:
//! - Standard format (finals2000A.all) - daily updates, rapid + predicted data
//! - C04 format - long-term consistent series, less frequent updates
//! - Different use cases for each format

use brahe::eop::*;
use std::path::Path;
fn main() {
    println!("Standard Format (finals2000A.all)");
    println!("{}", "=".repeat(50));
    println!("Content: Historical + rapid + predicted data");
    println!("Updates: Daily by IERS");
    println!("Use case: Most applications requiring current EOP data");
    println!();

    // Standard format provider
    let standard_provider = CachingEOPProvider::new(
        Path::new(Path::new("./eop_data/finals.all.iau2000.txt")),
        EOPType::StandardBulletinA,
        7 * 86400,  // max_age_seconds: 7 days (frequent updates)
        false,      // auto_refresh
        true,       // interpolate
        EOPExtrapolation::Hold
    ).unwrap();

    println!("Standard provider created");
    println!("  Max age: 7 days");
    println!("  Use for: Operational applications");
    println!();

    println!("C04 Format");
    println!("{}", "=".repeat(50));
    println!("Content: Long-term consistent EOP series");
    println!("Updates: Less frequent, but highly consistent");
    println!("Use case: Historical analysis, research, long-term consistency");
    println!();

    // C04 format provider
    let c04_provider = CachingEOPProvider::new(
        Path::new(Path::new("./eop_data/eopc04.txt")),
        EOPType::C04,
        30 * 86400,  // max_age_seconds: 30 days (less frequent updates)
        false,       // auto_refresh
        true,        // interpolate
        EOPExtrapolation::Hold
    ).unwrap();

    println!("C04 provider created");
    println!("  Max age: 30 days");
    println!("  Use for: Historical analysis and research");
}
