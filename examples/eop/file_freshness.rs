//! Monitoring EOP file freshness.
//!
//! Demonstrates:
//! - Checking when file was loaded
//! - Monitoring file age
//! - Conditional refresh based on age

use brahe::eop::*;
use std::path::Path;
fn main() {
    let provider = CachingEOPProvider::new(
        Path::new("./eop_data/finals.all.iau2000.txt"),
        EOPType::StandardBulletinA,
        7 * 86400,  // max_age_seconds: 7 days
        false,      // auto_refresh
        true,       // interpolate
        EOPExtrapolation::Hold
    ).unwrap();
    set_global_eop_provider(provider.clone());

    // Check when file was loaded
    let file_epoch = provider.file_epoch();
    println!("EOP file loaded at: {}", file_epoch);

    // Check file age in seconds
    let age_seconds = provider.file_age();
    let age_hours = age_seconds / 3600.0;
    let age_days = age_seconds / 86400.0;

    println!("File age: {:.1} hours ({:.1} days)", age_hours, age_days);

    // Refresh if needed
    if age_days > 7.0 {
        println!("EOP data is stale, refreshing...");
        // In real application: provider.refresh().unwrap();
        println!("(Refresh would be called here)");
    } else {
        println!("EOP data is current, no refresh needed");
    }
}
