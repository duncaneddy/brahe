//! Complete example of long-running service with EOP caching.
//!
//! Demonstrates:
//! - Service initialization with caching provider
//! - Periodic refresh in service loop
//! - Error handling for refresh failures
//! - Monitoring file age
//! - Using global EOP provider for frame transformations

use brahe::eop::*;
use std::path::Path;
use brahe::time::Epoch;
use std::thread;
use std::time::Duration;

fn main() {
    // Initialize caching provider for service
    let provider = CachingEOPProvider::new(
        Path::new("/tmp/brahe_service_eop.txt"),
        EOPType::StandardBulletinA,
        3 * 86400,  // max_age_seconds: 3 days
        false,      // auto_refresh
        true,       // interpolate
        EOPExtrapolation::Hold
    ).unwrap();

    // Set as global provider
    set_global_eop_provider(provider.clone());

    println!("Service started with EOP caching");
    println!("Initial EOP age: {:.1} days", provider.file_age() / 86400.0);

    // Service loop (limited iterations for demo)
    for cycle in 0..3 {
        println!("\n{}", "=".repeat(60));
        println!("Processing Cycle {} at {:?}", cycle + 1, std::time::SystemTime::now());
        println!("{}", "=".repeat(60));

        // Refresh EOP data at start of cycle
        match provider.refresh() {
            Ok(_) => {
                let age_days = provider.file_age() / 86400.0;
                println!("EOP refresh check (would update if needed)");
                println!("Current EOP file age: {:.1} days", age_days);
            }
            Err(e) => {
                println!("EOP refresh failed: {:?}", e);
                println!("Continuing with existing data...");
            }
        }

        // Perform calculations with current EOP data
        println!("\nProcessing epochs...");
        let mjds = [59000.0, 59050.0, 59100.0];
        for mjd in mjds.iter() {
            match (get_global_ut1_utc(*mjd), get_global_pm(*mjd)) {
                (Ok(ut1_utc), Ok((pm_x, pm_y))) => {
                    println!("  MJD {:.1}: UT1-UTC={:.6}s, PM=({:.6}, {:.6}) arcsec",
                             mjd, ut1_utc, pm_x, pm_y);
                }
                _ => {
                    println!("  Error processing MJD {}", mjd);
                }
            }
        }

        // Log current EOP file age
        let age_days = provider.file_age() / 86400.0;
        println!("\nEOP file age after cycle: {:.1} days", age_days);

        // Wait before next cycle (shortened for demo)
        if cycle < 2 {
            println!("\nWaiting for next cycle...");
            thread::sleep(Duration::from_millis(100));
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("Service demonstration complete");
    println!("{}", "=".repeat(60));
}
