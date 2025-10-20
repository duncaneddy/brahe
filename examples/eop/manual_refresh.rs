//! Manual refresh workflow for CachingEOPProvider.
//!
//! Demonstrates:
//! - Creating provider with manual refresh control
//! - Periodic refresh at controlled intervals
//! - Predictable refresh timing for batch processing

use brahe::eop::*;
use std::path::Path;
use brahe::time::Epoch;
use std::thread;
use std::time::Duration;

fn main() {
    // Create provider with manual refresh
    let provider = CachingEOPProvider::new(
        Path::new("./eop_data/finals.all.iau2000.txt"),
        EOPType::StandardBulletinA,
        7 * 86400,  // max_age_seconds: 7 days
        false,      // auto_refresh: Manual only
        true,       // interpolate
        EOPExtrapolation::Hold
    ).unwrap();
    set_global_eop_provider(provider.clone());

    println!("Manual refresh workflow started");
    println!("Advantages:");
    println!("  - No performance overhead during data access");
    println!("  - Predictable refresh timing");
    println!("  - Better for batch processing and scheduled tasks");

    // Simulate processing cycles
    for cycle in 0..3 {
        println!("\nCycle {}:", cycle + 1);

        // Check file age before refresh
        let age_before = provider.file_age();
        println!("  File age before refresh: {:.1} hours", age_before / 3600.0);

        // In real application: provider.refresh() would check and update if needed
        // For demo, just show the pattern

        // Simulate processing with current EOP
        let epc = Epoch::from_datetime(
            2021, 1, 1, 0, 0, 0.0, 0.0,
            brahe::time::TimeSystem::UTC
        );
        let mjd = epc.mjd_as_time_system(brahe::time::TimeSystem::UTC);

        // Get EOP values
        let ut1_utc = get_global_ut1_utc(mjd).unwrap();
        let (pm_x, pm_y) = get_global_pm(mjd).unwrap();

        println!("  UT1-UTC: {:.6} seconds", ut1_utc);
        println!("  Polar motion: ({:.6}, {:.6}) arcsec", pm_x, pm_y);

        // Wait before next cycle (shortened for demo)
        if cycle < 2 {
            thread::sleep(Duration::from_millis(100));
        }
    }

    println!("\nManual refresh workflow complete");
}
