//! Auto-refresh mode for CachingEOPProvider.
//!
//! Demonstrates:
//! - Automatic file age checking on every access
//! - Guaranteed data freshness
//! - Suitable for long-running services
//! - Performance considerations

use brahe::eop::*;
use std::path::Path;
use brahe::time::Epoch;

fn main() {
    // Provider checks file age on every access
    let provider = CachingEOPProvider::new(
        Path::new("./eop_data/finals.all.iau2000.txt"),
        EOPType::StandardBulletinA,
        24 * 3600,  // max_age_seconds: 24 hours
        true,       // auto_refresh: Check on every access
        true,       // interpolate
        EOPExtrapolation::Hold
    ).unwrap();
    set_global_eop_provider(provider.clone());

    println!("Auto-refresh mode enabled");
    println!("\nAdvantages:");
    println!("  - Guaranteed data freshness");
    println!("  - Simpler application code");
    println!("  - Suitable for long-running services");

    println!("\nConsiderations:");
    println!("  - Small performance overhead on each access (microseconds)");
    println!("  - May trigger downloads during time-critical operations");
    println!("  - Better suited for applications where data access is not in tight loops");

    // EOP data automatically stays current
    let epc = Epoch::from_datetime(
        2021, 1, 1, 0, 0, 0.0, 0.0,
        brahe::time::TimeSystem::UTC
    );
    let mjd = epc.mjd_as_time_system(brahe::time::TimeSystem::UTC);

    // Each access checks file age automatically
    let ut1_utc = get_global_ut1_utc(mjd).unwrap();
    let (pm_x, pm_y) = get_global_pm(mjd).unwrap();

    println!("\nCurrent EOP values at MJD {:.2}:", mjd);
    println!("  UT1-UTC: {:.6} seconds", ut1_utc);
    println!("  Polar motion: ({:.6}, {:.6}) arcsec", pm_x, pm_y);
    println!("\nFile age: {:.1} hours", provider.file_age() / 3600.0);
}
