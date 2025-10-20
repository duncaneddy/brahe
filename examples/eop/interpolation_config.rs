//! EOP interpolation configuration options.
//!
//! Demonstrates:
//! - Interpolation enabled (smooth data between points)
//! - Interpolation disabled (step function)
//! - Use cases for each mode

use brahe::eop::*;
use std::path::Path;
fn main() {
    println!("Interpolation Configuration");
    println!("{}", "=".repeat(50));
    println!();

    // With interpolation (recommended for most applications)
    println!("1. Interpolation ENABLED (recommended)");
    println!("{}", "-".repeat(50));
    let interpolated_provider = CachingEOPProvider::new(
        Path::new("./eop_data/finals.all.iau2000.txt"),
        EOPType::StandardBulletinA,
        7 * 86400,  // max_age_seconds
        false,      // auto_refresh
        true,       // interpolate: Smooth interpolation
        EOPExtrapolation::Hold
    ).unwrap();

    println!("  Provider created with interpolation");
    println!("  Benefits:");
    println!("    - Smooth EOP values between tabulated points");
    println!("    - More accurate for high-precision applications");
    println!("    - Recommended for most use cases");
    println!();

    // Without interpolation (step function between points)
    println!("2. Interpolation DISABLED");
    println!("{}", "-".repeat(50));
    let step_provider = CachingEOPProvider::new(
        Path::new("./eop_data/finals.all.iau2000.txt"),
        EOPType::StandardBulletinA,
        7 * 86400,  // max_age_seconds
        false,      // auto_refresh
        false,      // interpolate: No interpolation
        EOPExtrapolation::Hold
    ).unwrap();

    println!("  Provider created without interpolation");
    println!("  Behavior:");
    println!("    - Step function between tabulated points");
    println!("    - Faster lookups (no interpolation computation)");
    println!("    - Use when performance is critical and precision less so");
    println!();

    println!("Recommendation: Enable interpolation unless you have");
    println!("specific performance requirements that justify disabling it.");
}
