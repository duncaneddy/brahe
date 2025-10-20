//! EOP extrapolation configuration options.
//!
//! Demonstrates:
//! - Hold mode (use last known value)
//! - Zero mode (return 0.0 for out-of-range dates)
//! - Error mode (raise exception for out-of-range dates)
//! - Use cases for each mode

use brahe::eop::*;
use std::path::Path;
fn main() {
    println!("Extrapolation Configuration");
    println!("{}", "=".repeat(50));
    println!();

    // Hold last value (recommended for most applications)
    println!("1. HOLD mode (recommended)");
    println!("{}", "-".repeat(50));
    let hold_provider = CachingEOPProvider::new(
        Path::new("./eop_data/finals.all.iau2000.txt"),
        EOPType::StandardBulletinA,
        7 * 86400,  // max_age_seconds
        false,      // auto_refresh
        true,       // interpolate
        EOPExtrapolation::Hold  // Use last known value
    ).unwrap();

    println!("  Provider created with 'Hold' extrapolation");
    println!("  Behavior: Use last known value for out-of-range dates");
    println!("  Use case: Most applications, graceful degradation");
    println!();

    // Return zero for out-of-range dates
    println!("2. ZERO mode");
    println!("{}", "-".repeat(50));
    let zero_provider = CachingEOPProvider::new(
        Path::new("./eop_data/finals.all.iau2000.txt"),
        EOPType::StandardBulletinA,
        7 * 86400,  // max_age_seconds
        false,      // auto_refresh
        true,       // interpolate
        EOPExtrapolation::Zero  // Return 0.0
    ).unwrap();

    println!("  Provider created with 'Zero' extrapolation");
    println!("  Behavior: Return 0.0 for out-of-range dates");
    println!("  Use case: When zero is meaningful default");
    println!();

    // Raise error for out-of-range dates
    println!("3. ERROR mode");
    println!("{}", "-".repeat(50));
    let error_provider = CachingEOPProvider::new(
        Path::new("./eop_data/finals.all.iau2000.txt"),
        EOPType::StandardBulletinA,
        7 * 86400,  // max_age_seconds
        false,      // auto_refresh
        true,       // interpolate
        EOPExtrapolation::Error  // Raise exception
    ).unwrap();

    println!("  Provider created with 'Error' extrapolation");
    println!("  Behavior: Raise exception for out-of-range dates");
    println!("  Use case: Strict validation, fail-fast behavior");
    println!();

    println!("Recommendation: Use 'Hold' mode for most applications");
    println!("unless you have specific requirements for other modes.");
}
