//! Recommended EOP refresh intervals for different application types.
//!
//! Demonstrates:
//! - Real-time operations (1-3 days)
//! - Batch processing (7 days)
//! - Historical analysis (30+ days)
//! - Testing/development (manual refresh)

use brahe::eop::*;
use std::path::Path;
fn main() {
    println!("Recommended EOP Refresh Intervals");
    println!("{}", "=".repeat(50));
    println!();

    // Real-time operations
    println!("1. REAL-TIME OPERATIONS");
    println!("{}", "-".repeat(50));
    let realtime_provider = CachingEOPProvider::new(
        Path::new(Path::new("./eop_data/realtime.txt")),
        EOPType::StandardBulletinA,
        2 * 86400,  // max_age_seconds: 2 days
        false,      // auto_refresh
        true,       // interpolate
        EOPExtrapolation::Hold
    ).unwrap();
    println!("  Interval: 1-3 days");
    println!("  Rationale: Balance freshness with download overhead");
    println!("  Use for: Satellite tracking, live operations");
    println!();

    // Batch processing
    println!("2. BATCH PROCESSING");
    println!("{}", "-".repeat(50));
    let batch_provider = CachingEOPProvider::new(
        Path::new(Path::new("./eop_data/batch.txt")),
        EOPType::StandardBulletinA,
        7 * 86400,  // max_age_seconds: 7 days
        false,      // auto_refresh
        true,       // interpolate
        EOPExtrapolation::Hold
    ).unwrap();
    println!("  Interval: 7 days");
    println!("  Rationale: Weekly updates sufficient for most accuracy needs");
    println!("  Use for: Scheduled analyses, mission planning");
    println!();

    // Historical analysis
    println!("3. HISTORICAL ANALYSIS");
    println!("{}", "-".repeat(50));
    let historical_provider = CachingEOPProvider::new(
        Path::new(Path::new("./eop_data/historical.txt")),
        EOPType::C04,
        30 * 86400,  // max_age_seconds: 30 days
        false,       // auto_refresh
        true,        // interpolate
        EOPExtrapolation::Hold
    ).unwrap();
    println!("  Interval: 30+ days");
    println!("  Rationale: Data rarely changes for historical periods");
    println!("  Use for: Research, long-term studies");
    println!();

    // Testing/development
    println!("4. TESTING/DEVELOPMENT");
    println!("{}", "-".repeat(50));
    println!("  Interval: No auto-refresh (manual)");
    println!("  Rationale: Control updates explicitly during development");
    println!("  Use for: Testing, debugging, development");
    println!();

    println!("Summary Table");
    println!("{}", "=".repeat(50));
    println!("Application Type      | Interval  | Use Case");
    println!("{}", "-".repeat(50));
    println!("Real-time operations  | 1-3 days  | Satellite tracking");
    println!("Batch processing      | 7 days    | Mission planning");
    println!("Historical analysis   | 30+ days  | Research");
    println!("Testing/development   | Manual    | Development");
}
