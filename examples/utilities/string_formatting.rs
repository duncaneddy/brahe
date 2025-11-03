//! Demonstrate string formatting utilities.
//!
//! This example shows how to format time durations into human-readable strings
//! using both long and short formats.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Format various time durations in long format (default)
    println!("Long format (default):");
    println!("  30 seconds: {}", bh::utils::format_time_string(30.0, false));
    println!("  90 seconds: {}", bh::utils::format_time_string(90.0, false));
    println!("  362 seconds: {}", bh::utils::format_time_string(362.0, false));
    println!("  3665 seconds: {}", bh::utils::format_time_string(3665.0, false));
    println!("  90000 seconds: {}", bh::utils::format_time_string(90000.0, false));

    // Format the same durations in short format
    println!("\nShort format:");
    println!("  30 seconds: {}", bh::utils::format_time_string(30.0, true));
    println!("  90 seconds: {}", bh::utils::format_time_string(90.0, true));
    println!("  362 seconds: {}", bh::utils::format_time_string(362.0, true));
    println!("  3665 seconds: {}", bh::utils::format_time_string(3665.0, true));
    println!("  90000 seconds: {}", bh::utils::format_time_string(90000.0, true));

    // Practical use case: format orbital period
    let orbital_period = bh::orbits::orbital_period(bh::constants::R_EARTH + 500e3);
    println!("\nLEO orbital period: {}", bh::utils::format_time_string(orbital_period, false));
    println!("LEO orbital period (short): {}", bh::utils::format_time_string(orbital_period, true));

    // Expected output:
    // Long format (default):
    //   30 seconds: 30.00 seconds
    //   90 seconds: 1 minute and 30.00 seconds
    //   362 seconds: 6 minutes and 2.00 seconds
    //   3665 seconds: 1 hour, 1 minute and 5.00 seconds
    //   90000 seconds: 1 day, 1 hour and 0.00 seconds
    //
    // Short format:
    //   30 seconds: 30s
    //   90 seconds: 1m 30s
    //   362 seconds: 6m 2s
    //   3665 seconds: 1h 1m 5s
    //   90000 seconds: 1d 1h 0m
    //
    // LEO orbital period: 1 hour, 34 minutes and 38.34 seconds
    // LEO orbital period (short): 1h 34m 38s
}
