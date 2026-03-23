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

}

