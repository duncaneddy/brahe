//! Working with time systems and epochs.
//!
//! Demonstrates:
//! - Creating epochs from date/time
//! - Converting between time systems
//! - Time arithmetic operations

use brahe::time::*;

fn main() {
    // Create an epoch from a specific date and time
    let epc = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

    // Convert between time systems
    let mjd_utc = epc.mjd_as_time_system(TimeSystem::UTC);
    let mjd_tai = epc.mjd_as_time_system(TimeSystem::TAI);

    println!("MJD (UTC): {}", mjd_utc);
    println!("MJD (TAI): {}", mjd_tai);

    // Time arithmetic
    let future_epc = epc + 3600.0; // Add 3600 seconds (1 hour)
    let time_diff = future_epc - epc; // Difference in seconds

    println!("Time difference: {} seconds", time_diff);
}
