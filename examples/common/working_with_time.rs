//! This example demonstrates how to work with the Epoch struct in the Brahe library,
//! including creating epochs, converting between time systems, and performing
//! time arithmetic.

use brahe::{Epoch, TimeSystem};

fn main() {
    // Create an epoch from a specific date and time
    let epc = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

    // Print as ISO 8601 string
    println!("Epoch in UTC: {}", epc.isostring());
    let mjd_tai = epc.mjd_as_time_system(TimeSystem::TAI);
    println!("MJD in TAI: {}", mjd_tai);
    let jd_gps = epc.jd_as_time_system(TimeSystem::GPS);
    println!("JD in GPS: {}", jd_gps);
    let epc2 = Epoch::from_datetime(2024, 1, 2, 13, 30, 0.0, 0.0, TimeSystem::GPS);
    let delta_seconds = epc2 - epc;
    println!("Difference between epochs in seconds: {}", delta_seconds);
    let epc_utc = epc2.to_string_as_time_system(TimeSystem::UTC);
    println!("Epoch in GPS: {}", epc2);
    println!("Epoch in UTC: {}", epc_utc);
}

