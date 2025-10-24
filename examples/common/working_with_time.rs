//! This example demonstrates how to work with the Epoch struct in the Brahe library,
//! including creating epochs, converting between time systems, and performing
//! time arithmetic.

use brahe::{Epoch, TimeSystem};

fn main() {
    // Create an epoch from a specific date and time
    let epc = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

    // Print as ISO 8601 string
    println!("Epoch in UTC: {}", epc.isostring());
    // Output:
    // Epoch in UTC: 2024-01-01T12:00:00Z

    // Get the Modified Julian Date (MJD) in different time systems
    let mjd_tai = epc.mjd_as_time_system(TimeSystem::TAI);
    println!("MJD in TAI: {}", mjd_tai);
    // Output:
    // MJD in TAI: 60310.50042824074

    // Get the time as a Julian Date (JD) in GPS time system
    let jd_gps = epc.jd_as_time_system(TimeSystem::GPS);
    println!("JD in GPS: {}", jd_gps);
    // Output:
    // JD in GPS: 2460311.000208333

    // Take the difference between two epochs in different time systems
    let epc2 = Epoch::from_datetime(2024, 1, 2, 13, 30, 0.0, 0.0, TimeSystem::GPS);
    let delta_seconds = epc2 - epc;
    println!("Difference between epochs in seconds: {}", delta_seconds);
    // Output:
    // Difference between epochs in seconds: 91782.0

    // Get the epoch as a string in different time systems
    let epc_utc = epc2.to_string_as_time_system(TimeSystem::UTC);
    println!("Epoch in GPS: {}", epc2);
    println!("Epoch in UTC: {}", epc_utc);
    // Outputs:
    // Epoch in GPS: 2024-01-02 13:30:00.000 GPS
    // Epoch in UTC: 2024-01-02 13:29:42.000 UTC
}
