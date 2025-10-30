//! Extract the epoch from a Two-Line Element (TLE) set.
//!
//! This example demonstrates how to extract just the epoch timestamp from a TLE
//! without parsing the full orbital elements.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // ISS TLE (NORAD ID 25544)
    let line1 = "1 25544U 98067A   25302.48953544  .00013618  00000-0  24977-3 0  9995";
    let _line2 = "2 25544  51.6347   1.5519 0004808 353.3325   6.7599 15.49579513535999";

    // Extract epoch from line 1 (epoch is encoded in line 1 only)
    let epoch = bh::epoch_from_tle(line1).unwrap();

    println!("TLE Epoch: {}", epoch);
    println!("Time System: {:?}", epoch.time_system);
    println!("Julian Date: {:.10}", epoch.jd());
    println!("Modified Julian Date: {:.10}", epoch.mjd());

    // Convert to datetime components
    let dt = epoch.to_datetime();
    println!("\nDatetime Components:");
    println!("  Year: {}", dt.0);
    println!("  Month: {}", dt.1);
    println!("  Day: {}", dt.2);
    println!("  Hour: {}", dt.3);
    println!("  Minute: {}", dt.4);
    println!("  Second: {:.6}", dt.5);

    // Expected output:
    // TLE Epoch: 2025-10-29T11:44:55.766182400 UTC
    // Time System: UTC
    // Julian Date: 2460974.9895780
    // Modified Julian Date: 60974.4895780
    //
    // Datetime Components:
    //   Year: 2025
    //   Month: 10
    //   Day: 29
    //   Hour: 11
    //   Minute: 44
    //   Second: 55.766182
}
