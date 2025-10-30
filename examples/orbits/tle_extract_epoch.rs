//! Extract just the epoch from a TLE.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // ISS TLE (NORAD ID 25544)
    let line1 = "1 25544U 98067A   25302.48953544  .00013618  00000-0  24977-3 0  9995";

    // Extract epoch from line 1 (epoch is encoded in line 1 only)
    let epoch = bh::epoch_from_tle(line1).unwrap();

    println!("TLE Epoch: {}", epoch);
    println!("Time System: {:?}", epoch.time_system);

    // Expected output:
    // TLE Epoch: 2025-10-29 11:44:55.862 UTC
    // Time System: UTC
}
