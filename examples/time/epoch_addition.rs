//! Add time duration to Epoch instances

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Create an epoch
    let epc = bh::Epoch::from_datetime(2025, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    println!("Original epoch: {}", epc);
    // Original epoch: 2025-01-01 12:00:00.000 UTC

    // You can add time in seconds to an Epoch and get a new Epoch back

    // Add 1 hour (3600 seconds)
    let epc_plus_hour = epc + 3600.0;
    println!("Plus 1 hour: {}", epc_plus_hour);
    // Plus 1 hour: 2025-01-01 13:00:00.000 UTC

    // Add 1 day (86400 seconds)
    let epc_plus_day = epc + 86400.0;
    println!("Plus 1 day: {}", epc_plus_day);
    // Plus 1 day: 2025-01-02 12:00:00.000 UTC

    // You can also do in-place addition

    // Add 1 second in-place
    let mut epc = epc;
    epc += 1.0;
    println!("In-place plus 1 second: {}", epc);
    // In-place plus 1 second: 2025-01-01 12:00:01.000 UTC

    // Add 1 millisecond in-place
    epc += 0.001;
    println!("In-place plus 1 millisecond: {}", epc);
    // In-place plus 1 millisecond: 2025-01-01 12:00:01.001 UTC
}
