//! Subtract Epoch instances to compute time differences

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Create two epochs
    let epc1 = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let epc2 = bh::Epoch::from_datetime(2024, 1, 2, 13, 1, 1.0, 0.0, bh::TimeSystem::UTC);

    // Compute time difference in seconds
    let dt = epc2 - epc1;
    println!("Time difference: {:.1} seconds", dt);
    // Time difference: 90061.0 seconds

    // You can also subtract a float (in seconds) from an Epoch to get a new Epoch
    let epc = bh::Epoch::from_datetime(2024, 6, 15, 10, 30, 0.0, 0.0, bh::TimeSystem::UTC);
    let epc_minus_hour = epc - 3600.0;
    println!("Minus 1 hour: {}", epc_minus_hour);
    // Minus 1 hour: 2024-06-15 09:30:00.000 UTC

    // You can also update an Epoch in-place by subtracting seconds
    let mut epc = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    epc -= 61.0; // Subtract 61 seconds
    println!("In-place minus 61 seconds: {}", epc);
    // In-place minus 61 seconds: 2023-12-31 23:58:59.000 UTC
}
