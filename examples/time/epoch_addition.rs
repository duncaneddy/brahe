//! Add time duration to Epoch instances

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Create an epoch
    let epc = bh::Epoch::from_datetime(2025, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    println!("Original epoch: {}", epc);

    // You can add time in seconds to an Epoch and get a new Epoch back

    // Add 1 hour (3600 seconds)
    let epc_plus_hour = epc + 3600.0;
    println!("Plus 1 hour: {}", epc_plus_hour);

    // Add 1 day (86400 seconds)
    let epc_plus_day = epc + 86400.0;
    println!("Plus 1 day: {}", epc_plus_day);

    // You can also do in-place addition

    // Add 1 second in-place
    let mut epc = epc;
    epc += 1.0;
    println!("In-place plus 1 second: {}", epc);

    // Add 1 millisecond in-place
    epc += 0.001;
    println!("In-place plus 1 millisecond: {}", epc);
}

