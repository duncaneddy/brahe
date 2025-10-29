//! Create Epoch instances from datetime components

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Create epoch from date only (midnight)
    let epc1 = bh::Epoch::from_date(2024, 1, 1, bh::TimeSystem::UTC);
    println!("Date only: {}", epc1);

    // Create epoch from full datetime components
    let epc2 = bh::Epoch::from_datetime(2024, 6, 15, 14, 30, 45.5, 0.0, bh::TimeSystem::UTC);
    println!("Full datetime: {}", epc2);

    // Create epoch with different time system
    let epc3 = bh::Epoch::from_datetime(2024, 12, 25, 18, 0, 0.0, 0.0, bh::TimeSystem::GPS);
    println!("GPS time system: {}", epc3);
}
