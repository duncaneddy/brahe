//! Create Epoch instances from date-time strings

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // The string can be an ISO 8601 format
    let epc1 = bh::Epoch::from_string("2025-01-02T04:56:54.123Z").unwrap();
    println!("ISO 8601: {}", epc1);

    // It can be a simple space-separated format with a time system
    let epc2 = bh::Epoch::from_string("2024-06-15 14:30:45.500 GPS").unwrap();
    println!("Simple format: {}", epc2);

    // It can be a datetime without a time system (defaults to UTC)
    let epc3 = bh::Epoch::from_string("2023-12-31 23:59:59").unwrap();
    println!("Datetime without time system: {}", epc3);

    // Or it can just be a date
    let epc4 = bh::Epoch::from_string("2022-07-04").unwrap();
    println!("Date only: {}", epc4);
}
