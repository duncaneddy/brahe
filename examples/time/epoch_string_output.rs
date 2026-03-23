//! Convert Epoch instances to string representations

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Create an epoch
    let epc = bh::Epoch::from_datetime(2024, 6, 15, 14, 30, 45.123456789, 0.0, bh::TimeSystem::UTC);

    // Default string representation
    println!("Default: {}", epc);

    // Explicit string conversion (same as default in Rust)
    println!("String: {}", epc.to_string());

    // Debug representation
    println!("Debug: {:?}", epc);

    // Get string in a different time system
    println!("TT: {}", epc.to_string_as_time_system(bh::TimeSystem::TT));

    println!("ISO 8601: {}", epc.isostring());

    println!("ISO 8601 (0 decimal places): {}", epc.isostring_with_decimals(0));
    println!("ISO 8601 (3 decimal places): {}", epc.isostring_with_decimals(3));
    println!("ISO 8601 (6 decimal places): {}", epc.isostring_with_decimals(6));
}

