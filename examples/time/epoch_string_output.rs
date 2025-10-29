//! Convert Epoch instances to string representations

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Create an epoch
    let epc = bh::Epoch::from_datetime(2024, 6, 15, 14, 30, 45.123456789, 0.0, bh::TimeSystem::UTC);

    // Default string representation
    println!("Default: {}", epc);
    // String: 2024-06-15 14:30:45.123 UTC

    // Explicit string conversion (same as default in Rust)
    println!("String: {}", epc.to_string());
    // String: 2024-06-15 14:30:45.123 UTC

    // Debug representation
    println!("Debug: {:?}", epc);
    // Debug: Epoch<2460477, 9082, 123456788.98545027, 0, UTC>

    // Get string in a different time system
    println!("TT: {}", epc.to_string_as_time_system(bh::TimeSystem::TT));
    // TT: 2024-06-15 14:31:54.307 TT

    // Get as ISO 8601 formatted string
    println!("ISO 8601: {}", epc.isostring());
    // ISO 8601: 2024-06-15T14:30:45Z

    // Get as ISO 8601 with different number of decimal places
    println!("ISO 8601 (0 decimal places): {}", epc.isostring_with_decimals(0));
    println!("ISO 8601 (3 decimal places): {}", epc.isostring_with_decimals(3));
    println!("ISO 8601 (6 decimal places): {}", epc.isostring_with_decimals(6));
    // ISO 8601 (0 decimal places): 2024-06-15T14:30:45Z
    // ISO 8601 (3 decimal places): 2024-06-15T14:30:45.123Z
    // ISO 8601 (6 decimal places): 2024-06-15T14:30:45.123456Z
}
