#[allow(unused_imports)]
use brahe as bh;

fn main() {
    // Initialize EOP provider
    bh::initialize_eop().unwrap();

    // Create from calendar date
    let epoch = bh::Epoch::from_date(2024, 1, 1, bh::TimeSystem::UTC);

    // Add time (in seconds)
    let epoch_plus_1_day = epoch + 86400.0;    // Add one day
    let _epoch_plus_1_hour = epoch + 3600.0;    // Add one hour
    let _epoch_plus_1_ns = epoch + 1e-9;        // Add one nanosecond

    // Subtract time (in seconds)
    let _epoch_minus_1_day = epoch - 86400.0;   // Subtract one day
    let _epoch_minus_1_hour = epoch - 3600.0;   // Subtract one hour
    let _epoch_minus_1_ns = epoch - 1e-9;       // Subtract one nanosecond

    // Get difference between two epochs (in seconds)
    let difference = epoch_plus_1_day - epoch; // Should be 86400 seconds
    println!("Difference in seconds: {:.2}", difference);

    // Comparison operations
    println!("epoch < epoch_plus_1_day: {}", epoch < epoch_plus_1_day);
    println!("epoch == epoch_minus_1_day: {}", epoch == _epoch_minus_1_day);
    println!("epoch > epoch_minus_1_day: {}", epoch > _epoch_minus_1_day);
}

