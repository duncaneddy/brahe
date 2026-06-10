#[allow(unused_imports)]
use brahe as bh;

fn main() {
    // Create from calendar date
    let _epoch_1 = bh::Epoch::from_date(2024, 1, 1, bh::TimeSystem::UTC);
    let _epoch_2 = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // // Create from ISO 8601 string
    let _epoch_3 = bh::Epoch::from_string("2024-01-01T00:00:00Z");

    // Create from String with time system
    let _epoch_4 = bh::Epoch::from_string("2024-01-00 00:00:00 GPS");

    // Current instant
    let _epoch_5 = bh::Epoch::now();
}

