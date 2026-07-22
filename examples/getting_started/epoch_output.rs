#![allow(unused_imports)]
use brahe as bh;

fn main() {
    // Create an Epoch
    let epoch = bh::Epoch::from_date(2024, 1, 1, bh::TimeSystem::UTC);

    // Output as iso string
    println!("Epoch as ISO 8601 string: {}", epoch);

    // Output as Modified Julian Date
    println!("Epoch as Modified Julian Date: {}", epoch.mjd());

    // Output as calendar date
    println!("Epoch as calendar date tuple: {:?}", epoch.to_datetime());

    // Output as UNIX timestamp
    println!("Epoch as UNIX timestamp: {}", epoch.unix_timestamp());

    // Output as GPS time
    println!("Epoch as GPS time: {:?}", epoch.gps_date());
}

