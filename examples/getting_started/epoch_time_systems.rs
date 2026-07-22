#![allow(unused_imports)]
use brahe as bh;

fn main() {
    // Create from calendar date
    let epoch_1 = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let epoch_2 = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::GPS);
    println!("Epoch 1: {}", epoch_1);
    println!("Epoch 2: {}", epoch_2);
    println!("(Epoch 2) - (Epoch 1): {} seconds", epoch_2 - epoch_1);

    // Compare two Epochs
    let epoch_3 = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::TAI);
    let epoch_4 = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    println!("Epoch 3: {}", epoch_3);
    println!("Epoch 4: {}", epoch_4);
    println!("Epoch 3 > Epoch 4: {}", epoch_3 > epoch_4);

    // Output as MJD in time system
    println!("Epoch 1 MJD (TT): {}", epoch_1.mjd_as_time_system(bh::TimeSystem::TT));
    println!("Epoch 2 MJD (TT): {}", epoch_2.mjd_as_time_system(bh::TimeSystem::TT));
}

