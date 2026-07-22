#![allow(unused_imports)]
use brahe as bh;

fn main() {
    // Initialize EOP provider
    bh::initialize_eop().unwrap();

    // Get epochs in a range every 6 hours
    // It is inclusive of the start and exclusive of the end
    for epoch in bh::TimeRange::new(
                    bh::Epoch::from_date(2024, 1, 1, bh::TimeSystem::UTC), 
                    bh::Epoch::from_date(2024, 1, 2, bh::TimeSystem::UTC), 
                    6.0*3600.0) {
        println!("{}", epoch);
    }
}

