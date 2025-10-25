//! Initialize EOP Providers with simpliest way possible

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    for epc in bh::TimeRange::new(
        bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC),
        bh::Epoch::from_datetime(2024, 1, 2, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC),
        3600.0
    ) {
        println!("{}", epc);
    }
}
