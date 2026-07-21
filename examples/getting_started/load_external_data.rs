#[allow(unused_imports)]
use brahe as bh;

fn main() {
    // Default initializers use caching providers that automatically download new 
    // data if the local data is more than 7 days old. Only updates on initialization.
    bh::initialize_eop().unwrap();
    bh::initialize_sw().unwrap();

    // Print last date of data
    println!("EOP data available through: {}", bh::Epoch::from_mjd(bh::get_global_eop_mjd_max(), bh::TimeSystem::UTC));
    println!("SW data available through: {}", bh::Epoch::from_mjd(bh::get_global_sw_mjd_max(), bh::TimeSystem::UTC));
}

