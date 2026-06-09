#[allow(unused_imports)]
use brahe as bh;

fn main() {
    // Default initializers use caching providers that automatically download new 
    // data if the local data is more than 7 days old. Only updates on initialization.
    bh::initialize_eop().unwrap();
    bh::initialize_sw().unwrap();
}

