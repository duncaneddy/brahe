//! Initialize Space Weather Providers with simplest way possible

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    // Initialize with default caching provider (will download data as needed)
    bh::space_weather::initialize_sw().unwrap();
}
