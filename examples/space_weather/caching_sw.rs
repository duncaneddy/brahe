//! Initialize Space Weather Providers using caching provider

use brahe as bh;
use bh::space_weather::SpaceWeatherExtrapolation;

fn main() {
    // Method 1: Create with custom settings
    // - Downloads to ~/.cache/brahe/
    // - Refreshes if file is older than 24 hours
    let sw_caching = bh::space_weather::CachingSpaceWeatherProvider::new(
        None,                              // cache_dir: None for default
        86400,                             // max_age: seconds (86400 = 24 hours)
        false,                             // auto_refresh: check on each query
        SpaceWeatherExtrapolation::Hold,   // extrapolation
    )
    .unwrap();
    bh::space_weather::set_global_space_weather_provider(sw_caching);

    // Method 2: Use initialize_sw() which creates a caching provider
    bh::space_weather::initialize_sw().unwrap();
}
