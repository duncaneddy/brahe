//! This example demonstrates ways to initialize EOP from files using Brahe.

use brahe as bh;

fn main() {
    // Method 1: Initialize from Caching EOP Provider -> Internally caches data to ~/.cache/brahe/eop
    let provider = bh::eop::CachingEOPProvider::new(
        None,                                   // filepath (None for default cache location)
        bh::eop::EOPType::StandardBulletinA,   // eop_type
        7 * 86400,                             // max_age_seconds - Maximum age of file before refreshing
        false,                                 // auto_refresh - Check staleness of every access
        true,                                  // interpolate
        bh::eop::EOPExtrapolation::Hold       // extrapolate
    ).unwrap();
    bh::eop::set_global_eop_provider(provider);

    // Method 2: Initialize from Caching EOP Provider with custom location
    let cache_dir = bh::utils::get_brahe_cache_dir().unwrap();
    let custom_filepath = std::path::Path::new(&cache_dir).join("my_eop.txt");
    let provider_custom = bh::eop::CachingEOPProvider::new(
        Some(&custom_filepath),                        // Replace with desired file path to load / save from
        bh::eop::EOPType::StandardBulletinA,           // eop_type
        7 * 86400,                                     // max_age_seconds - Maximum age of file before refreshing
        false,                                         // auto_refresh - Check staleness of every access
        true,                                          // interpolate
        bh::eop::EOPExtrapolation::Hold               // extrapolate
    ).unwrap();
    bh::eop::set_global_eop_provider(provider_custom);
}
