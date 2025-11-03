//! Demonstrate cache directory management utilities.
//!
//! This example shows how to get the cache directory paths used by Brahe
//! for storing downloaded data such as EOP and TLE files.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Get main cache directory
    let cache_dir = bh::utils::get_brahe_cache_dir().unwrap();
    println!("Main cache directory: {}", cache_dir);

    // Get cache subdirectory for EOP data
    let eop_cache = bh::utils::get_eop_cache_dir().unwrap();
    println!("EOP cache directory: {}", eop_cache);

    // Get cache subdirectory for CelesTrak data
    let celestrak_cache = bh::utils::get_celestrak_cache_dir().unwrap();
    println!("CelesTrak cache directory: {}", celestrak_cache);

    // Get a custom subdirectory within the cache
    let custom_cache = bh::utils::get_brahe_cache_dir_with_subdir(Some("custom_data")).unwrap();
    println!("Custom cache subdirectory: {}", custom_cache);

    // Note: All directories are automatically created if they don't exist
    // You can override the default location by setting the BRAHE_CACHE
    // environment variable

    // Expected output (paths will vary by system):
    // Main cache directory: /home/USER/.cache/brahe
    // EOP cache directory: /home/USER/.cache/brahe/eop
    // CelesTrak cache directory: /home/USER/.cache/brahe/celestrak
    // Custom cache subdirectory: /home/USER/.cache/brahe/custom_data
}
