//! Demonstrate thread pool configuration utilities.
//!
//! This example shows how to configure and query the global thread pool
//! used by Brahe for parallel computation operations.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Query the default number of threads
    // By default, Brahe uses 90% of available CPU cores
    let default_threads = bh::utils::get_max_threads();
    println!("Default thread count: {}", default_threads);

    // Set a specific number of threads
    bh::utils::set_num_threads(4);
    let threads_after_set = bh::utils::get_max_threads();
    println!("Thread count after setting to 4: {}", threads_after_set);

    // Set to maximum available (100% of CPU cores)
    bh::utils::set_max_threads();
    let max_threads = bh::utils::get_max_threads();
    println!("Maximum thread count: {}", max_threads);

    // Alternative: use the fun alias!
    bh::utils::set_ludicrous_speed();
    let ludicrous_threads = bh::utils::get_max_threads();
    println!("Ludicrous speed thread count: {}", ludicrous_threads);

    // The thread pool can be reconfigured at any time
    bh::utils::set_num_threads(2);
    let final_threads = bh::utils::get_max_threads();
    println!("Final thread count: {}", final_threads);

    // Note: Thread pool is used for parallelizable operations like:
    // - Computing access windows between satellites and ground locations
    // - Processing large batches of orbital calculations

    // Expected output (actual numbers vary by system):
    // Default thread count: 7
    // Thread count after setting to 4: 4
    // Maximum thread count: 8
    // Ludicrous speed thread count: 8
    // Final thread count: 2
}
