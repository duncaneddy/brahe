//! 
//! Parallel propagation of the Starlink constellation.
//!
//! This example demonstrates fetching all Starlink satellite TLEs from CelesTrak
//! and propagating them in parallel using Rayon with 100% of available CPU cores.
//! 
//!
// FLAGS = ["IGNORE"]

#[allow(unused_imports)]
use brahe as bh;
use brahe::traits::SStatePropagator;
use rayon::prelude::*;
use rayon::iter::IntoParallelRefMutIterator;
use std::time::Instant;

fn main() {
    // Initialize Earth Orientation Parameters
    bh::initialize_eop().unwrap();

    // Set thread pool to use 100% of available CPU cores
    bh::utils::set_max_threads();

    // Start timing
    let start = Instant::now();

    // Download Starlink TLEs and create propagators
    // Note: get_tles_as_propagators already uses parallel processing
    let mut propagators = bh::datasets::celestrak::get_tles_as_propagators("starlink", 60.0)
        .expect("Failed to fetch Starlink TLEs");

    // Propagate all satellites forward by 24 hours in parallel
    propagators.par_iter_mut().for_each(|sat| {
        let target_epoch = sat.current_epoch() + 86400.0; // 24 hours in seconds
        sat.propagate_to(target_epoch);
    });

    // Calculate elapsed time
    let elapsed = start.elapsed().as_secs_f64();

    // Print results
    println!(
        "Propagated {} Starlink satellites to one orbit in {}.",
        propagators.len(),
        bh::utils::format_time_string(elapsed, false)
    );
}
