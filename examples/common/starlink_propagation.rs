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
use brahe::celestrak::CelestrakClient;
use brahe::propagators::SGPPropagator;
use brahe::traits::SStatePropagator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::prelude::*;
use std::time::Instant;

fn main() {
    // Initialize Earth Orientation Parameters
    bh::initialize_eop().unwrap();

    // Set thread pool to use 100% of available CPU cores
    bh::utils::set_max_threads();

    // Start timing
    let start = Instant::now();

    // Download Starlink GP records and create propagators
    let client = CelestrakClient::new();
    let records = client
        .get_gp_by_group("starlink")
        .expect("Failed to fetch Starlink GP records");
    let mut propagators: Vec<_> = records
        .iter()
        .filter_map(|record| SGPPropagator::from_gp_record(record, 60.0).ok())
        .collect();

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
