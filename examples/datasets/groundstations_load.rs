//! Load groundstation data from embedded providers.
//!
//! This example demonstrates loading groundstation locations from the
//! embedded datasets. All data is offline-capable.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Load groundstations from a single provider
    let ksat_stations = bh::datasets::groundstations::load_groundstations("ksat").unwrap();
    println!("KSAT stations: {}", ksat_stations.len());

    // Load all available providers at once
    let all_stations = bh::datasets::groundstations::load_all_groundstations().unwrap();
    println!("Total stations (all providers): {}", all_stations.len());

    // List available providers
    let providers = bh::datasets::groundstations::list_providers();
    println!("\nAvailable providers: {}", providers.join(", "));

    // Load multiple specific providers
    let aws_stations = bh::datasets::groundstations::load_groundstations("aws").unwrap();
    let ssc_stations = bh::datasets::groundstations::load_groundstations("ssc").unwrap();
    let combined: Vec<_> = aws_stations
        .iter()
        .chain(ssc_stations.iter())
        .cloned()
        .collect();
    println!("\nCombined AWS + SSC: {} stations", combined.len());

    // Expected output:
    // KSAT stations: 36
    // Total stations (all providers): 96

    // Available providers: atlas, aws, ksat, leaf, ssc, viasat

    // Combined AWS + SSC: 22 stations
}
