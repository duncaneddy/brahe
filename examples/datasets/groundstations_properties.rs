//! Access groundstation location properties.
//!
//! This example demonstrates how to access geographic coordinates and
//! metadata properties from groundstation locations.

#[allow(unused_imports)]
use brahe as bh;
use bh::utils::Identifiable;

fn main() {
    bh::initialize_eop().unwrap();

    // Load KSAT groundstations
    let stations = bh::datasets::groundstations::load_groundstations("ksat").unwrap();

    // Access the first station
    let station = &stations[0];

    // Geographic coordinates (degrees and meters)
    let name = station.get_name().unwrap_or("Unknown");
    println!("Station: {}", name);
    println!("Latitude: {:.4}°", station.lat());
    println!("Longitude: {:.4}°", station.lon());
    println!("Altitude: {:.1} m", station.alt());

    // Show all stations with their locations
    println!("\n{} KSAT Stations:", stations.len());
    for (i, gs) in stations.iter().enumerate() {
        let gs_name = gs.get_name().unwrap_or("Unknown");
        println!(
            "{:2}. {:30} ({:7.3}°, {:8.3}°)",
            i + 1,
            gs_name,
            gs.lat(),
            gs.lon()
        );
    }

}

