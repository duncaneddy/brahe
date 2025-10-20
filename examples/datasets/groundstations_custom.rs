//! Creating custom groundstation data.
//!
//! Demonstrates:
//! - Creating custom groundstation locations
//! - Adding properties to locations
//! - Combining custom with commercial networks

use brahe::datasets::groundstations;
use brahe::access::{PointLocation, AccessibleLocation};
use brahe::utils::identifiable::Identifiable;
use serde_json::json;

fn main() {
    println!("Custom Groundstation Data");
    println!("{}", "=".repeat(60));

    // Create custom groundstation
    let custom_station = PointLocation::new(
        -122.4,  // lon (degrees)
        37.8,    // lat (degrees)
        100.0    // alt (meters)
    )
    .with_name("San Francisco Custom")
    .add_property("provider", json!("Custom"))
    .add_property("frequency_bands", json!(["S", "X", "Ka"]));

    println!("\nCustom station created:");
    println!("  Location: {:.2}°N, {:.2}°E", custom_station.lat(), custom_station.lon());
    println!("  Altitude: {:.1} m", custom_station.alt());
    println!("  Name: {}", custom_station.get_name().unwrap_or("Unnamed"));
    if let Some(bands) = custom_station.properties().get("frequency_bands") {
        println!("  Bands: {}", bands);
    }

    // Combine with commercial network
    let ksat_stations = groundstations::load_groundstations("ksat").unwrap();
    let mut all_stations = vec![custom_station];
    all_stations.extend(ksat_stations.clone());

    println!("\nTotal network size: {} stations", all_stations.len());
    println!("  Commercial (KSAT): {}", ksat_stations.len());
    println!("  Custom: 1");

    println!("\n{}", "=".repeat(60));
    println!("Custom groundstation integration complete!");
}
