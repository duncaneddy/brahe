//! Basic groundstation data access.
//!
//! Demonstrates:
//! - Loading groundstation datasets
//! - Accessing geographic coordinates
//! - Reading metadata properties

use brahe::datasets::groundstations;
use brahe::access::AccessibleLocation;
use brahe::utils::identifiable::Identifiable;

fn main() {
    println!("Groundstations Basic Usage");
    println!("{}", "=".repeat(60));

    // Load a provider's groundstations
    let stations = groundstations::load_groundstations("ksat").unwrap();
    println!("\nLoaded {} KSAT groundstations", stations.len());

    // Access first station
    let station = &stations[0];

    // Geographic coordinates (WGS84)
    let lon = station.lon();  // Longitude in degrees
    let lat = station.lat();  // Latitude in degrees
    let alt = station.alt();  // Altitude in meters

    println!("\nFirst station:");
    println!("  Longitude: {:.6}°", lon);
    println!("  Latitude: {:.6}°", lat);
    println!("  Altitude: {:.1} m", alt);

    // Metadata properties
    let name = station.get_name().unwrap_or("Unknown");  // Station name
    let provider = station.properties().get("provider")
        .and_then(|v| v.as_str()).unwrap_or("Unknown");
    let bands_str = station.properties().get("frequency_bands")
        .map(|v| v.to_string()).unwrap_or_else(|| "[]".to_string());

    println!("  Name: {}", name);
    println!("  Provider: {}", provider);
    println!("  Frequency bands: {}", bands_str);

    println!("\n{}", "=".repeat(60));
    println!("Groundstation data access complete!");
}
