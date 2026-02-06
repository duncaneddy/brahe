//! Get GP data for a satellite group from CelesTrak.
//!
//! This example demonstrates the most efficient way to download GP data:
//! getting entire groups rather than individual satellites.
//!
//! FLAGS = ["CI-ONLY"]

#[allow(unused_imports)]
use brahe as bh;
use bh::celestrak::{CelestrakClient, CelestrakQuery};

fn main() {
    bh::initialize_eop().unwrap();

    // Download GP data for the Starlink group
    // This fetches all Starlink satellites in one request
    let client = CelestrakClient::new();
    let query = CelestrakQuery::gp().group("starlink");
    let records = client.query_gp(&query).unwrap();

    println!("Downloaded {} Starlink GP records", records.len());

    // Each record has orbital elements and metadata
    let record = &records[0];
    println!("\nFirst record:");
    println!("  Name: {}", record.object_name.as_deref().unwrap_or("Unknown"));
    println!("  NORAD ID: {}", record.norad_cat_id.unwrap_or(0));
    println!("  Epoch: {}", record.epoch.as_deref().unwrap_or("Unknown"));
    println!("  Inclination: {:.2}°", record.inclination.unwrap_or(0.0));
    println!("  Eccentricity: {:.6}", record.eccentricity.unwrap_or(0.0));

    // Expected output:
    // Downloaded 8647 Starlink GP records
    //
    // First record:
    //   Name: STARLINK-1008
    //   NORAD ID: 44714
    //   Epoch: 2025-11-02T10:49:16.197504
    //   Inclination: 53.05°
    //   Eccentricity: 0.000137
}
