//! Get satellite GP data by name from CelesTrak.
//!
//! This example demonstrates searching for satellites by name using the
//! CelestrakQuery builder's name_search method.
//!
//! FLAGS = ["CI-ONLY"]

#[allow(unused_imports)]
use brahe as bh;
use bh::celestrak::CelestrakClient;

fn main() {
    bh::initialize_eop().unwrap();

    // Search by name
    let client = CelestrakClient::new();
    let records = client.get_gp_by_name("ISS").unwrap();

    println!("Found {} results for 'ISS'", records.len());
    for record in records.iter().take(5) {
        println!(
            "  {} (NORAD ID: {})",
            record.object_name.as_deref().unwrap_or("Unknown"),
            record.norad_cat_id.unwrap_or(0)
        );
    }

    // The first result should be ISS (ZARYA)
    let iss = &records[0];
    println!("\nISS GP Data:");
    println!("  Name: {}", iss.object_name.as_deref().unwrap_or("Unknown"));
    println!("  NORAD ID: {}", iss.norad_cat_id.unwrap_or(0));
    println!("  Epoch: {}", iss.epoch.as_deref().unwrap_or("Unknown"));
    println!("  Inclination: {:.2}°", iss.inclination.unwrap_or(0.0));
    println!("  Eccentricity: {:.6}", iss.eccentricity.unwrap_or(0.0));

}

