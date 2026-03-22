//! Get a single satellite GP record by NORAD ID from CelesTrak.
//!
//! This example demonstrates querying CelesTrak for a satellite's general
//! perturbations (GP) data using its NORAD catalog number.
//!
//! FLAGS = ["CI-ONLY"]

#[allow(unused_imports)]
use brahe as bh;
use bh::celestrak::CelestrakClient;

fn main() {
    bh::initialize_eop().unwrap();

    // Query ISS GP data by NORAD catalog number
    let client = CelestrakClient::new();
    let records = client.get_gp_by_catnr(25544).unwrap();
    let record = &records[0];

    println!("ISS GP Data:");
    println!("  Name: {}", record.object_name.as_deref().unwrap_or("Unknown"));
    println!("  NORAD ID: {}", record.norad_cat_id.unwrap_or(0));
    println!("  Epoch: {}", record.epoch.as_deref().unwrap_or("Unknown"));
    println!("  Inclination: {:.2}°", record.inclination.unwrap_or(0.0));
    println!("  RAAN: {:.2}°", record.ra_of_asc_node.unwrap_or(0.0));
    println!("  Eccentricity: {:.6}", record.eccentricity.unwrap_or(0.0));

}

