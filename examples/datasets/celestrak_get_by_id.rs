//! Get a single satellite GP record by NORAD ID from CelesTrak.
//!
//! This example demonstrates querying CelesTrak for a satellite's general
//! perturbations (GP) data using its NORAD catalog number.
//!
//! FLAGS = ["CI-ONLY"]

#[allow(unused_imports)]
use brahe as bh;
use bh::celestrak::{CelestrakClient, CelestrakQuery};

fn main() {
    bh::initialize_eop().unwrap();

    // Query ISS GP data by NORAD catalog number
    let client = CelestrakClient::new();
    let query = CelestrakQuery::gp().catnr(25544);
    let records = client.query_gp(&query).unwrap();
    let record = &records[0];

    println!("ISS GP Data:");
    println!("  Name: {}", record.object_name.as_deref().unwrap_or("Unknown"));
    println!("  NORAD ID: {}", record.norad_cat_id.unwrap_or(0));
    println!("  Epoch: {}", record.epoch.as_deref().unwrap_or("Unknown"));
    println!("  Inclination: {:.2}°", record.inclination.unwrap_or(0.0));
    println!("  RAAN: {:.2}°", record.ra_of_asc_node.unwrap_or(0.0));
    println!("  Eccentricity: {:.6}", record.eccentricity.unwrap_or(0.0));

    // Expected output:
    // ISS GP Data:
    //   Name: ISS (ZARYA)
    //   NORAD ID: 25544
    //   Epoch: 2025-11-02T10:09:34.283392
    //   Inclination: 51.63°
    //   RAAN: 342.07°
    //   Eccentricity: 0.000497
}
