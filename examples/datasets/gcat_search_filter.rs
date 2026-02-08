//! Search and filter the GCAT SATCAT catalog.
//!
//! This example demonstrates name search and filter chaining to narrow
//! down the catalog to specific subsets of objects.
//!
//! FLAGS = ["CI-ONLY"]

#[allow(unused_imports)]
use brahe as bh;
use bh::datasets::gcat;

fn main() {
    // Download the SATCAT catalog
    let satcat = gcat::get_satcat(None).unwrap();
    println!("Total records: {}", satcat.len());

    // Search by name (case-insensitive, searches both name and pl_name)
    let starlink = satcat.search_by_name("starlink");
    println!("\nStarlink name search: {} results", starlink.len());

    // Filter chaining: payloads that are operational in LEO
    let payloads = satcat.filter_by_type("P");
    println!("\nAll payloads: {}", payloads.len());

    let operational = payloads.filter_by_status("O");
    println!("Operational payloads: {}", operational.len());

    let leo = operational.filter_by_perigee_range(160.0, 2000.0);
    println!("Operational LEO payloads: {}", leo.len());

    // Filter by inclination range (sun-synchronous orbits ~96-99 deg)
    let sso = operational.filter_by_inc_range(96.0, 99.0);
    println!("Operational SSO payloads: {}", sso.len());

    // Expected output:
    // Total records: NNNNN
    //
    // Starlink name search: NNNN results
    //
    // All payloads: NNNNN
    // Operational payloads: NNNNN
    // Operational LEO payloads: NNNNN
    // Operational SSO payloads: NNNN
}
