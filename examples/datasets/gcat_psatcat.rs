//! Download and explore the GCAT PSATCAT (payload) catalog.
//!
//! This example demonstrates downloading the PSATCAT catalog and using
//! payload-specific filters like category, class, and active status.
//!
//! FLAGS = ["CI-ONLY"]

#[allow(unused_imports)]
use brahe as bh;
use bh::datasets::gcat;

fn main() {
    // Download the PSATCAT catalog
    let psatcat = gcat::get_psatcat(None).unwrap();
    println!("Loaded {} PSATCAT records", psatcat.len());

    // Filter for active payloads (result="S" and no end date)
    let active = psatcat.filter_active();
    println!("Active payloads: {}", active.len());

    // Filter by mission category
    let comms = psatcat.filter_by_category("Communications");
    println!("\nCommunications payloads: {}", comms.len());

    // Filter by mission class
    let stations = psatcat.filter_by_class("Station");
    println!("Space stations: {}", stations.len());

    // Look up a specific payload
    if let Some(iss) = psatcat.get_by_jcat("S049652") {
        println!("\nISS Payload Details:");
        println!("  Name:       {}", iss.name.as_deref().unwrap_or("Unknown"));
        println!(
            "  Program:    {}",
            iss.program.as_deref().unwrap_or("Unknown")
        );
        println!(
            "  Category:   {}",
            iss.category.as_deref().unwrap_or("Unknown")
        );
        println!(
            "  Class:      {}",
            iss.class.as_deref().unwrap_or("Unknown")
        );
        println!(
            "  Result:     {}",
            iss.result.as_deref().unwrap_or("Unknown")
        );
        println!(
            "  Discipline: {}",
            iss.discipline.as_deref().unwrap_or("Unknown")
        );
    }

    // Expected output:
    // Loaded NNNNN PSATCAT records
    // Active payloads: NNNN
    //
    // Communications payloads: NNNN
    // Space stations: NN
    //
    // ISS Payload Details:
    //   Name:       ISS (Zarya)
    //   Program:    ISS
    //   Category:   Human spaceflight
    //   Class:      Station
    //   Result:     S
    //   Discipline: Life sci
}
