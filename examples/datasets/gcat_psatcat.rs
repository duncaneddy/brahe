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

    // Filter by mission category (COM=communications, IMG=imaging, NAV=navigation, etc.)
    let comms = psatcat.filter_by_category("COM");
    println!("\nCommunications payloads: {}", comms.len());

    // Filter by mission class (A=amateur, B=business, C=civil, D=defense)
    let civil = psatcat.filter_by_class("C");
    println!("Civil payloads: {}", civil.len());

    // Look up a specific payload (ISS Zarya module)
    if let Some(iss) = psatcat.get_by_jcat("S25544") {
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
    }

    // Expected output:
    // Loaded NNNNN PSATCAT records
    // Active payloads: NNNN
    //
    // Communications payloads: NNNNN
    // Civil payloads: NNNN
    //
    // ISS Payload Details:
    //   Name:       Zarya Cargo Block
    //   Program:    TsM
    //   Category:   SS
    //   Class:      C
    //   Result:     S
}
