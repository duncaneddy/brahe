//! Download and inspect the GCAT SATCAT catalog.
//!
//! This example demonstrates downloading the SATCAT catalog and looking up
//! individual records by SATCAT number and JCAT identifier.
//!
//! FLAGS = ["CI-ONLY"]

#[allow(unused_imports)]
use brahe as bh;
use bh::datasets::gcat;

fn main() {
    // Download the SATCAT catalog (cached for 24 hours by default)
    let satcat = gcat::get_satcat(None).unwrap();
    println!("Loaded {} SATCAT records", satcat.len());

    // Look up the ISS by NORAD SATCAT number
    if let Some(iss) = satcat.get_by_satcat("25544") {
        println!("\nISS (by SATCAT number 25544):");
        println!("  JCAT:    {}", iss.jcat);
        println!("  Name:    {}", iss.name.as_deref().unwrap_or("Unknown"));
        println!("  Status:  {}", iss.status.as_deref().unwrap_or("Unknown"));
        println!(
            "  Perigee: {} km",
            iss.perigee.map_or("N/A".to_string(), |v| format!("{v}"))
        );
        println!(
            "  Apogee:  {} km",
            iss.apogee.map_or("N/A".to_string(), |v| format!("{v}"))
        );
        println!(
            "  Inc:     {}°",
            iss.inc.map_or("N/A".to_string(), |v| format!("{v}"))
        );
    }

    // Look up by JCAT identifier
    if let Some(record) = satcat.get_by_jcat("S049652") {
        println!(
            "\nRecord by JCAT S049652: {}",
            record.name.as_deref().unwrap_or("Unknown")
        );
    }

    // Expected output:
    // Loaded NNNNN SATCAT records
    //
    // ISS (by SATCAT number 25544):
    //   JCAT:    S049652
    //   Name:    ISS (Zarya)
    //   Status:  O
    //   Perigee: 408.0 km
    //   Apogee:  418.0 km
    //   Inc:     51.64°
    //
    // Record by JCAT S049652: ISS (Zarya)
}
