//! Download the FK5 catalog, look up a star by its running number, and
//! filter it by magnitude and cone search.
//!
//! FLAGS = ["NETWORK"]

#[allow(unused_imports)]
use brahe as bh;
use bh::datasets::star_catalogs;
use bh::datasets::star_catalogs::StarRecord;

fn main() {
    // Download the FK5 catalog (cached permanently after the first download)
    let fk5 = star_catalogs::get_fk5_catalog(None).unwrap();
    println!("Loaded {} FK5 records", fk5.len());

    // Look up a specific star by its FK5 running number
    let rec = fk5.get_by_id(699).unwrap();
    let vmag_str = rec
        .vmag
        .map(|v| format!("{:.2}", v))
        .unwrap_or_else(|| "N/A".to_string());
    println!(
        "FK5 699: {}, ra={:.4} deg, dec={:.4} deg, vmag={}",
        rec.name().unwrap_or_else(|| rec.id()),
        rec.ra,
        rec.dec,
        vmag_str
    );

    // Magnitude filter: keeps vmag <= max_mag (smaller/more negative is brighter)
    let bright = fk5.filter_by_magnitude(3.0);
    println!("Stars brighter than Vmag 3.0: {}", bright.len());

    // Cone search around a right ascension/declination, in degrees
    let nearby = fk5.filter_by_cone(101.28, -16.72, 5.0, bh::AngleFormat::Degrees);
    println!("Stars within 5 deg of (101.28, -16.72): {}", nearby.len());

    // Chained: bright stars within 5 degrees of a target (filter methods
    // return a new catalog instance, so the original catalog is never
    // modified)
    let bright_nearby =
        fk5.filter_by_magnitude(3.0)
            .filter_by_cone(101.28, -16.72, 5.0, bh::AngleFormat::Degrees);
    println!("Bright stars within 5 deg of target: {}", bright_nearby.len());
    for r in bright_nearby.records() {
        println!("  {}: Vmag={:.2}", r.name().unwrap_or_else(|| r.id()), r.vmag.unwrap());
    }
}
