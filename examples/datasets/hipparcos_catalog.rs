//! Download the Hipparcos star catalog and find the brightest stars.
//!
//! This example demonstrates downloading the Hipparcos catalog, filtering by
//! visual magnitude, and inspecting the result as a Polars DataFrame.
//!
//! FLAGS = ["NETWORK"]

#[allow(unused_imports)]
use brahe as bh;
use bh::datasets::star_catalog;
use bh::datasets::star_catalog::StarRecord;

fn main() {
    // Download the Hipparcos catalog (cached permanently after the first download)
    let hipparcos = star_catalog::get_hipparcos_catalog(None).unwrap();
    println!("Loaded {} Hipparcos records", hipparcos.len());

    // Filter to naked-eye-bright stars (Vmag < 5.2)
    let bright = hipparcos.filter_by_magnitude(5.2);
    println!("Stars brighter than Vmag 5.2: {}", bright.len());

    // Sort by magnitude and print the 5 brightest names
    let mut records = bright.records().to_vec();
    records.sort_by(|a, b| a.vmag.partial_cmp(&b.vmag).unwrap());

    println!("\n5 brightest stars:");
    for r in records.iter().take(5) {
        println!(
            "  {}: Vmag={:.2}",
            r.name().unwrap_or_else(|| r.id()),
            r.vmag.unwrap()
        );
    }

    let df = bright.to_dataframe().unwrap();
    let head = df.head(Some(5));
    println!("\nDataFrame head shape: {:?}", head.shape());
    println!("Columns: {:?}", head.get_column_names());
}
