//! Query the Starlink manifest: lookups, iteration, change detection and a
//! Polars DataFrame view.
//!
//! Uses the manifest cached under the brahe cache directory (seeded by
//! `just seed-starlink-cache`), covering the entry count, ID and name
//! lookups, the first few entries, a manifest-to-manifest diff, and the
//! DataFrame form.

use brahe as bh;
use brahe::starlink::StarlinkClient;
use brahe::time::{Epoch, TimeSystem};

fn main() {
    bh::initialize_eop().unwrap();

    let client = StarlinkClient::with_cache_age(7.0 * 86400.0);
    let manifest = client.get_manifest().unwrap();
    println!("Manifest lists {} satellites", manifest.len());

    let by_id = manifest.find_by_norad_id(100001).unwrap();
    println!("NORAD 100001 is {}", by_id.object_name);
    let by_name = manifest.find_by_object_name("STARLINK-37711").unwrap();
    println!("STARLINK-37711 is NORAD {}", by_name.norad_cat_id);

    for entry in manifest.iter().take(3) {
        println!(
            "  {} {}: {} to {}",
            entry.norad_cat_id,
            entry.object_name,
            entry.ephemeris_start,
            entry
                .ephemeris_stop
                .map(|e| e.to_string())
                .unwrap_or_else(|| "None".to_string())
        );
    }

    match client.previous_manifest().unwrap() {
        Some(previous) => {
            let changed = manifest.changed_since(&previous);
            println!("{} satellites changed since the previous manifest", changed.len());
        }
        None => println!("No previous manifest cached; nothing to diff against"),
    }

    let df = manifest.to_dataframe().unwrap();
    println!("DataFrame shape: {:?}", df.shape());
    println!("Columns: {:?}", df.get_column_names());

    // The `polars` dependency here is built with only the `lazy` feature and
    // isn't itself a nameable crate for this example, so filter the manifest
    // entries directly instead of the Python example's Polars-level
    // `.filter()` expression.
    let threshold_text = "2026-09-11T01:42:30";
    let threshold = Epoch::from_datetime(2026, 9, 11, 1, 42, 30.0, 0.0, TimeSystem::UTC);
    let later: Vec<&str> = manifest
        .iter()
        .filter(|e| e.ephemeris_start > threshold)
        .map(|e| e.object_name.as_str())
        .collect();
    println!("Entries starting after {}: {:?}", threshold_text, later);
}
