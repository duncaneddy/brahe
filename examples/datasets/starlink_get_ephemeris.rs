//! Look up a Starlink satellite in the public manifest and load its ephemeris.
//!
//! Uses the manifest cached under the brahe cache directory (seeded by
//! `just seed-starlink-cache`), finds a satellite by name, downloads or reuses
//! its Modified ITC file, and converts it to a trajectory.

use brahe as bh;
use brahe::starlink::StarlinkClient;
use brahe::traits::*;

fn main() {
    bh::initialize_eop().unwrap();

    let client = StarlinkClient::with_cache_age(7.0 * 86400.0);
    let manifest = client.get_manifest().unwrap();
    println!("Manifest lists {} satellites (retrieved {})", manifest.len(), manifest.retrieved);

    let entry = manifest.find_by_object_name("STARLINK-38128").unwrap();
    println!("{}: NORAD {}, {}", entry.object_name, entry.norad_cat_id, entry.category);
    println!("  ephemeris {} to {}", entry.ephemeris_start, entry.ephemeris_stop.map(|e| e.to_string()).unwrap_or_default());
    println!("  file {}", entry.file_name_string());

    let trajectory = client.get_trajectory(entry.norad_cat_id).unwrap();
    println!("Trajectory: {} samples in {}", trajectory.len(), trajectory.frame);
    println!("Cached files: {}", client.cached_files().unwrap().len());

    let df = manifest.to_dataframe().unwrap();
    let head = df.select(["norad_cat_id", "object_name", "ephemeris_start"]).unwrap().head(Some(3));
    println!("Manifest table shape: {:?}", head.shape());
    println!("Columns: {:?}", head.get_column_names());
}
