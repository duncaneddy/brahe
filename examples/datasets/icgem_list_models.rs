//! List spherical harmonic gravity models from the ICGEM catalog.
//!
//! The first call fetches the listing from ICGEM and caches it under
//! $BRAHE_CACHE/icgem/. Subsequent calls within the 30-day TTL read from disk.
//!
//! FLAGS = ["NETWORK"]

use brahe as bh;
use bh::datasets::icgem::{ICGEMBody, list_icgem_models};

fn main() {
    // List all Earth gravity models in the catalog
    let earth_models = list_icgem_models(ICGEMBody::Earth).unwrap();
    println!("Earth models available: {}", earth_models.len());
    for entry in earth_models.iter().take(3) {
        println!(
            "  {:30} degree={:<6} year={:?}",
            entry.name, entry.degree, entry.year
        );
    }

    // Each entry is a plain IndexEntry, so standard iterator filtering works
    let egm_family = earth_models
        .iter()
        .filter(|e| e.name.starts_with("EGM"))
        .count();
    println!("\nEGM-family Earth models: {egm_family}");

    // The same call works for other bodies — Moon, Mars, Venus, Ceres, or any
    // custom celestial body present in the ICGEM celestial catalog.
    let moon_models = list_icgem_models(ICGEMBody::Moon).unwrap();
    println!("\nLunar models available: {}", moon_models.len());
    for entry in moon_models.iter().take(3) {
        println!(
            "  {:30} degree={:<6} year={:?}",
            entry.name, entry.degree, entry.year
        );
    }
}
