//! Force-refresh the cached ICGEM index files.
//!
//! Indexes auto-refresh every 30 days, so this is only needed when ICGEM has
//! published a new model and you don't want to wait for the next normal cache
//! miss to pick it up.
//!
//! FLAGS = ["NETWORK"]

use brahe as bh;
use bh::datasets::icgem::{
    ICGEMBody, list_icgem_models, refresh_all_icgem_indexes, refresh_icgem_index,
};

fn main() {
    // Refresh a single body's listing. The Earth listing comes from ICGEM's
    // `tom_longtime` page; all non-Earth bodies share the `tom_celestial` index.
    refresh_icgem_index(ICGEMBody::Earth).unwrap();
    println!("Refreshed Earth index");

    // Refresh both index files in one call — equivalent to refreshing Earth plus
    // any non-Earth body (since the celestial listing covers Moon/Mars/Venus/Ceres/...).
    refresh_all_icgem_indexes().unwrap();
    println!("Refreshed all ICGEM indexes");

    // Confirm the refresh took effect by listing a known body
    let earth_models = list_icgem_models(ICGEMBody::Earth).unwrap();
    println!("\n{} Earth models after refresh", earth_models.len());
}
