//! Download a spherical harmonic gravity model from ICGEM.
//!
//! FLAGS = ["IGNORE"]
//!
//! Downloads are cached under $BRAHE_CACHE/icgem/models/<body>/ keyed on
//! (name, degree, hash-prefix). To pin a specific degree variant — when
//! ICGEM publishes more than one truncation of the same model — append
//! "-<DEGREE>" to the name argument, e.g. download_icgem_model(
//! ICGEMBody::Earth, "XGM2019e_2159-760", None).

use std::path::PathBuf;

use brahe as bh;
use bh::datasets::icgem::{ICGEMBody, download_icgem_model};

fn main() {
    // JGM3 is small (~70x70) and stable — a good demonstration target.
    // Passing just the name selects the largest published degree variant.
    let path = download_icgem_model(ICGEMBody::Earth, "JGM3", None).unwrap();
    println!("Cached at: {}", path.display());

    // Optionally also copy the file to a chosen location (cache still populated)
    let copied = download_icgem_model(
        ICGEMBody::Earth,
        "JGM3",
        Some(PathBuf::from("/tmp/icgem_jgm3.gfc")),
    )
    .unwrap();
    println!("Copied to: {}", copied.display());

    // Lunar model — body name routes to the celestial catalog
    let moon_path = download_icgem_model(ICGEMBody::Moon, "GLGM-1", None).unwrap();
    println!("Lunar model cached at: {}", moon_path.display());
}
