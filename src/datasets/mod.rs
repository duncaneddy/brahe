/*!
 * The datasets module provides functionality for downloading and working with
 * satellite ephemeris data and groundstation locations from various sources.
 *
 * This module is organized with generic infrastructure (parsing, serialization, loading)
 * and source-specific implementations (CelesTrak, groundstations, etc.).
 */

pub mod gcat;
pub mod groundstations;
pub mod loaders;
pub mod naif;
pub mod parsers;
pub mod serializers;

// Re-export commonly used functions from groundstations
pub use groundstations::{
    list_providers, load_all_groundstations, load_groundstations, load_groundstations_from_file,
};

// Re-export commonly used functions from naif
pub use naif::download_de_kernel;

// Re-export commonly used types and functions from gcat
pub use gcat::{
    GCATPsatcat, GCATPsatcatRecord, GCATSatcat, GCATSatcatRecord, get_psatcat, get_satcat,
};
