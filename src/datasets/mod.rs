/*!
 * The datasets module provides functionality for downloading and working with
 * satellite ephemeris data and groundstation locations from various sources.
 *
 * This module is organized with generic infrastructure (parsing, serialization, loading)
 * and source-specific implementations (CelesTrak, groundstations, etc.).
 */

pub mod celestrak;
pub mod groundstations;
pub mod loaders;
pub mod naif;
pub mod parsers;
pub mod serializers;

// Re-export commonly used functions from celestrak
pub use celestrak::{
    download_tles, get_tle_by_id, get_tle_by_id_as_propagator, get_tle_by_name,
    get_tle_by_name_as_propagator, get_tles, get_tles_as_propagators,
};

// Re-export commonly used functions from groundstations
pub use groundstations::{
    list_providers, load_all_groundstations, load_groundstations, load_groundstations_from_file,
};

// Re-export commonly used functions from naif
pub use naif::download_de_kernel;
