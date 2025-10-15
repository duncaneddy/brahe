/*!
 * The datasets module provides functionality for downloading and working with
 * satellite ephemeris data from various sources.
 *
 * This module is organized with generic infrastructure (parsing, serialization)
 * and source-specific implementations (CelesTrak, future: Space-Track, etc.).
 */

pub mod celestrak;
pub mod parsers;
pub mod serializers;

// Re-export commonly used functions from celestrak
pub use celestrak::{download_ephemeris, get_ephemeris, get_ephemeris_as_propagators};
