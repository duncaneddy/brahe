/*!
 * SP3 precise orbit file format.
 *
 * SP3 (Standard Product 3) is the IGS exchange format for precise satellite
 * ephemerides and clocks. Versions a through d are described by the NGS
 * SP3 documentation and the IGS format specifications. All quantities exposed
 * by this module use SI units (meters, meters per second, seconds); the
 * kilometer, decimeter-per-second and microsecond units of the file format
 * appear only in the parser and writer.
 */

mod types;

pub use types::{GNSSSystem, SP3FileType, SP3OrbitType, SP3SatelliteId, SP3TimeSystem, SP3Version};
