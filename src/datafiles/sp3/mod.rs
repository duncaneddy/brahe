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

mod interop;
mod parse;
mod satellite;
mod types;
mod write;

pub use satellite::{SP3_DEFAULT_LAGRANGE_DEGREE, SP3Satellite};

pub use types::{
    GNSSSystem, SP3, SP3Correlation, SP3FileType, SP3Header, SP3OrbitType, SP3Record,
    SP3SatelliteId, SP3TimeSystem, SP3Version, SP3WriteOptions,
};
