/*!
 * CCSDS Orbit Data Message (ODM) support.
 *
 * This module provides parsing and writing of CCSDS standard orbit data
 * messages in three encoding formats: KVN (text), XML, and JSON.
 *
 * Supported message types:
 * - **OEM** (Orbit Ephemeris Message): Time-series of state vectors
 * - **OMM** (Orbit Mean-elements Message): Mean elements for SGP4 propagation
 * - **OPM** (Orbit Parameter Message): Single state vector with optional extras
 *
 * Reference: CCSDS 502.0-B-3 (Orbit Data Messages), April 2023
 *
 * # Usage
 *
 * ```no_run
 * use brahe::ccsds::oem::OEM;
 *
 * let oem = OEM::from_file("ephemeris.oem").unwrap();
 * for segment in &oem.segments {
 *     println!("Object: {}", segment.metadata.object_name);
 *     println!("States: {}", segment.states.len());
 * }
 * ```
 */

pub mod cdm;
pub mod common;
pub mod error;
pub mod interop;
pub mod json;
pub mod kvn;
pub mod oem;
pub mod omm;
pub mod opm;
pub mod xml;

// Re-export commonly used types at the ccsds module level
pub use cdm::{CDM, CDMObject, CDMObjectMetadata, CDMRTNCovariance, CDMStateVector};
pub use common::{CCSDSFormat, CCSDSJsonKeyCase, CCSDSRefFrame, CCSDSTimeSystem};
pub use oem::{OEM, OEMMetadata, OEMSegment, OEMStateVector};
pub use omm::OMM;
pub use opm::OPM;
