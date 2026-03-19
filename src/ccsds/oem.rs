/*!
 * CCSDS Orbit Ephemeris Message (OEM) data structures.
 *
 * OEM messages contain time-ordered sequences of state vectors (position and
 * velocity), optionally with accelerations and covariance matrices. They are
 * the standard format for exchanging ephemeris data between space agencies.
 *
 * Reference: CCSDS 502.0-B-3 (Orbit Data Messages), April 2023
 */

use std::path::Path;

use crate::ccsds::common::{
    CCSDSCovariance, CCSDSFormat, CCSDSRefFrame, CCSDSTimeSystem, ODMHeader,
};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

/// A complete CCSDS Orbit Ephemeris Message.
///
/// An OEM contains a header and one or more segments, each with its own
/// metadata and ephemeris data. Multiple segments allow for different
/// time spans, reference frames, or objects within a single file.
#[derive(Debug, Clone)]
pub struct OEM {
    /// Message header with version, originator, and creation date
    pub header: ODMHeader,
    /// One or more ephemeris data segments
    pub segments: Vec<OEMSegment>,
}

/// A single segment within an OEM message.
///
/// Each segment has its own metadata (object, frame, time system) and
/// contains ephemeris data lines and optional covariance matrices.
#[derive(Debug, Clone)]
pub struct OEMSegment {
    /// Segment metadata defining the object, reference frame, and time span
    pub metadata: OEMMetadata,
    /// Comments associated with the data block
    pub comments: Vec<String>,
    /// Time-ordered state vectors
    pub states: Vec<OEMStateVector>,
    /// Optional covariance matrices
    pub covariances: Vec<CCSDSCovariance>,
}

/// Metadata for an OEM segment.
#[derive(Debug, Clone)]
pub struct OEMMetadata {
    /// Spacecraft name
    pub object_name: String,
    /// International designator (e.g., "1996-062A")
    pub object_id: String,
    /// Name of the central body (e.g., "EARTH", "MARS BARYCENTER")
    pub center_name: String,
    /// Reference frame for the state vectors
    pub ref_frame: CCSDSRefFrame,
    /// Optional epoch for the reference frame
    pub ref_frame_epoch: Option<Epoch>,
    /// Time system for all epochs in this segment
    pub time_system: CCSDSTimeSystem,
    /// Start time of the ephemeris data
    pub start_time: Epoch,
    /// Optional useable start time
    pub useable_start_time: Option<Epoch>,
    /// Optional useable stop time
    pub useable_stop_time: Option<Epoch>,
    /// Stop time of the ephemeris data
    pub stop_time: Epoch,
    /// Interpolation method (e.g., "HERMITE", "LAGRANGE")
    pub interpolation: Option<String>,
    /// Interpolation degree
    pub interpolation_degree: Option<u32>,
    /// Comments in the metadata block
    pub comments: Vec<String>,
}

/// A single state vector entry in an OEM ephemeris block.
///
/// Position and velocity are stored in SI units (meters, m/s).
/// CCSDS files use km and km/s, which are converted on parse/write.
#[derive(Debug, Clone)]
pub struct OEMStateVector {
    /// Epoch of this state vector
    pub epoch: Epoch,
    /// Position vector [x, y, z]. Units: meters
    pub position: [f64; 3],
    /// Velocity vector [vx, vy, vz]. Units: m/s
    pub velocity: [f64; 3],
    /// Optional acceleration vector [ax, ay, az]. Units: m/s²
    pub acceleration: Option<[f64; 3]>,
}

impl OEM {
    /// Parse an OEM message from a string, auto-detecting the format.
    ///
    /// # Arguments
    ///
    /// * `content` - String content of the OEM message
    ///
    /// # Returns
    ///
    /// * `Result<OEM, BraheError>` - Parsed OEM message or error
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(content: &str) -> Result<Self, BraheError> {
        let format = crate::ccsds::detect_format(content);
        match format {
            CCSDSFormat::KVN => crate::ccsds::kvn::parse_oem(content),
            CCSDSFormat::XML => crate::ccsds::xml::parse_oem_xml(content),
            CCSDSFormat::JSON => crate::ccsds::json::parse_oem_json(content),
        }
    }

    /// Parse an OEM message from a file, auto-detecting the format.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the OEM file
    ///
    /// # Returns
    ///
    /// * `Result<OEM, BraheError>` - Parsed OEM message or error
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, BraheError> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| BraheError::IoError(format!("Failed to read OEM file: {}", e)))?;
        Self::from_str(&content)
    }

    /// Write the OEM message to a string in the specified format.
    ///
    /// # Arguments
    ///
    /// * `format` - Output encoding format (KVN, XML, or JSON)
    ///
    /// # Returns
    ///
    /// * `Result<String, BraheError>` - Serialized OEM string or error
    pub fn to_string(&self, format: CCSDSFormat) -> Result<String, BraheError> {
        match format {
            CCSDSFormat::KVN => crate::ccsds::kvn::write_oem(self),
            CCSDSFormat::XML => crate::ccsds::xml::write_oem_xml(self),
            CCSDSFormat::JSON => crate::ccsds::json::write_oem_json(self),
        }
    }

    /// Write the OEM message to a file in the specified format.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    /// * `format` - Output encoding format (KVN, XML, or JSON)
    ///
    /// # Returns
    ///
    /// * `Result<(), BraheError>` - Success or error
    pub fn to_file<P: AsRef<Path>>(&self, path: P, format: CCSDSFormat) -> Result<(), BraheError> {
        let content = self.to_string(format)?;
        std::fs::write(path.as_ref(), content)
            .map_err(|e| BraheError::IoError(format!("Failed to write OEM file: {}", e)))
    }
}
