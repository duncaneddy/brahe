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

impl OEMStateVector {
    /// Create a new state vector.
    ///
    /// # Arguments
    ///
    /// * `epoch` - Epoch of this state vector
    /// * `position` - Position vector [x, y, z]. Units: meters
    /// * `velocity` - Velocity vector [vx, vy, vz]. Units: m/s
    pub fn new(epoch: Epoch, position: [f64; 3], velocity: [f64; 3]) -> Self {
        Self {
            epoch,
            position,
            velocity,
            acceleration: None,
        }
    }

    /// Set the optional acceleration vector.
    ///
    /// # Arguments
    ///
    /// * `acceleration` - Acceleration vector [ax, ay, az]. Units: m/s²
    pub fn with_acceleration(mut self, acceleration: [f64; 3]) -> Self {
        self.acceleration = Some(acceleration);
        self
    }
}

impl OEMSegment {
    /// Create a new empty segment with the given metadata.
    ///
    /// # Arguments
    ///
    /// * `metadata` - Segment metadata
    pub fn new(metadata: OEMMetadata) -> Self {
        Self {
            metadata,
            comments: Vec::new(),
            states: Vec::new(),
            covariances: Vec::new(),
        }
    }

    /// Add a state vector to this segment.
    ///
    /// # Arguments
    ///
    /// * `state` - State vector to add
    pub fn push_state(&mut self, state: OEMStateVector) {
        self.states.push(state);
    }
}

impl OEMMetadata {
    /// Create new metadata with required fields.
    ///
    /// # Arguments
    ///
    /// * `object_name` - Spacecraft name
    /// * `object_id` - International designator (e.g., "1996-062A")
    /// * `center_name` - Central body name (e.g., "EARTH")
    /// * `ref_frame` - Reference frame for the state vectors
    /// * `time_system` - Time system for all epochs
    /// * `start_time` - Start time of the ephemeris data
    /// * `stop_time` - Stop time of the ephemeris data
    pub fn new(
        object_name: String,
        object_id: String,
        center_name: String,
        ref_frame: CCSDSRefFrame,
        time_system: CCSDSTimeSystem,
        start_time: Epoch,
        stop_time: Epoch,
    ) -> Self {
        Self {
            object_name,
            object_id,
            center_name,
            ref_frame,
            ref_frame_epoch: None,
            time_system,
            start_time,
            useable_start_time: None,
            useable_stop_time: None,
            stop_time,
            interpolation: None,
            interpolation_degree: None,
            comments: Vec::new(),
        }
    }

    /// Set the interpolation method.
    pub fn with_interpolation(mut self, method: String, degree: Option<u32>) -> Self {
        self.interpolation = Some(method);
        self.interpolation_degree = degree;
        self
    }
}

impl OEM {
    /// Create a new empty OEM message.
    ///
    /// # Arguments
    ///
    /// * `originator` - Originator of the message
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::ccsds::oem::OEM;
    ///
    /// let oem = OEM::new("MY_ORG".to_string());
    /// assert_eq!(oem.segments.len(), 0);
    /// ```
    pub fn new(originator: String) -> Self {
        Self {
            header: ODMHeader {
                format_version: 3.0,
                classification: None,
                creation_date: Epoch::now(),
                originator,
                message_id: None,
                comments: Vec::new(),
            },
            segments: Vec::new(),
        }
    }

    /// Add a segment to the OEM.
    ///
    /// # Arguments
    ///
    /// * `segment` - Segment to add
    pub fn push_segment(&mut self, segment: OEMSegment) {
        self.segments.push(segment);
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oem_builder() {
        let mut oem = OEM::new("TEST_ORG".to_string());
        assert_eq!(oem.header.originator, "TEST_ORG");
        assert_eq!(oem.segments.len(), 0);

        let metadata = OEMMetadata::new(
            "SAT1".to_string(),
            "2024-001A".to_string(),
            "EARTH".to_string(),
            CCSDSRefFrame::GCRF,
            CCSDSTimeSystem::UTC,
            Epoch::now(),
            Epoch::now(),
        );
        let mut seg = OEMSegment::new(metadata);
        assert_eq!(seg.states.len(), 0);

        let sv = OEMStateVector::new(Epoch::now(), [7000e3, 0.0, 0.0], [0.0, 7500.0, 0.0]);
        seg.push_state(sv);
        assert_eq!(seg.states.len(), 1);

        let sv_with_acc =
            OEMStateVector::new(Epoch::now(), [6000e3, 3000e3, 0.0], [-2000.0, 6000.0, 0.0])
                .with_acceleration([0.001, 0.002, 0.003]);
        seg.push_state(sv_with_acc);
        assert_eq!(seg.states.len(), 2);
        assert!(seg.states[1].acceleration.is_some());

        oem.push_segment(seg);
        assert_eq!(oem.segments.len(), 1);
        assert_eq!(oem.segments[0].metadata.object_name, "SAT1");
        assert_eq!(oem.segments[0].states.len(), 2);
    }

    #[test]
    fn test_oem_builder_round_trip() {
        let mut oem = OEM::new("ROUND_TRIP".to_string());
        let metadata = OEMMetadata::new(
            "TEST_SAT".to_string(),
            "2024-999A".to_string(),
            "EARTH".to_string(),
            CCSDSRefFrame::J2000,
            CCSDSTimeSystem::UTC,
            Epoch::from_datetime(2024, 6, 1, 0, 0, 0.0, 0.0, crate::time::TimeSystem::UTC),
            Epoch::from_datetime(2024, 6, 1, 1, 0, 0.0, 0.0, crate::time::TimeSystem::UTC),
        );
        let mut seg = OEMSegment::new(metadata);
        seg.push_state(OEMStateVector::new(
            Epoch::from_datetime(2024, 6, 1, 0, 0, 0.0, 0.0, crate::time::TimeSystem::UTC),
            [7000e3, 0.0, 0.0],
            [0.0, 7500.0, 0.0],
        ));
        oem.push_segment(seg);

        let kvn = oem.to_string(CCSDSFormat::KVN).unwrap();
        let oem2 = OEM::from_str(&kvn).unwrap();
        assert_eq!(oem2.header.originator, "ROUND_TRIP");
        assert_eq!(oem2.segments.len(), 1);
        assert_eq!(oem2.segments[0].states.len(), 1);
    }

    #[test]
    fn test_oem_metadata_with_interpolation() {
        let metadata = OEMMetadata::new(
            "SAT".to_string(),
            "2024-001A".to_string(),
            "EARTH".to_string(),
            CCSDSRefFrame::GCRF,
            CCSDSTimeSystem::UTC,
            Epoch::now(),
            Epoch::now(),
        )
        .with_interpolation("HERMITE".to_string(), Some(7));

        assert_eq!(metadata.interpolation.as_deref(), Some("HERMITE"));
        assert_eq!(metadata.interpolation_degree, Some(7));
    }
}
