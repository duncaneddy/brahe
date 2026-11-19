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
    ensure_compliant_write, odm_object_compliance_issues,
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
    /// Spacecraft name, `None` when the source omits `OBJECT_NAME` or leaves
    /// it empty. Mandatory in CCSDS 502.0-B-3; reported by `compliance_issues`.
    pub object_name: Option<String>,
    /// International designator (e.g., "1996-062A"), `None` when the source
    /// omits `OBJECT_ID` or leaves it empty. Mandatory in CCSDS 502.0-B-3;
    /// reported by `compliance_issues`.
    pub object_id: Option<String>,
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
            object_name: Some(object_name),
            object_id: Some(object_id),
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
                creation_date: Some(Epoch::now()),
                originator: Some(originator),
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

    /// List the CCSDS 502.0-B-3 mandatory keywords this message lacks.
    ///
    /// Readers accept messages that omit `CREATION_DATE`, `ORIGINATOR`,
    /// `OBJECT_NAME`, or `OBJECT_ID`, or leave them empty, because the
    /// orbit data remains usable without them. This method reports which of
    /// those keywords are absent.
    ///
    /// # Returns
    ///
    /// * `Vec<String>` - Absent keywords prefixed by their location (e.g.,
    ///   `header.CREATION_DATE`), empty when the message is compliant.
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::ccsds::OEM;
    ///
    /// let oem = OEM::from_file("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
    /// assert_eq!(oem.compliance_issues(), Vec::<String>::new());
    /// ```
    pub fn compliance_issues(&self) -> Vec<String> {
        let mut issues = self.header.compliance_issues();
        for (i, segment) in self.segments.iter().enumerate() {
            issues.extend(odm_object_compliance_issues(
                &format!("segments[{}].metadata", i),
                &segment.metadata.object_name,
                &segment.metadata.object_id,
            ));
        }
        issues
    }

    /// Whether the message carries every CCSDS 502.0-B-3 mandatory keyword.
    ///
    /// # Returns
    ///
    /// * `bool` - `true` when [`Self::compliance_issues`] is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::ccsds::OEM;
    ///
    /// let oem = OEM::from_file("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
    /// assert_eq!(oem.is_compliant(), Vec::<String>::new().is_empty());
    /// ```
    pub fn is_compliant(&self) -> bool {
        self.compliance_issues().is_empty()
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
        let format = crate::ccsds::common::detect_format(content);
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
    /// The message must be compliant with CCSDS 502.0-B-3: when
    /// [`Self::compliance_issues`] is non-empty the write is refused. Use
    /// [`Self::to_string_noncompliant`] to write such a message anyway.
    ///
    /// # Arguments
    ///
    /// * `format` - Output encoding format (KVN, XML, or JSON)
    ///
    /// # Returns
    ///
    /// * `Result<String, BraheError>` - Serialized OEM string, or an error when
    ///   the message lacks mandatory keywords
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::ccsds::{CCSDSFormat, OEM};
    ///
    /// let mut oem = OEM::from_file("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
    /// assert!(oem.to_string(CCSDSFormat::KVN).is_ok());
    ///
    /// oem.header.originator = None;
    /// assert!(oem.to_string(CCSDSFormat::KVN).is_err());
    /// ```
    pub fn to_string(&self, format: CCSDSFormat) -> Result<String, BraheError> {
        ensure_compliant_write("OEM", self.compliance_issues())?;
        self.to_string_noncompliant(format)
    }

    /// Write the OEM message to a string without checking CCSDS compliance.
    ///
    /// Mandatory keywords listed by [`Self::compliance_issues`] are written
    /// with empty values. Orbit-defining fields are still required.
    ///
    /// # Arguments
    ///
    /// * `format` - Output encoding format (KVN, XML, or JSON)
    ///
    /// # Returns
    ///
    /// * `Result<String, BraheError>` - Serialized OEM string
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::ccsds::{CCSDSFormat, OEM};
    ///
    /// let mut oem = OEM::from_file("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
    /// oem.header.originator = None;
    /// let kvn = oem.to_string_noncompliant(CCSDSFormat::KVN).unwrap();
    /// assert!(OEM::from_str(&kvn).unwrap().header.originator.is_none());
    /// ```
    pub fn to_string_noncompliant(&self, format: CCSDSFormat) -> Result<String, BraheError> {
        match format {
            CCSDSFormat::KVN => crate::ccsds::kvn::write_oem(self),
            CCSDSFormat::XML => crate::ccsds::xml::write_oem_xml(self),
            CCSDSFormat::JSON => crate::ccsds::json::write_oem_json(
                self,
                crate::ccsds::common::CCSDSJsonKeyCase::Lower,
            ),
        }
    }

    /// Write the OEM message to JSON with explicit key case control.
    ///
    /// The write is refused when [`Self::compliance_issues`] is non-empty.
    /// Use [`Self::to_json_string_noncompliant`] to write such a message
    /// anyway.
    ///
    /// # Arguments
    ///
    /// * `key_case` - Whether CCSDS keywords should be lowercase or uppercase
    ///
    /// # Returns
    ///
    /// * `Result<String, BraheError>` - Serialized JSON string, or an error when
    ///   the message lacks mandatory keywords
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::ccsds::OEM;
    /// use brahe::ccsds::common::CCSDSJsonKeyCase;
    ///
    /// let oem = OEM::from_file("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
    /// let json = oem.to_json_string(CCSDSJsonKeyCase::Upper).unwrap();
    /// assert!(json.contains("ORIGINATOR"));
    /// ```
    pub fn to_json_string(
        &self,
        key_case: crate::ccsds::common::CCSDSJsonKeyCase,
    ) -> Result<String, BraheError> {
        ensure_compliant_write("OEM", self.compliance_issues())?;
        self.to_json_string_noncompliant(key_case)
    }

    /// Write the OEM message to JSON without checking CCSDS compliance.
    ///
    /// # Arguments
    ///
    /// * `key_case` - Whether CCSDS keywords should be lowercase or uppercase
    ///
    /// # Returns
    ///
    /// * `Result<String, BraheError>` - Serialized JSON string
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::ccsds::OEM;
    /// use brahe::ccsds::common::CCSDSJsonKeyCase;
    ///
    /// let mut oem = OEM::from_file("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
    /// oem.header.originator = None;
    /// assert!(oem.to_json_string_noncompliant(CCSDSJsonKeyCase::Upper).is_ok());
    /// ```
    pub fn to_json_string_noncompliant(
        &self,
        key_case: crate::ccsds::common::CCSDSJsonKeyCase,
    ) -> Result<String, BraheError> {
        crate::ccsds::json::write_oem_json(self, key_case)
    }

    /// Write the OEM message to a file in the specified format.
    ///
    /// The write is refused, and no file is created, when
    /// [`Self::compliance_issues`] is non-empty. Use
    /// [`Self::to_file_noncompliant`] to write such a message anyway.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    /// * `format` - Output encoding format (KVN, XML, or JSON)
    ///
    /// # Returns
    ///
    /// * `Result<(), BraheError>` - Success, or an error when the file cannot be
    ///   written or the message lacks mandatory keywords
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::ccsds::{CCSDSFormat, OEM};
    ///
    /// let oem = OEM::from_file("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
    /// let path = std::env::temp_dir().join("brahe_oem_to_file_example.txt");
    /// oem.to_file(&path, CCSDSFormat::KVN).unwrap();
    /// ```
    pub fn to_file<P: AsRef<Path>>(&self, path: P, format: CCSDSFormat) -> Result<(), BraheError> {
        let content = self.to_string(format)?;
        Self::write_file(path.as_ref(), content)
    }

    /// Write the OEM message to a file without checking CCSDS compliance.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    /// * `format` - Output encoding format (KVN, XML, or JSON)
    ///
    /// # Returns
    ///
    /// * `Result<(), BraheError>` - Success, or an error when the file cannot be
    ///   written
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::ccsds::{CCSDSFormat, OEM};
    ///
    /// let mut oem = OEM::from_file("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
    /// oem.header.originator = None;
    /// let path = std::env::temp_dir().join("brahe_oem_to_file_noncompliant_example.txt");
    /// oem.to_file_noncompliant(&path, CCSDSFormat::KVN).unwrap();
    /// ```
    pub fn to_file_noncompliant<P: AsRef<Path>>(
        &self,
        path: P,
        format: CCSDSFormat,
    ) -> Result<(), BraheError> {
        let content = self.to_string_noncompliant(format)?;
        Self::write_file(path.as_ref(), content)
    }

    fn write_file(path: &Path, content: String) -> Result<(), BraheError> {
        std::fs::write(path, content)
            .map_err(|e| BraheError::IoError(format!("Failed to write OEM file: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serial_test::parallel;

    fn without_keyword(path: &str, keyword: &str) -> String {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .filter(|l| !l.starts_with(keyword))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[rstest]
    #[case("CREATION_DATE", &["header.CREATION_DATE"])]
    #[case("ORIGINATOR", &["header.ORIGINATOR"])]
    #[case(
        "OBJECT_NAME",
        &[
            "segments[0].metadata.OBJECT_NAME",
            "segments[1].metadata.OBJECT_NAME",
            "segments[2].metadata.OBJECT_NAME",
        ]
    )]
    #[case(
        "OBJECT_ID",
        &[
            "segments[0].metadata.OBJECT_ID",
            "segments[1].metadata.OBJECT_ID",
            "segments[2].metadata.OBJECT_ID",
        ]
    )]
    #[parallel]
    fn test_oem_compliance_issues_missing_keyword(#[case] keyword: &str, #[case] issues: &[&str]) {
        let oem = OEM::from_str(&without_keyword(
            "test_assets/ccsds/oem/OEMExample1.txt",
            keyword,
        ))
        .unwrap();

        assert_eq!(oem.compliance_issues(), issues);
        assert!(!oem.is_compliant());
    }

    #[test]
    #[parallel]
    fn test_oem_is_compliant() {
        let oem = OEM::from_file("test_assets/ccsds/oem/OEMExample1.txt").unwrap();

        assert!(oem.compliance_issues().is_empty());
        assert!(oem.is_compliant());
    }

    #[rstest]
    #[case(CCSDSFormat::KVN)]
    #[case(CCSDSFormat::XML)]
    #[case(CCSDSFormat::JSON)]
    #[parallel]
    fn test_oem_strict_write_rejects_noncompliant(#[case] format: CCSDSFormat) {
        let mut oem = OEM::from_file("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        oem.header.originator = None;

        let err = oem.to_string(format).unwrap_err();
        assert!(err.to_string().contains("header.ORIGINATOR"), "{}", err);
        assert!(
            oem.to_json_string(crate::ccsds::common::CCSDSJsonKeyCase::Lower)
                .is_err()
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("oem.txt");
        assert!(oem.to_file(&path, format).is_err());
        assert!(!path.exists());

        let written = oem.to_string_noncompliant(format).unwrap();
        assert_eq!(OEM::from_str(&written).unwrap().header.originator, None);
        oem.to_file_noncompliant(&path, format).unwrap();
        assert!(path.exists());
    }

    #[rstest]
    #[case("test_assets/ccsds/oem/OEMExample1.txt", "1996-11-04T17:22:31")]
    #[case("test_assets/ccsds/oem/OEMExample3.xml", "1996-11-04T17:22:31")]
    #[parallel]
    fn test_oem_empty_creation_date(#[case] path: &str, #[case] date: &str) {
        let content = std::fs::read_to_string(path).unwrap().replacen(date, "", 1);

        let oem = OEM::from_str(&content).unwrap();
        assert!(oem.header.creation_date.is_none());

        for format in [CCSDSFormat::KVN, CCSDSFormat::XML, CCSDSFormat::JSON] {
            let oem2 = OEM::from_str(&oem.to_string_noncompliant(format).unwrap()).unwrap();
            assert!(oem2.header.creation_date.is_none());
        }
    }

    #[test]
    #[parallel]
    fn test_oem_builder() {
        let mut oem = OEM::new("TEST_ORG".to_string());
        assert_eq!(oem.header.originator.as_deref(), Some("TEST_ORG"));
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
        assert_eq!(
            oem.segments[0].metadata.object_name.as_deref(),
            Some("SAT1")
        );
        assert_eq!(oem.segments[0].states.len(), 2);
    }

    #[test]
    #[parallel]
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
        assert_eq!(oem2.header.originator.as_deref(), Some("ROUND_TRIP"));
        assert_eq!(oem2.segments.len(), 1);
        assert_eq!(oem2.segments[0].states.len(), 1);
    }

    #[test]
    #[parallel]
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

    #[test]
    #[parallel]
    fn test_oem_json_round_trip_via_dispatch() {
        let mut oem = OEM::new("JSON_RT".to_string());
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

        let json_str = oem.to_string(CCSDSFormat::JSON).unwrap();
        assert!(json_str.contains("7000") || json_str.contains("7.0"));
        let oem2 = OEM::from_str(&json_str).unwrap();
        assert_eq!(oem2.header.originator.as_deref(), Some("JSON_RT"));
        assert_eq!(oem2.segments.len(), 1);
        assert_eq!(oem2.segments[0].states.len(), 1);
        assert!((oem2.segments[0].states[0].position[0] - 7000e3).abs() < 1.0);
    }

    #[test]
    #[parallel]
    fn test_oem_xml_round_trip() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample3.xml").unwrap();
        let oem = OEM::from_str(&content).unwrap();
        let xml_str = oem.to_string(CCSDSFormat::XML).unwrap();
        let oem2 = OEM::from_str(&xml_str).unwrap();
        assert_eq!(oem2.header.originator, oem.header.originator);
        assert_eq!(oem2.segments.len(), oem.segments.len());
        assert_eq!(oem2.segments[0].states.len(), oem.segments[0].states.len());
        // Position round-trip
        assert!(
            (oem2.segments[0].states[0].position[0] - oem.segments[0].states[0].position[0]).abs()
                < 1.0
        );
        // Acceleration round-trip
        assert!(oem2.segments[0].states[0].acceleration.is_some());
        let acc1 = oem.segments[0].states[0].acceleration.unwrap();
        let acc2 = oem2.segments[0].states[0].acceleration.unwrap();
        assert!((acc2[0] - acc1[0]).abs() < 0.1);
        // Covariance round-trip
        assert_eq!(oem2.segments[0].covariances.len(), 1);
        let cov1 = &oem.segments[0].covariances[0];
        let cov2 = &oem2.segments[0].covariances[0];
        assert!((cov2.matrix[(0, 0)] - cov1.matrix[(0, 0)]).abs() < 1.0);
    }

    #[test]
    #[parallel]
    fn test_oem_from_file_nonexistent() {
        let result = OEM::from_file("nonexistent_file.txt");
        assert!(result.is_err());
    }

    /// Helper: assert all OEM fields match between original and round-tripped.
    fn assert_oem_fields_match(oem1: &OEM, oem2: &OEM) {
        // Header
        assert_eq!(oem1.header.format_version, oem2.header.format_version);
        assert_eq!(oem1.header.originator, oem2.header.originator);
        assert_eq!(oem1.header.classification, oem2.header.classification);
        assert_eq!(oem1.header.message_id, oem2.header.message_id);

        // Segments
        assert_eq!(oem1.segments.len(), oem2.segments.len());
        for (seg1, seg2) in oem1.segments.iter().zip(oem2.segments.iter()) {
            // Metadata
            assert_eq!(seg1.metadata.object_name, seg2.metadata.object_name);
            assert_eq!(seg1.metadata.object_id, seg2.metadata.object_id);
            assert_eq!(seg1.metadata.center_name, seg2.metadata.center_name);
            assert_eq!(seg1.metadata.ref_frame, seg2.metadata.ref_frame);
            assert_eq!(seg1.metadata.time_system, seg2.metadata.time_system);
            assert_eq!(seg1.metadata.interpolation, seg2.metadata.interpolation);
            assert_eq!(
                seg1.metadata.interpolation_degree,
                seg2.metadata.interpolation_degree
            );

            // States
            assert_eq!(seg1.states.len(), seg2.states.len());
            for (s1, s2) in seg1.states.iter().zip(seg2.states.iter()) {
                for i in 0..3 {
                    assert!(
                        (s1.position[i] - s2.position[i]).abs() < 1.0,
                        "position[{}] mismatch: {} vs {}",
                        i,
                        s1.position[i],
                        s2.position[i]
                    );
                    assert!(
                        (s1.velocity[i] - s2.velocity[i]).abs() < 0.001,
                        "velocity[{}] mismatch: {} vs {}",
                        i,
                        s1.velocity[i],
                        s2.velocity[i]
                    );
                }
                assert_eq!(s1.acceleration.is_some(), s2.acceleration.is_some());
                if let (Some(a1), Some(a2)) = (s1.acceleration, s2.acceleration) {
                    for i in 0..3 {
                        assert!((a1[i] - a2[i]).abs() < 0.1);
                    }
                }
            }

            // Covariances
            assert_eq!(seg1.covariances.len(), seg2.covariances.len());
            for (cov1, cov2) in seg1.covariances.iter().zip(seg2.covariances.iter()) {
                assert_eq!(cov1.cov_ref_frame, cov2.cov_ref_frame);
                for i in 0..6 {
                    for j in 0..6 {
                        let rel = if cov1.matrix[(i, j)].abs() > 1e-20 {
                            ((cov1.matrix[(i, j)] - cov2.matrix[(i, j)]) / cov1.matrix[(i, j)])
                                .abs()
                        } else {
                            (cov1.matrix[(i, j)] - cov2.matrix[(i, j)]).abs()
                        };
                        assert!(
                            rel < 1e-4,
                            "cov({},{}) mismatch: {} vs {}",
                            i,
                            j,
                            cov1.matrix[(i, j)],
                            cov2.matrix[(i, j)]
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[parallel]
    fn test_oem_kvn_full_round_trip() {
        // OEMExample5 has multiple state vectors and metadata
        let oem1 = OEM::from_file("test_assets/ccsds/oem/OEMExample5.txt").unwrap();
        let kvn = oem1.to_string(CCSDSFormat::KVN).unwrap();
        let oem2 = OEM::from_str(&kvn).unwrap();
        assert_oem_fields_match(&oem1, &oem2);
    }

    #[test]
    #[parallel]
    fn test_oem_xml_full_round_trip() {
        // OEMExample3.xml has acceleration + covariance + interpolation
        let oem1 = OEM::from_file("test_assets/ccsds/oem/OEMExample3.xml").unwrap();
        let xml = oem1.to_string(CCSDSFormat::XML).unwrap();
        let oem2 = OEM::from_str(&xml).unwrap();
        assert_oem_fields_match(&oem1, &oem2);
    }

    #[test]
    #[parallel]
    fn test_oem_json_full_round_trip() {
        let oem1 = OEM::from_file("test_assets/ccsds/oem/OEMExample3.xml").unwrap();
        let json = oem1.to_string(CCSDSFormat::JSON).unwrap();
        let oem2 = OEM::from_str(&json).unwrap();
        assert_oem_fields_match(&oem1, &oem2);
    }
}
