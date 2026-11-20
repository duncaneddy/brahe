/*!
 * CCSDS Orbit Mean-elements Message (OMM) data structures.
 *
 * OMM messages contain mean orbital elements and TLE-related parameters
 * for SGP4/SDP4 propagation. They are the standard format for exchanging
 * GP (General Perturbations) data.
 *
 * Reference: CCSDS 502.0-B-3 (Orbit Data Messages), April 2023
 */

use std::path::Path;

use crate::ccsds::common::{
    CCSDSCovariance, CCSDSFormat, CCSDSRefFrame, CCSDSSpacecraftParameters, CCSDSTimeSystem,
    CCSDSUserDefined, ODMHeader, ensure_compliant_write, odm_object_compliance_issues,
};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

/// A complete CCSDS Orbit Mean-elements Message.
#[derive(Debug, Clone)]
pub struct OMM {
    /// Message header
    pub header: ODMHeader,
    /// Metadata
    pub metadata: OMMMetadata,
    /// Mean Keplerian elements
    pub mean_elements: OMMeanElements,
    /// Optional TLE-related parameters
    pub tle_parameters: Option<OMMTleParameters>,
    /// Optional spacecraft physical parameters
    pub spacecraft_parameters: Option<CCSDSSpacecraftParameters>,
    /// Optional covariance matrix
    pub covariance: Option<CCSDSCovariance>,
    /// Optional user-defined parameters
    pub user_defined: Option<CCSDSUserDefined>,
    /// Comments associated with the metadata block
    pub comments: Vec<String>,
}

/// OMM metadata.
#[derive(Debug, Clone)]
pub struct OMMMetadata {
    /// Spacecraft name, `None` when the source omits `OBJECT_NAME` or leaves
    /// it empty. Mandatory in CCSDS 502.0-B-3; reported by `compliance_issues`.
    pub object_name: Option<String>,
    /// International designator (e.g., "1996-062A"), `None` when the source
    /// omits `OBJECT_ID` or leaves it empty. Mandatory in CCSDS 502.0-B-3;
    /// reported by `compliance_issues`.
    pub object_id: Option<String>,
    /// Center body name
    pub center_name: String,
    /// Reference frame
    pub ref_frame: CCSDSRefFrame,
    /// Optional reference frame epoch
    pub ref_frame_epoch: Option<Epoch>,
    /// Time system
    pub time_system: CCSDSTimeSystem,
    /// Mean element theory (e.g., "SGP/SGP4", "SGP4-XP")
    pub mean_element_theory: String,
    /// Comments
    pub comments: Vec<String>,
}

/// Mean Keplerian elements.
///
/// Units follow CCSDS/TLE conventions:
/// - Mean motion: rev/day
/// - Angles: degrees
/// - Eccentricity: dimensionless
/// - Semi-major axis: km (only present for non-SGP4 theories)
/// - GM: km³/s² in CCSDS, stored as m³/s² after conversion
#[derive(Debug, Clone)]
pub struct OMMeanElements {
    /// Epoch of the mean elements
    pub epoch: Epoch,
    /// Mean motion. Units: rev/day
    pub mean_motion: Option<f64>,
    /// Semi-major axis. Units: km (for non-SGP4 theories)
    pub semi_major_axis: Option<f64>,
    /// Eccentricity (dimensionless)
    pub eccentricity: f64,
    /// Inclination. Units: degrees
    pub inclination: f64,
    /// Right ascension of ascending node. Units: degrees
    pub ra_of_asc_node: f64,
    /// Argument of pericenter. Units: degrees
    pub arg_of_pericenter: f64,
    /// Mean anomaly. Units: degrees
    pub mean_anomaly: f64,
    /// Gravitational parameter. Units: m³/s² (converted from CCSDS km³/s²)
    pub gm: Option<f64>,
    /// Comments
    pub comments: Vec<String>,
}

/// TLE-related parameters for SGP4/SDP4 propagation.
#[derive(Debug, Clone)]
pub struct OMMTleParameters {
    /// Ephemeris type (0 = SGP4)
    pub ephemeris_type: Option<u32>,
    /// Classification type ('U' = unclassified)
    pub classification_type: Option<char>,
    /// NORAD catalog ID
    pub norad_cat_id: Option<u32>,
    /// Element set number
    pub element_set_no: Option<u32>,
    /// Revolution number at epoch
    pub rev_at_epoch: Option<u32>,
    /// BSTAR drag term (1/earth-radii)
    pub bstar: Option<f64>,
    /// Ballistic coefficient B-term
    pub bterm: Option<f64>,
    /// First derivative of mean motion (rev/day²)
    pub mean_motion_dot: Option<f64>,
    /// Second derivative of mean motion (rev/day³)
    pub mean_motion_ddot: Option<f64>,
    /// Solar radiation pressure coefficient AGOM
    pub agom: Option<f64>,
    /// Comments
    pub comments: Vec<String>,
}

impl OMMMetadata {
    /// Create new metadata with required fields.
    pub fn new(
        object_name: String,
        object_id: String,
        center_name: String,
        ref_frame: CCSDSRefFrame,
        time_system: CCSDSTimeSystem,
        mean_element_theory: String,
    ) -> Self {
        Self {
            object_name: Some(object_name),
            object_id: Some(object_id),
            center_name,
            ref_frame,
            ref_frame_epoch: None,
            time_system,
            mean_element_theory,
            comments: Vec::new(),
        }
    }
}

impl OMMeanElements {
    /// Create new mean elements with required fields.
    pub fn new(
        epoch: Epoch,
        eccentricity: f64,
        inclination: f64,
        ra_of_asc_node: f64,
        arg_of_pericenter: f64,
        mean_anomaly: f64,
    ) -> Self {
        Self {
            epoch,
            mean_motion: None,
            semi_major_axis: None,
            eccentricity,
            inclination,
            ra_of_asc_node,
            arg_of_pericenter,
            mean_anomaly,
            gm: None,
            comments: Vec::new(),
        }
    }

    /// Set mean motion.
    pub fn with_mean_motion(mut self, mean_motion: f64) -> Self {
        self.mean_motion = Some(mean_motion);
        self
    }

    /// Set GM.
    pub fn with_gm(mut self, gm: f64) -> Self {
        self.gm = Some(gm);
        self
    }
}

impl OMM {
    /// Create a new OMM message with required fields.
    ///
    /// # Arguments
    ///
    /// * `originator` - Originator of the message
    /// * `metadata` - OMM metadata
    /// * `mean_elements` - Mean Keplerian elements
    pub fn new(originator: String, metadata: OMMMetadata, mean_elements: OMMeanElements) -> Self {
        Self {
            header: ODMHeader {
                format_version: 3.0,
                classification: None,
                creation_date: Some(Epoch::now()),
                originator: Some(originator),
                message_id: None,
                comments: Vec::new(),
            },
            metadata,
            mean_elements,
            tle_parameters: None,
            spacecraft_parameters: None,
            covariance: None,
            user_defined: None,
            comments: Vec::new(),
        }
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
    /// use brahe::ccsds::OMM;
    ///
    /// let omm = OMM::from_file("test_assets/ccsds/omm/OMM-celestrak-empty-header.txt").unwrap();
    /// assert_eq!(omm.compliance_issues(), vec!["header.CREATION_DATE".to_string(), "header.ORIGINATOR".to_string()]);
    /// ```
    pub fn compliance_issues(&self) -> Vec<String> {
        let mut issues = self.header.compliance_issues();
        issues.extend(odm_object_compliance_issues(
            "metadata",
            &self.metadata.object_name,
            &self.metadata.object_id,
        ));
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
    /// use brahe::ccsds::OMM;
    ///
    /// let omm = OMM::from_file("test_assets/ccsds/omm/OMM-celestrak-empty-header.txt").unwrap();
    /// assert_eq!(omm.is_compliant(), vec!["header.CREATION_DATE".to_string(), "header.ORIGINATOR".to_string()].is_empty());
    /// ```
    pub fn is_compliant(&self) -> bool {
        self.compliance_issues().is_empty()
    }

    /// Parse an OMM message from a string, auto-detecting the format.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(content: &str) -> Result<Self, BraheError> {
        let format = crate::ccsds::common::detect_format(content);
        match format {
            CCSDSFormat::KVN => crate::ccsds::kvn::parse_omm(content),
            CCSDSFormat::XML => crate::ccsds::xml::parse_omm_xml(content),
            CCSDSFormat::JSON => crate::ccsds::json::parse_omm_json(content),
        }
    }

    /// Parse an OMM message from a file, auto-detecting the format.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, BraheError> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| BraheError::IoError(format!("Failed to read OMM file: {}", e)))?;
        Self::from_str(&content)
    }

    /// Write the OMM message to a string in the specified format.
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
    /// * `Result<String, BraheError>` - Serialized OMM string, or an error when
    ///   the message lacks mandatory keywords
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::ccsds::{CCSDSFormat, OMM};
    ///
    /// let mut omm = OMM::from_file("test_assets/ccsds/omm/OMMExample1.txt").unwrap();
    /// assert!(omm.to_string(CCSDSFormat::KVN).is_ok());
    ///
    /// omm.header.originator = None;
    /// assert!(omm.to_string(CCSDSFormat::KVN).is_err());
    /// ```
    pub fn to_string(&self, format: CCSDSFormat) -> Result<String, BraheError> {
        ensure_compliant_write("OMM", self.compliance_issues())?;
        self.to_string_noncompliant(format)
    }

    /// Write the OMM message to a string without checking CCSDS compliance.
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
    /// * `Result<String, BraheError>` - Serialized OMM string
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::ccsds::{CCSDSFormat, OMM};
    ///
    /// let mut omm = OMM::from_file("test_assets/ccsds/omm/OMMExample1.txt").unwrap();
    /// omm.header.originator = None;
    /// let kvn = omm.to_string_noncompliant(CCSDSFormat::KVN).unwrap();
    /// assert!(OMM::from_str(&kvn).unwrap().header.originator.is_none());
    /// ```
    pub fn to_string_noncompliant(&self, format: CCSDSFormat) -> Result<String, BraheError> {
        match format {
            CCSDSFormat::KVN => crate::ccsds::kvn::write_omm(self),
            CCSDSFormat::XML => crate::ccsds::xml::write_omm_xml(self),
            CCSDSFormat::JSON => crate::ccsds::json::write_omm_json(
                self,
                crate::ccsds::common::CCSDSJsonKeyCase::Lower,
            ),
        }
    }

    /// Write the OMM message to JSON with explicit key case control.
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
    /// use brahe::ccsds::OMM;
    /// use brahe::ccsds::common::CCSDSJsonKeyCase;
    ///
    /// let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample1.txt").unwrap();
    /// let json = omm.to_json_string(CCSDSJsonKeyCase::Upper).unwrap();
    /// assert!(json.contains("ORIGINATOR"));
    /// ```
    pub fn to_json_string(
        &self,
        key_case: crate::ccsds::common::CCSDSJsonKeyCase,
    ) -> Result<String, BraheError> {
        ensure_compliant_write("OMM", self.compliance_issues())?;
        self.to_json_string_noncompliant(key_case)
    }

    /// Write the OMM message to JSON without checking CCSDS compliance.
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
    /// use brahe::ccsds::OMM;
    /// use brahe::ccsds::common::CCSDSJsonKeyCase;
    ///
    /// let mut omm = OMM::from_file("test_assets/ccsds/omm/OMMExample1.txt").unwrap();
    /// omm.header.originator = None;
    /// assert!(omm.to_json_string_noncompliant(CCSDSJsonKeyCase::Upper).is_ok());
    /// ```
    pub fn to_json_string_noncompliant(
        &self,
        key_case: crate::ccsds::common::CCSDSJsonKeyCase,
    ) -> Result<String, BraheError> {
        crate::ccsds::json::write_omm_json(self, key_case)
    }

    /// Write the OMM message to a file in the specified format.
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
    /// use brahe::ccsds::{CCSDSFormat, OMM};
    ///
    /// let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample1.txt").unwrap();
    /// let path = std::env::temp_dir().join("brahe_omm_to_file_example.txt");
    /// omm.to_file(&path, CCSDSFormat::KVN).unwrap();
    /// ```
    pub fn to_file<P: AsRef<Path>>(&self, path: P, format: CCSDSFormat) -> Result<(), BraheError> {
        let content = self.to_string(format)?;
        Self::write_file(path.as_ref(), content)
    }

    /// Write the OMM message to a file without checking CCSDS compliance.
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
    /// use brahe::ccsds::{CCSDSFormat, OMM};
    ///
    /// let mut omm = OMM::from_file("test_assets/ccsds/omm/OMMExample1.txt").unwrap();
    /// omm.header.originator = None;
    /// let path = std::env::temp_dir().join("brahe_omm_to_file_noncompliant_example.txt");
    /// omm.to_file_noncompliant(&path, CCSDSFormat::KVN).unwrap();
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
            .map_err(|e| BraheError::IoError(format!("Failed to write OMM file: {}", e)))
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::ccsds::common::CCSDSJsonKeyCase;
    use crate::time::TimeSystem;
    use rstest::rstest;
    use serial_test::parallel;

    #[rstest]
    #[case("test_assets/ccsds/omm/OMM-celestrak-empty-header.txt")]
    #[case("test_assets/ccsds/omm/OMM-celestrak-empty-header.xml")]
    #[parallel]
    fn test_omm_parse_celestrak_empty_header(#[case] path: &str) {
        let omm = OMM::from_file(path).unwrap();

        assert!(omm.header.creation_date.is_none());
        assert_eq!(omm.header.originator, None);
        assert_eq!(omm.metadata.object_name.as_deref(), Some("ISS (ZARYA)"));
        assert_eq!(
            omm.compliance_issues(),
            vec!["header.CREATION_DATE", "header.ORIGINATOR"]
        );
        assert_eq!(
            omm.tle_parameters.as_ref().unwrap().norad_cat_id,
            Some(25544)
        );
    }

    #[rstest]
    #[case(CCSDSFormat::KVN)]
    #[case(CCSDSFormat::XML)]
    #[case(CCSDSFormat::JSON)]
    #[parallel]
    fn test_omm_empty_creation_date_round_trip(#[case] format: CCSDSFormat) {
        let omm = OMM::from_file("test_assets/ccsds/omm/OMM-celestrak-empty-header.txt").unwrap();

        let written = omm.to_string_noncompliant(format).unwrap();
        let omm2 = OMM::from_str(&written).unwrap();

        assert!(omm2.header.creation_date.is_none());
        assert_eq!(omm2.header.originator, None);
        assert_eq!(omm2.metadata.object_name, omm.metadata.object_name);
    }

    fn without_keyword(path: &str, keyword: &str) -> String {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .filter(|l| !l.starts_with(keyword))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[rstest]
    #[case("CREATION_DATE", "header.CREATION_DATE")]
    #[case("ORIGINATOR", "header.ORIGINATOR")]
    #[case("OBJECT_NAME", "metadata.OBJECT_NAME")]
    #[case("OBJECT_ID", "metadata.OBJECT_ID")]
    #[parallel]
    fn test_omm_compliance_issues_missing_keyword(#[case] keyword: &str, #[case] issue: &str) {
        let omm = OMM::from_str(&without_keyword(
            "test_assets/ccsds/omm/OMMExample1.txt",
            keyword,
        ))
        .unwrap();

        assert_eq!(omm.compliance_issues(), vec![issue.to_string()]);
        assert!(!omm.is_compliant());
    }

    #[test]
    #[parallel]
    fn test_omm_is_compliant() {
        let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample1.txt").unwrap();

        assert!(omm.compliance_issues().is_empty());
        assert!(omm.is_compliant());
    }

    #[rstest]
    #[case(CCSDSFormat::KVN)]
    #[case(CCSDSFormat::XML)]
    #[case(CCSDSFormat::JSON)]
    #[parallel]
    fn test_omm_strict_write_rejects_noncompliant(#[case] format: CCSDSFormat) {
        let mut omm = OMM::from_file("test_assets/ccsds/omm/OMMExample1.txt").unwrap();
        omm.header.originator = None;

        let err = omm.to_string(format).unwrap_err();
        assert!(err.to_string().contains("header.ORIGINATOR"), "{}", err);
        assert!(
            omm.to_json_string(crate::ccsds::common::CCSDSJsonKeyCase::Lower)
                .is_err()
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("omm.txt");
        assert!(omm.to_file(&path, format).is_err());
        assert!(!path.exists());

        let written = omm.to_string_noncompliant(format).unwrap();
        assert_eq!(OMM::from_str(&written).unwrap().header.originator, None);
        omm.to_file_noncompliant(&path, format).unwrap();
        assert!(path.exists());
    }

    #[test]
    #[parallel]
    fn test_omm_kvn_missing_epoch_errors() {
        let err = OMM::from_str(&without_keyword(
            "test_assets/ccsds/omm/OMMExample1.txt",
            "EPOCH",
        ))
        .unwrap_err();

        assert!(err.to_string().contains("EPOCH"), "{}", err);
    }

    #[test]
    #[parallel]
    fn test_omm_json_round_trip_via_dispatch() {
        let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample1.txt").unwrap();
        let json_str = omm.to_string(CCSDSFormat::JSON).unwrap();
        assert!(json_str.contains("object_name") || json_str.contains("OBJECT_NAME"));
        let omm2 = OMM::from_str(&json_str).unwrap();
        assert_eq!(omm2.metadata.object_name, omm.metadata.object_name);
        assert_eq!(omm2.metadata.object_id, omm.metadata.object_id);
        assert!((omm2.mean_elements.eccentricity - omm.mean_elements.eccentricity).abs() < 1e-10);
        assert!((omm2.mean_elements.inclination - omm.mean_elements.inclination).abs() < 1e-10);
    }

    #[test]
    #[parallel]
    fn test_omm_from_file_nonexistent() {
        let result = OMM::from_file("nonexistent_file.txt");
        assert!(result.is_err());
    }

    #[test]
    #[parallel]
    fn test_omm_metadata_new() {
        let meta = OMMMetadata::new(
            "ISS".to_string(),
            "1998-067A".to_string(),
            "EARTH".to_string(),
            CCSDSRefFrame::TEME,
            CCSDSTimeSystem::UTC,
            "SGP4".to_string(),
        );
        assert_eq!(meta.object_name.as_deref(), Some("ISS"));
        assert_eq!(meta.object_id.as_deref(), Some("1998-067A"));
        assert_eq!(meta.center_name, "EARTH");
        assert!(matches!(meta.ref_frame, CCSDSRefFrame::TEME));
        assert!(matches!(meta.time_system, CCSDSTimeSystem::UTC));
        assert_eq!(meta.mean_element_theory, "SGP4");
        assert!(meta.ref_frame_epoch.is_none());
        assert!(meta.comments.is_empty());
    }

    #[test]
    #[parallel]
    fn test_omm_mean_elements_new() {
        let epoch = Epoch::from_datetime(2024, 1, 15, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let elems = OMMeanElements::new(epoch, 0.001, 51.6, 120.0, 90.0, 45.0);
        assert!((elems.eccentricity - 0.001).abs() < 1e-15);
        assert!((elems.inclination - 51.6).abs() < 1e-15);
        assert!((elems.ra_of_asc_node - 120.0).abs() < 1e-15);
        assert!((elems.arg_of_pericenter - 90.0).abs() < 1e-15);
        assert!((elems.mean_anomaly - 45.0).abs() < 1e-15);
        assert!(elems.mean_motion.is_none());
        assert!(elems.semi_major_axis.is_none());
        assert!(elems.gm.is_none());
        assert!(elems.comments.is_empty());
    }

    #[test]
    #[parallel]
    fn test_omm_mean_elements_with_mean_motion() {
        let epoch = Epoch::from_datetime(2024, 1, 15, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let elems =
            OMMeanElements::new(epoch, 0.001, 51.6, 120.0, 90.0, 45.0).with_mean_motion(15.5);
        assert_eq!(elems.mean_motion, Some(15.5));
    }

    #[test]
    #[parallel]
    fn test_omm_mean_elements_with_gm() {
        let epoch = Epoch::from_datetime(2024, 1, 15, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let elems = OMMeanElements::new(epoch, 0.001, 51.6, 120.0, 90.0, 45.0).with_gm(398600.8e9);
        assert_eq!(elems.gm, Some(398600.8e9));
    }

    #[test]
    #[parallel]
    fn test_omm_new() {
        let meta = OMMMetadata::new(
            "SAT1".to_string(),
            "2024-001A".to_string(),
            "EARTH".to_string(),
            CCSDSRefFrame::TEME,
            CCSDSTimeSystem::UTC,
            "SGP/SGP4".to_string(),
        );
        let epoch = Epoch::from_datetime(2024, 1, 15, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let elems = OMMeanElements::new(epoch, 0.001, 51.6, 120.0, 90.0, 45.0);
        let omm = OMM::new("TEST_ORG".to_string(), meta, elems);

        assert_eq!(omm.header.originator.as_deref(), Some("TEST_ORG"));
        assert!((omm.header.format_version - 3.0).abs() < 1e-15);
        assert!(omm.header.classification.is_none());
        assert!(omm.header.message_id.is_none());
        assert_eq!(omm.metadata.object_name.as_deref(), Some("SAT1"));
        assert_eq!(omm.metadata.object_id.as_deref(), Some("2024-001A"));
        assert!((omm.mean_elements.eccentricity - 0.001).abs() < 1e-15);
        assert!(omm.tle_parameters.is_none());
        assert!(omm.spacecraft_parameters.is_none());
        assert!(omm.covariance.is_none());
        assert!(omm.user_defined.is_none());
        assert!(omm.comments.is_empty());
    }

    #[test]
    #[parallel]
    fn test_omm_kvn_parse_example1() {
        let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample1.txt").unwrap();
        assert_eq!(omm.metadata.object_name.as_deref(), Some("GOES 9"));
        assert_eq!(omm.metadata.object_id.as_deref(), Some("1995-025A"));
        assert_eq!(omm.metadata.center_name, "EARTH");
        assert_eq!(omm.metadata.mean_element_theory, "SGP/SGP4");
        assert!((omm.mean_elements.eccentricity - 0.0005013).abs() < 1e-10);
        assert!((omm.mean_elements.inclination - 3.0539).abs() < 1e-10);
        assert!((omm.mean_elements.ra_of_asc_node - 81.7939).abs() < 1e-10);
        assert!((omm.mean_elements.arg_of_pericenter - 249.2363).abs() < 1e-10);
        assert!((omm.mean_elements.mean_anomaly - 150.1602).abs() < 1e-10);
        assert!((omm.mean_elements.mean_motion.unwrap() - 1.00273272).abs() < 1e-10);
        // TLE parameters
        let tle = omm.tle_parameters.as_ref().unwrap();
        assert_eq!(tle.norad_cat_id, Some(23581));
        assert_eq!(tle.element_set_no, Some(925));
        assert_eq!(tle.rev_at_epoch, Some(4316));
        assert!((tle.bstar.unwrap() - 0.0001).abs() < 1e-10);
    }

    #[test]
    #[parallel]
    fn test_omm_kvn_parse_example2_with_covariance() {
        // OMMExample2 has covariance data
        let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample2.txt").unwrap();
        assert_eq!(omm.metadata.object_name.as_deref(), Some("GOES 9"));
        assert_eq!(omm.metadata.object_id.as_deref(), Some("1995-025A"));
        assert!(omm.covariance.is_some());
        let cov = omm.covariance.as_ref().unwrap();
        // CX_X = 3.331349476038534e-04 km^2 -> * 1e6 m^2
        assert!((cov.matrix[(0, 0)] - 3.331349476038534e-04 * 1e6).abs() < 1e-2);
    }

    #[test]
    #[parallel]
    fn test_omm_kvn_round_trip() {
        let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample1.txt").unwrap();
        let kvn_str = omm.to_string(CCSDSFormat::KVN).unwrap();
        let omm2 = OMM::from_str(&kvn_str).unwrap();
        assert_eq!(omm2.metadata.object_name, omm.metadata.object_name);
        assert_eq!(omm2.metadata.object_id, omm.metadata.object_id);
        assert!((omm2.mean_elements.eccentricity - omm.mean_elements.eccentricity).abs() < 1e-10);
        assert!((omm2.mean_elements.inclination - omm.mean_elements.inclination).abs() < 1e-10);
        assert!(
            (omm2.mean_elements.mean_motion.unwrap() - omm.mean_elements.mean_motion.unwrap())
                .abs()
                < 1e-10
        );
        // GM round-trip (m³/s² → km³/s² → m³/s²)
        assert!((omm2.mean_elements.gm.unwrap() - omm.mean_elements.gm.unwrap()).abs() < 1e3);
        // TLE parameters
        let tle1 = omm.tle_parameters.as_ref().unwrap();
        let tle2 = omm2.tle_parameters.as_ref().unwrap();
        assert_eq!(tle2.norad_cat_id, tle1.norad_cat_id);
        assert!((tle2.bstar.unwrap() - tle1.bstar.unwrap()).abs() < 1e-10);
    }

    #[test]
    #[parallel]
    fn test_omm_kvn_round_trip_with_covariance() {
        let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample2.txt").unwrap();
        let kvn_str = omm.to_string(CCSDSFormat::KVN).unwrap();
        let omm2 = OMM::from_str(&kvn_str).unwrap();
        assert!(omm2.covariance.is_some());
        let cov1 = omm.covariance.as_ref().unwrap();
        let cov2 = omm2.covariance.as_ref().unwrap();
        // CX_X round-trip: m² → km² → m²
        assert!((cov2.matrix[(0, 0)] - cov1.matrix[(0, 0)]).abs() < 1.0);
    }

    #[test]
    #[parallel]
    fn test_omm_xml_round_trip() {
        let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample2.xml").unwrap();
        let xml_str = omm.to_string(CCSDSFormat::XML).unwrap();
        let omm2 = OMM::from_str(&xml_str).unwrap();
        assert_eq!(omm2.metadata.object_name, omm.metadata.object_name);
        assert_eq!(omm2.metadata.object_id, omm.metadata.object_id);
        assert!((omm2.mean_elements.eccentricity - omm.mean_elements.eccentricity).abs() < 1e-10);
        assert!(
            (omm2.mean_elements.mean_motion.unwrap() - omm.mean_elements.mean_motion.unwrap())
                .abs()
                < 1e-10
        );
        // Covariance round-trip
        assert!(omm2.covariance.is_some());
        let cov1 = omm.covariance.as_ref().unwrap();
        let cov2 = omm2.covariance.as_ref().unwrap();
        assert!((cov2.matrix[(0, 0)] - cov1.matrix[(0, 0)]).abs() < 1.0);
    }

    #[test]
    #[parallel]
    fn test_omm_xml_parse_example4() {
        let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample4.xml").unwrap();
        assert_eq!(omm.metadata.object_name.as_deref(), Some("STARLETTE"));
        assert_eq!(omm.metadata.object_id.as_deref(), Some("1975-010A"));
        assert_eq!(omm.metadata.mean_element_theory, "SGP4");
        assert!((omm.mean_elements.mean_motion.unwrap() - 13.82309053).abs() < 1e-8);
        let tle = omm.tle_parameters.as_ref().unwrap();
        assert_eq!(tle.norad_cat_id, Some(7646));
    }

    #[test]
    #[parallel]
    fn test_omm_to_file_kvn() {
        let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample1.txt").unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join("brahe_test_omm.txt");
        omm.to_file(&path, CCSDSFormat::KVN).unwrap();
        let omm2 = OMM::from_file(&path).unwrap();
        assert_eq!(omm2.metadata.object_name, omm.metadata.object_name);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    #[parallel]
    fn test_omm_kvn_parse_with_gm() {
        // OMMExample1 has GM = 398600.8, verify it is parsed and converted to m^3/s^2
        let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample1.txt").unwrap();
        assert!(omm.mean_elements.gm.is_some());
        let gm = omm.mean_elements.gm.unwrap();
        // GM in file is 398600.8 km^3/s^2 = 398600.8e9 m^3/s^2
        assert!((gm - 398600.8e9).abs() < 1e6);
    }

    #[test]
    #[parallel]
    fn test_omm_to_json_string_upper_key_case() {
        let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample1.txt").unwrap();
        let json_str = omm.to_json_string(CCSDSJsonKeyCase::Upper).unwrap();
        assert!(json_str.contains("OBJECT_NAME"));
        assert!(json_str.contains("ECCENTRICITY"));
    }

    /// Helper: assert all OMM fields match between original and round-tripped.
    fn assert_omm_fields_match(omm1: &OMM, omm2: &OMM) {
        // Header
        assert_eq!(omm1.header.format_version, omm2.header.format_version);
        assert_eq!(omm1.header.originator, omm2.header.originator);
        assert_eq!(omm1.header.classification, omm2.header.classification);
        assert_eq!(omm1.header.message_id, omm2.header.message_id);

        // Metadata
        assert_eq!(omm1.metadata.object_name, omm2.metadata.object_name);
        assert_eq!(omm1.metadata.object_id, omm2.metadata.object_id);
        assert_eq!(omm1.metadata.center_name, omm2.metadata.center_name);
        assert_eq!(omm1.metadata.ref_frame, omm2.metadata.ref_frame);
        assert_eq!(omm1.metadata.time_system, omm2.metadata.time_system);
        assert_eq!(
            omm1.metadata.mean_element_theory,
            omm2.metadata.mean_element_theory
        );

        // Mean elements
        assert!((omm1.mean_elements.eccentricity - omm2.mean_elements.eccentricity).abs() < 1e-10);
        assert!((omm1.mean_elements.inclination - omm2.mean_elements.inclination).abs() < 1e-6);
        assert!(
            (omm1.mean_elements.ra_of_asc_node - omm2.mean_elements.ra_of_asc_node).abs() < 1e-6
        );
        assert!(
            (omm1.mean_elements.arg_of_pericenter - omm2.mean_elements.arg_of_pericenter).abs()
                < 1e-6
        );
        assert!((omm1.mean_elements.mean_anomaly - omm2.mean_elements.mean_anomaly).abs() < 1e-6);
        assert_eq!(
            omm1.mean_elements.mean_motion.is_some(),
            omm2.mean_elements.mean_motion.is_some()
        );
        if let (Some(mm1), Some(mm2)) = (
            omm1.mean_elements.mean_motion,
            omm2.mean_elements.mean_motion,
        ) {
            assert!((mm1 - mm2).abs() < 1e-10);
        }
        assert_eq!(
            omm1.mean_elements.semi_major_axis.is_some(),
            omm2.mean_elements.semi_major_axis.is_some()
        );
        assert_eq!(
            omm1.mean_elements.gm.is_some(),
            omm2.mean_elements.gm.is_some()
        );
        if let (Some(gm1), Some(gm2)) = (omm1.mean_elements.gm, omm2.mean_elements.gm) {
            assert!((gm1 - gm2).abs() < 1e3);
        }

        // TLE parameters
        assert_eq!(omm1.tle_parameters.is_some(), omm2.tle_parameters.is_some());
        if let (Some(t1), Some(t2)) = (&omm1.tle_parameters, &omm2.tle_parameters) {
            assert_eq!(t1.ephemeris_type, t2.ephemeris_type);
            assert_eq!(t1.classification_type, t2.classification_type);
            assert_eq!(t1.norad_cat_id, t2.norad_cat_id);
            assert_eq!(t1.element_set_no, t2.element_set_no);
            assert_eq!(t1.rev_at_epoch, t2.rev_at_epoch);
            assert_eq!(t1.bstar.is_some(), t2.bstar.is_some());
            if let (Some(b1), Some(b2)) = (t1.bstar, t2.bstar) {
                assert!((b1 - b2).abs() < 1e-10);
            }
            assert_eq!(t1.mean_motion_dot.is_some(), t2.mean_motion_dot.is_some());
            if let (Some(d1), Some(d2)) = (t1.mean_motion_dot, t2.mean_motion_dot) {
                assert!((d1 - d2).abs() < 1e-12);
            }
            assert_eq!(t1.mean_motion_ddot.is_some(), t2.mean_motion_ddot.is_some());
            if let (Some(d1), Some(d2)) = (t1.mean_motion_ddot, t2.mean_motion_ddot) {
                assert!((d1 - d2).abs() < 1e-12);
            }
        }

        // Spacecraft parameters
        assert_eq!(
            omm1.spacecraft_parameters.is_some(),
            omm2.spacecraft_parameters.is_some()
        );

        // Covariance
        assert_eq!(omm1.covariance.is_some(), omm2.covariance.is_some());
        if let (Some(cov1), Some(cov2)) = (&omm1.covariance, &omm2.covariance) {
            assert_eq!(cov1.cov_ref_frame, cov2.cov_ref_frame);
            for i in 0..6 {
                for j in 0..6 {
                    let rel = if cov1.matrix[(i, j)].abs() > 1e-20 {
                        ((cov1.matrix[(i, j)] - cov2.matrix[(i, j)]) / cov1.matrix[(i, j)]).abs()
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

        // User-defined
        assert_eq!(omm1.user_defined.is_some(), omm2.user_defined.is_some());
        if let (Some(ud1), Some(ud2)) = (&omm1.user_defined, &omm2.user_defined) {
            assert_eq!(ud1.parameters.len(), ud2.parameters.len());
            for (k, v) in &ud1.parameters {
                assert_eq!(ud2.parameters.get(k), Some(v));
            }
        }
    }

    #[test]
    #[parallel]
    fn test_omm_kvn_full_round_trip() {
        // OMMExample2 has covariance + full TLE parameters + GM
        let omm1 = OMM::from_file("test_assets/ccsds/omm/OMMExample2.txt").unwrap();
        let kvn = omm1.to_string(CCSDSFormat::KVN).unwrap();
        let omm2 = OMM::from_str(&kvn).unwrap();
        assert_omm_fields_match(&omm1, &omm2);
    }

    #[test]
    #[parallel]
    fn test_omm_xml_full_round_trip() {
        // Start from KVN (richest fields), write to XML, re-parse
        let omm1 = OMM::from_file("test_assets/ccsds/omm/OMMExample2.txt").unwrap();
        let xml = omm1.to_string(CCSDSFormat::XML).unwrap();
        let omm2 = OMM::from_str(&xml).unwrap();
        assert_omm_fields_match(&omm1, &omm2);
    }

    #[test]
    #[parallel]
    fn test_omm_json_full_round_trip() {
        let omm1 = OMM::from_file("test_assets/ccsds/omm/OMMExample2.txt").unwrap();
        let json = omm1.to_string(CCSDSFormat::JSON).unwrap();
        let omm2 = OMM::from_str(&json).unwrap();
        assert_omm_fields_match(&omm1, &omm2);
    }
}
