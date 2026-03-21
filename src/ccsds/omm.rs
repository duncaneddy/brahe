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
    CCSDSUserDefined, ODMHeader,
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
    /// Spacecraft name
    pub object_name: String,
    /// International designator
    pub object_id: String,
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
            object_name,
            object_id,
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
                creation_date: Epoch::now(),
                originator,
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
    pub fn to_string(&self, format: CCSDSFormat) -> Result<String, BraheError> {
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
    pub fn to_json_string(
        &self,
        key_case: crate::ccsds::common::CCSDSJsonKeyCase,
    ) -> Result<String, BraheError> {
        crate::ccsds::json::write_omm_json(self, key_case)
    }

    /// Write the OMM message to a file in the specified format.
    pub fn to_file<P: AsRef<Path>>(&self, path: P, format: CCSDSFormat) -> Result<(), BraheError> {
        let content = self.to_string(format)?;
        std::fs::write(path.as_ref(), content)
            .map_err(|e| BraheError::IoError(format!("Failed to write OMM file: {}", e)))
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::ccsds::common::CCSDSJsonKeyCase;
    use crate::time::TimeSystem;

    #[test]
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
    fn test_omm_from_file_nonexistent() {
        let result = OMM::from_file("nonexistent_file.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_omm_metadata_new() {
        let meta = OMMMetadata::new(
            "ISS".to_string(),
            "1998-067A".to_string(),
            "EARTH".to_string(),
            CCSDSRefFrame::TEME,
            CCSDSTimeSystem::UTC,
            "SGP4".to_string(),
        );
        assert_eq!(meta.object_name, "ISS");
        assert_eq!(meta.object_id, "1998-067A");
        assert_eq!(meta.center_name, "EARTH");
        assert!(matches!(meta.ref_frame, CCSDSRefFrame::TEME));
        assert!(matches!(meta.time_system, CCSDSTimeSystem::UTC));
        assert_eq!(meta.mean_element_theory, "SGP4");
        assert!(meta.ref_frame_epoch.is_none());
        assert!(meta.comments.is_empty());
    }

    #[test]
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
    fn test_omm_mean_elements_with_mean_motion() {
        let epoch = Epoch::from_datetime(2024, 1, 15, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let elems =
            OMMeanElements::new(epoch, 0.001, 51.6, 120.0, 90.0, 45.0).with_mean_motion(15.5);
        assert_eq!(elems.mean_motion, Some(15.5));
    }

    #[test]
    fn test_omm_mean_elements_with_gm() {
        let epoch = Epoch::from_datetime(2024, 1, 15, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let elems = OMMeanElements::new(epoch, 0.001, 51.6, 120.0, 90.0, 45.0).with_gm(398600.8e9);
        assert_eq!(elems.gm, Some(398600.8e9));
    }

    #[test]
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

        assert_eq!(omm.header.originator, "TEST_ORG");
        assert!((omm.header.format_version - 3.0).abs() < 1e-15);
        assert!(omm.header.classification.is_none());
        assert!(omm.header.message_id.is_none());
        assert_eq!(omm.metadata.object_name, "SAT1");
        assert_eq!(omm.metadata.object_id, "2024-001A");
        assert!((omm.mean_elements.eccentricity - 0.001).abs() < 1e-15);
        assert!(omm.tle_parameters.is_none());
        assert!(omm.spacecraft_parameters.is_none());
        assert!(omm.covariance.is_none());
        assert!(omm.user_defined.is_none());
        assert!(omm.comments.is_empty());
    }

    #[test]
    fn test_omm_kvn_parse_example1() {
        let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample1.txt").unwrap();
        assert_eq!(omm.metadata.object_name, "GOES 9");
        assert_eq!(omm.metadata.object_id, "1995-025A");
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
    fn test_omm_kvn_parse_example2_with_covariance() {
        // OMMExample2 has covariance data
        let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample2.txt").unwrap();
        assert_eq!(omm.metadata.object_name, "GOES 9");
        assert_eq!(omm.metadata.object_id, "1995-025A");
        assert!(omm.covariance.is_some());
        let cov = omm.covariance.as_ref().unwrap();
        // CX_X = 3.331349476038534e-04 km^2 -> * 1e6 m^2
        assert!((cov.matrix[(0, 0)] - 3.331349476038534e-04 * 1e6).abs() < 1e-2);
    }

    #[test]
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
    fn test_omm_xml_parse_example4() {
        let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample4.xml").unwrap();
        assert_eq!(omm.metadata.object_name, "STARLETTE");
        assert_eq!(omm.metadata.object_id, "1975-010A");
        assert_eq!(omm.metadata.mean_element_theory, "SGP4");
        assert!((omm.mean_elements.mean_motion.unwrap() - 13.82309053).abs() < 1e-8);
        let tle = omm.tle_parameters.as_ref().unwrap();
        assert_eq!(tle.norad_cat_id, Some(7646));
    }

    #[test]
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
    fn test_omm_kvn_parse_with_gm() {
        // OMMExample1 has GM = 398600.8, verify it is parsed and converted to m^3/s^2
        let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample1.txt").unwrap();
        assert!(omm.mean_elements.gm.is_some());
        let gm = omm.mean_elements.gm.unwrap();
        // GM in file is 398600.8 km^3/s^2 = 398600.8e9 m^3/s^2
        assert!((gm - 398600.8e9).abs() < 1e6);
    }

    #[test]
    fn test_omm_to_json_string_upper_key_case() {
        let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample1.txt").unwrap();
        let json_str = omm.to_json_string(CCSDSJsonKeyCase::Upper).unwrap();
        assert!(json_str.contains("OBJECT_NAME"));
        assert!(json_str.contains("ECCENTRICITY"));
    }
}
