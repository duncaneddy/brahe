/*!
 * CCSDS Orbit Parameter Message (OPM) data structures.
 *
 * OPM messages contain a single state vector (position and velocity) with
 * optional Keplerian elements, spacecraft parameters, maneuvers, and
 * covariance data.
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

/// A complete CCSDS Orbit Parameter Message.
#[derive(Debug, Clone)]
pub struct OPM {
    /// Message header
    pub header: ODMHeader,
    /// Metadata
    pub metadata: OPMMetadata,
    /// State vector
    pub state_vector: OPMStateVector,
    /// Optional Keplerian elements
    pub keplerian_elements: Option<OPMKeplerianElements>,
    /// Optional spacecraft parameters
    pub spacecraft_parameters: Option<CCSDSSpacecraftParameters>,
    /// Optional covariance matrix
    pub covariance: Option<CCSDSCovariance>,
    /// Maneuvers
    pub maneuvers: Vec<OPMManeuver>,
    /// Optional user-defined parameters
    pub user_defined: Option<CCSDSUserDefined>,
}

/// OPM metadata.
#[derive(Debug, Clone)]
pub struct OPMMetadata {
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
    /// Comments
    pub comments: Vec<String>,
}

/// State vector data in an OPM.
///
/// Position and velocity stored in SI units (meters, m/s).
#[derive(Debug, Clone)]
pub struct OPMStateVector {
    /// Epoch of the state vector
    pub epoch: Epoch,
    /// Position [x, y, z]. Units: meters
    pub position: [f64; 3],
    /// Velocity [vx, vy, vz]. Units: m/s
    pub velocity: [f64; 3],
    /// Comments
    pub comments: Vec<String>,
}

/// Keplerian elements in an OPM.
///
/// Semi-major axis stored in meters (SI). Angles in degrees (CCSDS native).
#[derive(Debug, Clone)]
pub struct OPMKeplerianElements {
    /// Semi-major axis. Units: meters (converted from km)
    pub semi_major_axis: f64,
    /// Eccentricity (dimensionless)
    pub eccentricity: f64,
    /// Inclination. Units: degrees
    pub inclination: f64,
    /// Right ascension of ascending node. Units: degrees
    pub ra_of_asc_node: f64,
    /// Argument of pericenter. Units: degrees
    pub arg_of_pericenter: f64,
    /// True anomaly. Units: degrees
    pub true_anomaly: Option<f64>,
    /// Mean anomaly. Units: degrees
    pub mean_anomaly: Option<f64>,
    /// Gravitational parameter. Units: m³/s² (converted from km³/s²)
    pub gm: Option<f64>,
    /// Comments
    pub comments: Vec<String>,
}

impl OPMMetadata {
    /// Create new metadata with required fields.
    pub fn new(
        object_name: String,
        object_id: String,
        center_name: String,
        ref_frame: CCSDSRefFrame,
        time_system: CCSDSTimeSystem,
    ) -> Self {
        Self {
            object_name,
            object_id,
            center_name,
            ref_frame,
            ref_frame_epoch: None,
            time_system,
            comments: Vec::new(),
        }
    }
}

impl OPMStateVector {
    /// Create a new state vector.
    ///
    /// # Arguments
    ///
    /// * `epoch` - Epoch of the state vector
    /// * `position` - Position [x, y, z]. Units: meters
    /// * `velocity` - Velocity [vx, vy, vz]. Units: m/s
    pub fn new(epoch: Epoch, position: [f64; 3], velocity: [f64; 3]) -> Self {
        Self {
            epoch,
            position,
            velocity,
            comments: Vec::new(),
        }
    }
}

/// A maneuver specification in an OPM.
///
/// Delta-V stored in m/s (SI).
#[derive(Debug, Clone)]
pub struct OPMManeuver {
    /// Epoch of ignition
    pub epoch_ignition: Epoch,
    /// Duration. Units: seconds
    pub duration: f64,
    /// Mass change. Units: kg (negative for mass decrease)
    pub delta_mass: Option<f64>,
    /// Reference frame for the delta-V
    pub ref_frame: CCSDSRefFrame,
    /// Delta-V vector [dv1, dv2, dv3]. Units: m/s
    pub dv: [f64; 3],
    /// Comments
    pub comments: Vec<String>,
}

impl OPMManeuver {
    /// Create a new maneuver.
    ///
    /// # Arguments
    ///
    /// * `epoch_ignition` - Epoch of ignition
    /// * `duration` - Duration in seconds
    /// * `ref_frame` - Reference frame for the delta-V
    /// * `dv` - Delta-V vector [dv1, dv2, dv3]. Units: m/s
    pub fn new(
        epoch_ignition: Epoch,
        duration: f64,
        ref_frame: CCSDSRefFrame,
        dv: [f64; 3],
    ) -> Self {
        Self {
            epoch_ignition,
            duration,
            delta_mass: None,
            ref_frame,
            dv,
            comments: Vec::new(),
        }
    }

    /// Set the delta mass.
    pub fn with_delta_mass(mut self, delta_mass: f64) -> Self {
        self.delta_mass = Some(delta_mass);
        self
    }
}

impl OPM {
    /// Create a new OPM message with required fields.
    ///
    /// # Arguments
    ///
    /// * `originator` - Originator of the message
    /// * `metadata` - OPM metadata
    /// * `state_vector` - State vector data
    pub fn new(originator: String, metadata: OPMMetadata, state_vector: OPMStateVector) -> Self {
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
            state_vector,
            keplerian_elements: None,
            spacecraft_parameters: None,
            covariance: None,
            maneuvers: Vec::new(),
            user_defined: None,
        }
    }

    /// Add a maneuver to the OPM.
    pub fn push_maneuver(&mut self, maneuver: OPMManeuver) {
        self.maneuvers.push(maneuver);
    }

    /// Parse an OPM message from a string, auto-detecting the format.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(content: &str) -> Result<Self, BraheError> {
        let format = crate::ccsds::common::detect_format(content);
        match format {
            CCSDSFormat::KVN => crate::ccsds::kvn::parse_opm(content),
            CCSDSFormat::XML => crate::ccsds::xml::parse_opm_xml(content),
            CCSDSFormat::JSON => crate::ccsds::json::parse_opm_json(content),
        }
    }

    /// Parse an OPM message from a file, auto-detecting the format.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, BraheError> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| BraheError::IoError(format!("Failed to read OPM file: {}", e)))?;
        Self::from_str(&content)
    }

    /// Write the OPM message to a string in the specified format.
    pub fn to_string(&self, format: CCSDSFormat) -> Result<String, BraheError> {
        match format {
            CCSDSFormat::KVN => crate::ccsds::kvn::write_opm(self),
            CCSDSFormat::XML => crate::ccsds::xml::write_opm_xml(self),
            CCSDSFormat::JSON => crate::ccsds::json::write_opm_json(
                self,
                crate::ccsds::common::CCSDSJsonKeyCase::Lower,
            ),
        }
    }

    /// Write the OPM message to JSON with explicit key case control.
    pub fn to_json_string(
        &self,
        key_case: crate::ccsds::common::CCSDSJsonKeyCase,
    ) -> Result<String, BraheError> {
        crate::ccsds::json::write_opm_json(self, key_case)
    }

    /// Write the OPM message to a file in the specified format.
    pub fn to_file<P: AsRef<Path>>(&self, path: P, format: CCSDSFormat) -> Result<(), BraheError> {
        let content = self.to_string(format)?;
        std::fs::write(path.as_ref(), content)
            .map_err(|e| BraheError::IoError(format!("Failed to write OPM file: {}", e)))
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::ccsds::common::CCSDSJsonKeyCase;
    use crate::time::TimeSystem;

    #[test]
    fn test_opm_builder() {
        let metadata = OPMMetadata::new(
            "SAT1".to_string(),
            "2024-001A".to_string(),
            "EARTH".to_string(),
            CCSDSRefFrame::GCRF,
            CCSDSTimeSystem::UTC,
        );
        let sv = OPMStateVector::new(Epoch::now(), [7000e3, 0.0, 0.0], [0.0, 7500.0, 0.0]);
        let mut opm = OPM::new("TEST_ORG".to_string(), metadata, sv);
        assert_eq!(opm.header.originator, "TEST_ORG");
        assert_eq!(opm.maneuvers.len(), 0);

        let m = OPMManeuver::new(Epoch::now(), 120.0, CCSDSRefFrame::RTN, [10.0, 0.0, 0.0])
            .with_delta_mass(-15.0);
        opm.push_maneuver(m);
        assert_eq!(opm.maneuvers.len(), 1);
        assert_eq!(opm.maneuvers[0].delta_mass, Some(-15.0));
    }

    #[test]
    fn test_opm_json_round_trip_via_dispatch() {
        let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample1.txt").unwrap();
        let json_str = opm.to_string(CCSDSFormat::JSON).unwrap();
        assert!(json_str.contains("object_name") || json_str.contains("OBJECT_NAME"));
        let opm2 = OPM::from_str(&json_str).unwrap();
        assert_eq!(opm2.metadata.object_name, opm.metadata.object_name);
        assert_eq!(opm2.metadata.object_id, opm.metadata.object_id);
        assert!((opm2.state_vector.position[0] - opm.state_vector.position[0]).abs() < 1.0);
        assert!((opm2.state_vector.velocity[0] - opm.state_vector.velocity[0]).abs() < 0.001);
    }

    #[test]
    fn test_opm_from_file_nonexistent() {
        let result = OPM::from_file("nonexistent_file.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_opm_metadata_new() {
        let meta = OPMMetadata::new(
            "ISS".to_string(),
            "1998-067A".to_string(),
            "EARTH".to_string(),
            CCSDSRefFrame::ITRF2000,
            CCSDSTimeSystem::UTC,
        );
        assert_eq!(meta.object_name, "ISS");
        assert_eq!(meta.object_id, "1998-067A");
        assert_eq!(meta.center_name, "EARTH");
        assert!(matches!(meta.ref_frame, CCSDSRefFrame::ITRF2000));
        assert!(matches!(meta.time_system, CCSDSTimeSystem::UTC));
        assert!(meta.ref_frame_epoch.is_none());
        assert!(meta.comments.is_empty());
    }

    #[test]
    fn test_opm_state_vector_new() {
        let epoch = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let pos = [6503.514e3, 1239.647e3, -717.490e3];
        let vel = [-873.160, 8740.420, -4191.076];
        let sv = OPMStateVector::new(epoch, pos, vel);

        assert!((sv.position[0] - 6503.514e3).abs() < 1e-6);
        assert!((sv.position[1] - 1239.647e3).abs() < 1e-6);
        assert!((sv.position[2] - (-717.490e3)).abs() < 1e-6);
        assert!((sv.velocity[0] - (-873.160)).abs() < 1e-6);
        assert!((sv.velocity[1] - 8740.420).abs() < 1e-6);
        assert!((sv.velocity[2] - (-4191.076)).abs() < 1e-6);
        assert!(sv.comments.is_empty());
    }

    #[test]
    fn test_opm_new() {
        let meta = OPMMetadata::new(
            "SAT1".to_string(),
            "2024-001A".to_string(),
            "EARTH".to_string(),
            CCSDSRefFrame::GCRF,
            CCSDSTimeSystem::UTC,
        );
        let epoch = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let sv = OPMStateVector::new(epoch, [7000e3, 0.0, 0.0], [0.0, 7500.0, 0.0]);
        let opm = OPM::new("TEST_ORG".to_string(), meta, sv);

        assert_eq!(opm.header.originator, "TEST_ORG");
        assert!((opm.header.format_version - 3.0).abs() < 1e-15);
        assert!(opm.header.classification.is_none());
        assert!(opm.header.message_id.is_none());
        assert_eq!(opm.metadata.object_name, "SAT1");
        assert_eq!(opm.metadata.object_id, "2024-001A");
        assert!(opm.keplerian_elements.is_none());
        assert!(opm.spacecraft_parameters.is_none());
        assert!(opm.covariance.is_none());
        assert!(opm.maneuvers.is_empty());
        assert!(opm.user_defined.is_none());
    }

    #[test]
    fn test_opm_kvn_parse_example1() {
        let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample1.txt").unwrap();
        assert_eq!(opm.metadata.object_name, "GODZILLA 5");
        assert_eq!(opm.metadata.object_id, "1998-999A");
        assert_eq!(opm.metadata.center_name, "EARTH");
        assert!(matches!(opm.metadata.ref_frame, CCSDSRefFrame::ITRF2000));
        assert!(matches!(opm.metadata.time_system, CCSDSTimeSystem::UTC));
        // Position in meters (converted from km in file)
        assert!((opm.state_vector.position[0] - 6503.514e3).abs() < 1.0);
        assert!((opm.state_vector.position[1] - 1239.647e3).abs() < 1.0);
        assert!((opm.state_vector.position[2] - (-717.490e3)).abs() < 1.0);
        // Velocity in m/s (converted from km/s in file)
        assert!((opm.state_vector.velocity[0] - (-873.160)).abs() < 0.01);
        assert!((opm.state_vector.velocity[1] - 8740.420).abs() < 0.01);
        assert!((opm.state_vector.velocity[2] - (-4191.076)).abs() < 0.01);
    }

    #[test]
    fn test_opm_with_keplerian_elements() {
        let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample5.txt").unwrap();
        assert!(
            opm.keplerian_elements.is_some(),
            "OPMExample5 should have Keplerian elements"
        );
        let ke = opm.keplerian_elements.as_ref().unwrap();
        assert!((ke.eccentricity - 0.020842611).abs() < 1e-9);
        assert!((ke.inclination - 0.117746).abs() < 1e-6);
        assert!((ke.ra_of_asc_node - 17.604721).abs() < 1e-6);
        assert!((ke.arg_of_pericenter - 218.242943).abs() < 1e-6);
        assert!(ke.true_anomaly.is_some());
        assert!((ke.true_anomaly.unwrap() - 41.922339).abs() < 1e-6);
    }

    #[test]
    fn test_opm_with_maneuvers() {
        // OPMExample5 has 3 maneuvers
        let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample5.txt").unwrap();
        assert!(
            opm.maneuvers.len() >= 2,
            "OPMExample5 should have maneuvers"
        );
        // Verify first maneuver has expected delta_mass
        assert!(opm.maneuvers[0].delta_mass.is_some());
        assert!((opm.maneuvers[0].delta_mass.unwrap() - (-18.418)).abs() < 0.001);
        assert!((opm.maneuvers[0].duration - 132.60).abs() < 0.01);
    }

    #[test]
    fn test_opm_kvn_round_trip() {
        let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample1.txt").unwrap();
        let kvn_str = opm.to_string(CCSDSFormat::KVN).unwrap();
        let opm2 = OPM::from_str(&kvn_str).unwrap();
        assert_eq!(opm2.metadata.object_name, opm.metadata.object_name);
        assert_eq!(opm2.metadata.object_id, opm.metadata.object_id);
        // Position round-trip: m → km → m
        assert!((opm2.state_vector.position[0] - opm.state_vector.position[0]).abs() < 1.0);
        assert!((opm2.state_vector.velocity[0] - opm.state_vector.velocity[0]).abs() < 0.001);
        // Spacecraft parameters
        assert!(opm2.spacecraft_parameters.is_some());
        let sp2 = opm2.spacecraft_parameters.as_ref().unwrap();
        assert!((sp2.mass.unwrap() - 3000.0).abs() < 0.01);
    }

    #[test]
    fn test_opm_kvn_round_trip_with_keplerian_and_maneuvers() {
        let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample5.txt").unwrap();
        let kvn_str = opm.to_string(CCSDSFormat::KVN).unwrap();
        let opm2 = OPM::from_str(&kvn_str).unwrap();
        // Keplerian elements
        assert!(opm2.keplerian_elements.is_some());
        let ke1 = opm.keplerian_elements.as_ref().unwrap();
        let ke2 = opm2.keplerian_elements.as_ref().unwrap();
        assert!((ke2.eccentricity - ke1.eccentricity).abs() < 1e-9);
        assert!((ke2.semi_major_axis - ke1.semi_major_axis).abs() < 1.0);
        // Maneuvers
        assert_eq!(opm2.maneuvers.len(), opm.maneuvers.len());
        assert!((opm2.maneuvers[0].duration - opm.maneuvers[0].duration).abs() < 0.01);
        assert!((opm2.maneuvers[0].dv[0] - opm.maneuvers[0].dv[0]).abs() < 0.01);
    }

    #[test]
    fn test_opm_xml_round_trip() {
        let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample3.xml").unwrap();
        let xml_str = opm.to_string(CCSDSFormat::XML).unwrap();
        let opm2 = OPM::from_str(&xml_str).unwrap();
        assert_eq!(opm2.metadata.object_name, opm.metadata.object_name);
        assert_eq!(opm2.metadata.object_id, opm.metadata.object_id);
        assert!((opm2.state_vector.position[0] - opm.state_vector.position[0]).abs() < 1.0);
        assert!((opm2.state_vector.velocity[0] - opm.state_vector.velocity[0]).abs() < 0.001);
        // Covariance
        assert!(opm2.covariance.is_some());
        let cov1 = opm.covariance.as_ref().unwrap();
        let cov2 = opm2.covariance.as_ref().unwrap();
        assert!((cov2.matrix[(0, 0)] - cov1.matrix[(0, 0)]).abs() < 1.0);
    }

    #[test]
    fn test_opm_xml_parse_example3() {
        let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample3.xml").unwrap();
        assert_eq!(opm.metadata.object_name, "OSPREY 5");
        assert_eq!(opm.metadata.object_id, "1998-999A");
        assert_eq!(opm.metadata.center_name, "EARTH");
        assert!(matches!(opm.metadata.ref_frame, CCSDSRefFrame::TOD));
        assert!(opm.metadata.ref_frame_epoch.is_some());
        // Position: 6503.514 km → m
        assert!((opm.state_vector.position[0] - 6503514.0).abs() < 1.0);
        assert!((opm.state_vector.velocity[0] - (-873.16)).abs() < 0.01);
        // Spacecraft parameters
        assert!(opm.spacecraft_parameters.is_some());
        let sp = opm.spacecraft_parameters.as_ref().unwrap();
        assert!((sp.mass.unwrap() - 3000.0).abs() < 0.01);
        // Covariance
        assert!(opm.covariance.is_some());
        let cov = opm.covariance.as_ref().unwrap();
        assert_eq!(cov.cov_ref_frame.as_ref().unwrap(), &CCSDSRefFrame::ITRF97);
        // CX_X = 0.316 km² = 316000 m²
        assert!((cov.matrix[(0, 0)] - 316000.0).abs() < 1.0);
    }

    #[test]
    fn test_opm_to_file_kvn() {
        let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample1.txt").unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join("brahe_test_opm.txt");
        opm.to_file(&path, CCSDSFormat::KVN).unwrap();
        let opm2 = OPM::from_file(&path).unwrap();
        assert_eq!(opm2.metadata.object_name, opm.metadata.object_name);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_opm_maneuver_new() {
        let epoch = Epoch::from_datetime(2024, 6, 1, 9, 0, 0.0, 0.0, TimeSystem::UTC);
        let m = OPMManeuver::new(epoch, 132.6, CCSDSRefFrame::EME2000, [10.0, -5.0, 2.0]);
        assert!((m.duration - 132.6).abs() < 1e-15);
        assert!(matches!(m.ref_frame, CCSDSRefFrame::EME2000));
        assert!((m.dv[0] - 10.0).abs() < 1e-15);
        assert!((m.dv[1] - (-5.0)).abs() < 1e-15);
        assert!((m.dv[2] - 2.0).abs() < 1e-15);
        assert!(m.delta_mass.is_none());
        assert!(m.comments.is_empty());
    }

    #[test]
    fn test_opm_maneuver_with_delta_mass() {
        let epoch = Epoch::from_datetime(2024, 6, 1, 9, 0, 0.0, 0.0, TimeSystem::UTC);
        let m = OPMManeuver::new(epoch, 100.0, CCSDSRefFrame::RTN, [1.0, 0.0, 0.0])
            .with_delta_mass(-10.5);
        assert_eq!(m.delta_mass, Some(-10.5));
    }

    #[test]
    fn test_opm_push_maneuver() {
        let meta = OPMMetadata::new(
            "SAT".to_string(),
            "2024-001A".to_string(),
            "EARTH".to_string(),
            CCSDSRefFrame::GCRF,
            CCSDSTimeSystem::UTC,
        );
        let sv = OPMStateVector::new(Epoch::now(), [7000e3, 0.0, 0.0], [0.0, 7500.0, 0.0]);
        let mut opm = OPM::new("ORG".to_string(), meta, sv);
        assert_eq!(opm.maneuvers.len(), 0);

        let epoch = Epoch::from_datetime(2024, 6, 1, 9, 0, 0.0, 0.0, TimeSystem::UTC);
        opm.push_maneuver(OPMManeuver::new(
            epoch,
            60.0,
            CCSDSRefFrame::RTN,
            [1.0, 0.0, 0.0],
        ));
        opm.push_maneuver(OPMManeuver::new(
            epoch,
            30.0,
            CCSDSRefFrame::RTN,
            [0.0, 1.0, 0.0],
        ));
        assert_eq!(opm.maneuvers.len(), 2);
        assert!((opm.maneuvers[0].duration - 60.0).abs() < 1e-15);
        assert!((opm.maneuvers[1].duration - 30.0).abs() < 1e-15);
    }

    #[test]
    fn test_opm_kvn_parse_spacecraft_params() {
        // OPMExample1 has spacecraft parameters
        let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample1.txt").unwrap();
        assert!(opm.spacecraft_parameters.is_some());
        let sp = opm.spacecraft_parameters.as_ref().unwrap();
        assert!(sp.mass.is_some());
        assert!((sp.mass.unwrap() - 3000.0).abs() < 0.01);
    }

    #[test]
    fn test_opm_to_json_string_upper_key_case() {
        let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample1.txt").unwrap();
        let json_str = opm.to_json_string(CCSDSJsonKeyCase::Upper).unwrap();
        assert!(json_str.contains("OBJECT_NAME"));
        assert!(json_str.contains("OBJECT_ID"));
    }
}
