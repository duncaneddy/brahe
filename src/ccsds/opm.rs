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
            CCSDSFormat::JSON => Err(BraheError::Error(
                "OPM JSON format is not yet supported. Use KVN or XML format instead.".to_string(),
            )),
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
            CCSDSFormat::JSON => Err(BraheError::Error(
                "OPM JSON format is not yet supported. Use KVN or XML format instead.".to_string(),
            )),
        }
    }

    /// Write the OPM message to a file in the specified format.
    pub fn to_file<P: AsRef<Path>>(&self, path: P, format: CCSDSFormat) -> Result<(), BraheError> {
        let content = self.to_string(format)?;
        std::fs::write(path.as_ref(), content)
            .map_err(|e| BraheError::IoError(format!("Failed to write OPM file: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_opm_from_str_json_unsupported() {
        let result = OPM::from_str(r#"{"CCSDS_OPM_VERS": "3.0"}"#);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("JSON"),
            "Error should mention JSON: {}",
            err_msg
        );
    }

    #[test]
    fn test_opm_to_string_json_unsupported() {
        let metadata = OPMMetadata::new(
            "SAT".to_string(),
            "2024-001A".to_string(),
            "EARTH".to_string(),
            CCSDSRefFrame::GCRF,
            CCSDSTimeSystem::UTC,
        );
        let sv = OPMStateVector::new(Epoch::now(), [7000e3, 0.0, 0.0], [0.0, 7500.0, 0.0]);
        let opm = OPM::new("TEST".to_string(), metadata, sv);
        let result = opm.to_string(CCSDSFormat::JSON);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("JSON"),
            "Error should mention JSON: {}",
            err_msg
        );
    }

    #[test]
    fn test_opm_from_file_nonexistent() {
        let result = OPM::from_file("nonexistent_file.txt");
        assert!(result.is_err());
    }
}
