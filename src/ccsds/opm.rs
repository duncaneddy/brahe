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

impl OPM {
    /// Parse an OPM message from a string, auto-detecting the format.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(content: &str) -> Result<Self, BraheError> {
        let format = crate::ccsds::detect_format(content);
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
            CCSDSFormat::JSON => crate::ccsds::json::write_opm_json(self),
        }
    }

    /// Write the OPM message to a file in the specified format.
    pub fn to_file<P: AsRef<Path>>(&self, path: P, format: CCSDSFormat) -> Result<(), BraheError> {
        let content = self.to_string(format)?;
        std::fs::write(path.as_ref(), content)
            .map_err(|e| BraheError::IoError(format!("Failed to write OPM file: {}", e)))
    }
}
