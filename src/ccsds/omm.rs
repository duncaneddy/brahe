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
        let format = crate::ccsds::detect_format(content);
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
            CCSDSFormat::JSON => crate::ccsds::json::write_omm_json(self),
        }
    }

    /// Write the OMM message to a file in the specified format.
    pub fn to_file<P: AsRef<Path>>(&self, path: P, format: CCSDSFormat) -> Result<(), BraheError> {
        let content = self.to_string(format)?;
        std::fs::write(path.as_ref(), content)
            .map_err(|e| BraheError::IoError(format!("Failed to write OMM file: {}", e)))
    }
}
