/*!
 * CCSDS Conjunction Data Message (CDM) data structures.
 *
 * CDM messages describe a conjunction between two space objects, containing
 * state vectors, covariance matrices, and collision probability data at
 * the Time of Closest Approach (TCA).
 *
 * Reference: CCSDS 508.0-P-1.1 (Conjunction Data Message), March 2024
 */

use std::path::Path;

use nalgebra::{SMatrix, SVector};

use crate::ccsds::common::{CCSDSFormat, CCSDSRefFrame, CCSDSUserDefined, CDMCovarianceDimension};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

/// CDM message header.
///
/// Standalone type (not reusing `ODMHeader`) because CDM has `message_for`
/// (absent in ODM) and `message_id` is mandatory (optional in ODM). The
/// version keyword is `CCSDS_CDM_VERS`, not `CCSDS_OPM_VERS`.
#[derive(Debug, Clone)]
pub struct CDMHeader {
    /// CCSDS_CDM_VERS (M) — format version, e.g. 1.0 or 2.0
    pub format_version: f64,
    /// CLASSIFICATION (O) — user-defined classification/caveats
    pub classification: Option<String>,
    /// CREATION_DATE (M) — always UTC
    pub creation_date: Epoch,
    /// ORIGINATOR (M) — SANA-registered organisation abbreviation
    pub originator: String,
    /// MESSAGE_FOR (O) — spacecraft name(s) CDM applies to
    pub message_for: Option<String>,
    /// MESSAGE_ID (M) — unique identifier per originator
    pub message_id: String,
    /// Comments
    pub comments: Vec<String>,
}

// ---------------------------------------------------------------------------
// Relative Metadata
// ---------------------------------------------------------------------------

/// Conjunction-level metadata shared between both objects.
///
/// Contains TCA, miss distance, relative state in RTN, screening volume
/// parameters, collision probability, and message tracking fields.
#[derive(Debug, Clone)]
pub struct CDMRelativeMetadata {
    /// CONJUNCTION_ID (O) v2.0
    pub conjunction_id: Option<String>,
    /// TCA (M) — Time of Closest Approach, always UTC
    pub tca: Epoch,
    /// MISS_DISTANCE (M) — Units: m
    pub miss_distance: f64,
    /// MAHALANOBIS_DISTANCE (O)
    pub mahalanobis_distance: Option<f64>,
    /// RELATIVE_SPEED (O) — Units: m/s
    pub relative_speed: Option<f64>,
    /// RELATIVE_POSITION_R (O) — Units: m
    pub relative_position_r: Option<f64>,
    /// RELATIVE_POSITION_T (O) — Units: m
    pub relative_position_t: Option<f64>,
    /// RELATIVE_POSITION_N (O) — Units: m
    pub relative_position_n: Option<f64>,
    /// RELATIVE_VELOCITY_R (O) — Units: m/s
    pub relative_velocity_r: Option<f64>,
    /// RELATIVE_VELOCITY_T (O) — Units: m/s
    pub relative_velocity_t: Option<f64>,
    /// RELATIVE_VELOCITY_N (O) — Units: m/s
    pub relative_velocity_n: Option<f64>,
    /// APPROACH_ANGLE (O) — Units: degrees
    pub approach_angle: Option<f64>,

    // Screening volume
    /// START_SCREEN_PERIOD (O)
    pub start_screen_period: Option<Epoch>,
    /// STOP_SCREEN_PERIOD (O)
    pub stop_screen_period: Option<Epoch>,
    /// SCREEN_TYPE (O) — e.g. SHAPE, PC, PC_MAX
    pub screen_type: Option<String>,
    /// SCREEN_VOLUME_FRAME (C) — RTN or TVN
    pub screen_volume_frame: Option<CCSDSRefFrame>,
    /// SCREEN_VOLUME_SHAPE (C) — SPHERE, ELLIPSOID, BOX
    pub screen_volume_shape: Option<String>,
    /// SCREEN_VOLUME_RADIUS (C) — Units: m (for SPHERE)
    pub screen_volume_radius: Option<f64>,
    /// SCREEN_VOLUME_X (C) — Units: m
    pub screen_volume_x: Option<f64>,
    /// SCREEN_VOLUME_Y (C) — Units: m
    pub screen_volume_y: Option<f64>,
    /// SCREEN_VOLUME_Z (C) — Units: m
    pub screen_volume_z: Option<f64>,
    /// SCREEN_ENTRY_TIME (C)
    pub screen_entry_time: Option<Epoch>,
    /// SCREEN_EXIT_TIME (C)
    pub screen_exit_time: Option<Epoch>,
    /// SCREEN_PC_THRESHOLD (C) v2.0
    pub screen_pc_threshold: Option<f64>,

    // Collision probability
    /// COLLISION_PERCENTILE (O) v2.0
    pub collision_percentile: Option<Vec<u32>>,
    /// COLLISION_PROBABILITY (O)
    pub collision_probability: Option<f64>,
    /// COLLISION_PROBABILITY_METHOD (O)
    pub collision_probability_method: Option<String>,
    /// COLLISION_MAX_PROBABILITY (O) v2.0
    pub collision_max_probability: Option<f64>,
    /// COLLISION_MAX_PC_METHOD (O) v2.0
    pub collision_max_pc_method: Option<String>,

    // SEFI
    /// SEFI_COLLISION_PROBABILITY (O) v2.0
    pub sefi_collision_probability: Option<f64>,
    /// SEFI_COLLISION_PROBABILITY_METHOD (O) v2.0
    pub sefi_collision_probability_method: Option<String>,
    /// SEFI_FRAGMENTATION_MODEL (O) v2.0
    pub sefi_fragmentation_model: Option<String>,

    // Message tracking
    /// PREVIOUS_MESSAGE_ID (O) v2.0
    pub previous_message_id: Option<String>,
    /// PREVIOUS_MESSAGE_EPOCH (O) v2.0
    pub previous_message_epoch: Option<Epoch>,
    /// NEXT_MESSAGE_EPOCH (O) v2.0
    pub next_message_epoch: Option<Epoch>,

    /// Comments
    pub comments: Vec<String>,
}

// ---------------------------------------------------------------------------
// Object Metadata
// ---------------------------------------------------------------------------

/// Metadata for one object in the conjunction.
#[derive(Debug, Clone)]
pub struct CDMObjectMetadata {
    /// OBJECT (M) — "OBJECT1" or "OBJECT2"
    pub object: String,
    /// OBJECT_DESIGNATOR (M) — catalog ID
    pub object_designator: String,
    /// CATALOG_NAME (M)
    pub catalog_name: String,
    /// OBJECT_NAME (M)
    pub object_name: String,
    /// INTERNATIONAL_DESIGNATOR (M) — COSPAR format
    pub international_designator: String,
    /// OBJECT_TYPE (O) — PAYLOAD, DEBRIS, ROCKET BODY, UNKNOWN, etc.
    pub object_type: Option<String>,
    /// OPS_STATUS (O) v2.0
    pub ops_status: Option<String>,
    /// OPERATOR_CONTACT_POSITION (O)
    pub operator_contact_position: Option<String>,
    /// OPERATOR_ORGANIZATION (O)
    pub operator_organization: Option<String>,
    /// OPERATOR_PHONE (O)
    pub operator_phone: Option<String>,
    /// OPERATOR_EMAIL (O)
    pub operator_email: Option<String>,
    /// EPHEMERIS_NAME (M)
    pub ephemeris_name: String,
    /// ODM_MSG_LINK (C) v2.0 — mandatory if EPHEMERIS_NAME=ODM
    pub odm_msg_link: Option<String>,
    /// ADM_MSG_LINK (O) v2.0
    pub adm_msg_link: Option<String>,
    /// OBS_BEFORE_NEXT_MESSAGE (O) v2.0
    pub obs_before_next_message: Option<String>,
    /// COVARIANCE_METHOD (M) — CALCULATED or DEFAULT
    pub covariance_method: String,
    /// COVARIANCE_SOURCE (O) v2.0
    pub covariance_source: Option<String>,
    /// MANEUVERABLE (M) — YES, NO, N/A, UNKNOWN
    pub maneuverable: String,
    /// ORBIT_CENTER (O) — defaults to EARTH
    pub orbit_center: Option<String>,
    /// REF_FRAME (M)
    pub ref_frame: CCSDSRefFrame,
    /// ALT_COV_TYPE (O) — XYZ
    pub alt_cov_type: Option<String>,
    /// ALT_COV_REF_FRAME (C) — mandatory if ALT_COV_TYPE present
    pub alt_cov_ref_frame: Option<CCSDSRefFrame>,
    /// GRAVITY_MODEL (O)
    pub gravity_model: Option<String>,
    /// ATMOSPHERIC_MODEL (O)
    pub atmospheric_model: Option<String>,
    /// N_BODY_PERTURBATIONS (O)
    pub n_body_perturbations: Option<String>,
    /// SOLAR_RAD_PRESSURE (O) — YES/NO
    pub solar_rad_pressure: Option<String>,
    /// EARTH_TIDES (O) — YES/NO
    pub earth_tides: Option<String>,
    /// INTRACK_THRUST (O) — YES/NO
    pub intrack_thrust: Option<String>,
    /// Comments
    pub comments: Vec<String>,
}

// ---------------------------------------------------------------------------
// OD Parameters
// ---------------------------------------------------------------------------

/// Orbit determination parameters for one object.
#[derive(Debug, Clone)]
pub struct CDMODParameters {
    /// TIME_LASTOB_START (O)
    pub time_lastob_start: Option<Epoch>,
    /// TIME_LASTOB_END (O)
    pub time_lastob_end: Option<Epoch>,
    /// RECOMMENDED_OD_SPAN (O) — Units: days
    pub recommended_od_span: Option<f64>,
    /// ACTUAL_OD_SPAN (O) — Units: days
    pub actual_od_span: Option<f64>,
    /// OBS_AVAILABLE (O)
    pub obs_available: Option<u32>,
    /// OBS_USED (O)
    pub obs_used: Option<u32>,
    /// TRACKS_AVAILABLE (O)
    pub tracks_available: Option<u32>,
    /// TRACKS_USED (O)
    pub tracks_used: Option<u32>,
    /// RESIDUALS_ACCEPTED (O) — Units: percent (0–100)
    pub residuals_accepted: Option<f64>,
    /// WEIGHTED_RMS (O)
    pub weighted_rms: Option<f64>,
    /// OD_EPOCH (O) v2.0
    pub od_epoch: Option<Epoch>,
    /// Comments
    pub comments: Vec<String>,
}

// ---------------------------------------------------------------------------
// Additional Parameters
// ---------------------------------------------------------------------------

/// Physical and operational parameters for one object.
#[derive(Debug, Clone)]
pub struct CDMAdditionalParameters {
    /// AREA_PC (O) — Units: m²
    pub area_pc: Option<f64>,
    /// AREA_PC_MIN (O) v2.0 — Units: m²
    pub area_pc_min: Option<f64>,
    /// AREA_PC_MAX (O) v2.0 — Units: m²
    pub area_pc_max: Option<f64>,
    /// AREA_DRG (O) — Units: m²
    pub area_drg: Option<f64>,
    /// AREA_SRP (O) — Units: m²
    pub area_srp: Option<f64>,

    // OEB (Optimally Enclosing Box) v2.0
    /// OEB_PARENT_FRAME (C)
    pub oeb_parent_frame: Option<String>,
    /// OEB_PARENT_FRAME_EPOCH (C)
    pub oeb_parent_frame_epoch: Option<Epoch>,
    /// OEB_Q1 (O)
    pub oeb_q1: Option<f64>,
    /// OEB_Q2 (O)
    pub oeb_q2: Option<f64>,
    /// OEB_Q3 (O)
    pub oeb_q3: Option<f64>,
    /// OEB_QC (O) — quaternion scalar
    pub oeb_qc: Option<f64>,
    /// OEB_MAX (O) — Units: m
    pub oeb_max: Option<f64>,
    /// OEB_INT (O) — Units: m
    pub oeb_int: Option<f64>,
    /// OEB_MIN (O) — Units: m
    pub oeb_min: Option<f64>,
    /// AREA_ALONG_OEB_MAX (O) — Units: m²
    pub area_along_oeb_max: Option<f64>,
    /// AREA_ALONG_OEB_INT (O) — Units: m²
    pub area_along_oeb_int: Option<f64>,
    /// AREA_ALONG_OEB_MIN (O) — Units: m²
    pub area_along_oeb_min: Option<f64>,

    // RCS / Visual magnitude v2.0
    /// RCS (O) — Units: m²
    pub rcs: Option<f64>,
    /// RCS_MIN (O) — Units: m²
    pub rcs_min: Option<f64>,
    /// RCS_MAX (O) — Units: m²
    pub rcs_max: Option<f64>,
    /// VM_ABSOLUTE (O)
    pub vm_absolute: Option<f64>,
    /// VM_APPARENT_MIN (O)
    pub vm_apparent_min: Option<f64>,
    /// VM_APPARENT (O)
    pub vm_apparent: Option<f64>,
    /// VM_APPARENT_MAX (O)
    pub vm_apparent_max: Option<f64>,
    /// REFLECTANCE (O) — 0 to 1
    pub reflectance: Option<f64>,

    /// MASS (O) — Units: kg
    pub mass: Option<f64>,
    /// HBR (O) — hard-body radius. Units: m
    pub hbr: Option<f64>,
    /// CD_AREA_OVER_MASS (O) — Units: m²/kg
    pub cd_area_over_mass: Option<f64>,
    /// CR_AREA_OVER_MASS (O) — Units: m²/kg
    pub cr_area_over_mass: Option<f64>,
    /// THRUST_ACCELERATION (O) — Units: m/s²
    pub thrust_acceleration: Option<f64>,
    /// SEDR (O) — Units: W/kg
    pub sedr: Option<f64>,

    // Delta-V v2.0
    /// MIN_DV (O) — Units: m/s, 3 elements [R, T, N]
    pub min_dv: Option<[f64; 3]>,
    /// MAX_DV (O) — Units: m/s, 3 elements [R, T, N]
    pub max_dv: Option<[f64; 3]>,

    /// LEAD_TIME_REQD_BEFORE_TCA (O) — Units: hours (stored as-is per spec)
    pub lead_time_reqd_before_tca: Option<f64>,

    // Orbital descriptors v2.0
    /// APOAPSIS_ALTITUDE (O) — Units: m (converted from km)
    pub apoapsis_altitude: Option<f64>,
    /// PERIAPSIS_ALTITUDE (O) — Units: m (converted from km)
    pub periapsis_altitude: Option<f64>,
    /// INCLINATION (O) — Units: degrees
    pub inclination: Option<f64>,

    // Covariance confidence v2.0
    /// COV_CONFIDENCE (O)
    pub cov_confidence: Option<f64>,
    /// COV_CONFIDENCE_METHOD (C)
    pub cov_confidence_method: Option<String>,

    /// Comments
    pub comments: Vec<String>,
}

// ---------------------------------------------------------------------------
// State Vector
// ---------------------------------------------------------------------------

/// State vector for one object at TCA.
///
/// Position and velocity stored in SI units (meters, m/s).
/// The epoch is implicitly the TCA from `CDMRelativeMetadata`.
#[derive(Debug, Clone)]
pub struct CDMStateVector {
    /// Position [x, y, z]. Units: meters (converted from km in CCSDS files)
    pub position: [f64; 3],
    /// Velocity [vx, vy, vz]. Units: m/s (converted from km/s in CCSDS files)
    pub velocity: [f64; 3],
    /// Comments
    pub comments: Vec<String>,
}

// ---------------------------------------------------------------------------
// Covariance types
// ---------------------------------------------------------------------------

/// RTN covariance matrix for one CDM object.
///
/// Stores a symmetric matrix of up to 9×9 (6×6 position/velocity core plus
/// optional drag, SRP, and thrust rows). Values are in CCSDS native units
/// which are already SI for the core block:
/// - Position-position: m²
/// - Position-velocity: m²/s
/// - Velocity-velocity: m²/s²
/// - Drag/SRP rows: m³/kg, m³/(kg·s), m⁴/kg²
/// - Thrust row: m²/s², m²/s³, m³/(kg·s²), m²/s⁴
#[derive(Debug, Clone)]
pub struct CDMRTNCovariance {
    /// Full 9×9 symmetric covariance matrix. Unused rows/columns (beyond
    /// `dimension`) are zero.
    pub matrix: SMatrix<f64, 9, 9>,
    /// How many rows/columns are populated.
    pub dimension: CDMCovarianceDimension,
    /// Comments
    pub comments: Vec<String>,
}

/// XYZ covariance matrix (alternate representation).
///
/// Same structure as RTN covariance but with XYZ field naming.
/// Conditional on `ALT_COV_TYPE = XYZ` in the object metadata.
#[derive(Debug, Clone)]
pub struct CDMXYZCovariance {
    /// Full 9×9 symmetric covariance matrix in the XYZ frame specified
    /// by `ALT_COV_REF_FRAME`.
    pub matrix: SMatrix<f64, 9, 9>,
    /// How many rows/columns are populated.
    pub dimension: CDMCovarianceDimension,
    /// Comments
    pub comments: Vec<String>,
}

// ---------------------------------------------------------------------------
// Additional Covariance Metadata
// ---------------------------------------------------------------------------

/// Additional covariance metadata for one CDM object (v2.0).
#[derive(Debug, Clone)]
pub struct CDMAdditionalCovarianceMetadata {
    /// DENSITY_FORECAST_UNCERTAINTY (O)
    pub density_forecast_uncertainty: Option<f64>,
    /// CSCALE_FACTOR_MIN (O)
    pub cscale_factor_min: Option<f64>,
    /// CSCALE_FACTOR (O)
    pub cscale_factor: Option<f64>,
    /// CSCALE_FACTOR_MAX (O)
    pub cscale_factor_max: Option<f64>,
    /// SCREENING_DATA_SOURCE (O)
    pub screening_data_source: Option<String>,
    /// DCP_SENSITIVITY_VECTOR_POSITION (O) — 3 elements, Units: m
    pub dcp_sensitivity_vector_position: Option<[f64; 3]>,
    /// DCP_SENSITIVITY_VECTOR_VELOCITY (O) — 3 elements, Units: m/s
    pub dcp_sensitivity_vector_velocity: Option<[f64; 3]>,
    /// Comments
    pub comments: Vec<String>,
}

// ---------------------------------------------------------------------------
// Object Data (combines all data sections)
// ---------------------------------------------------------------------------

/// All data for one object in the conjunction.
#[derive(Debug, Clone)]
pub struct CDMObjectData {
    /// OD parameters (all optional fields)
    pub od_parameters: Option<CDMODParameters>,
    /// Additional physical/operational parameters
    pub additional_parameters: Option<CDMAdditionalParameters>,
    /// State vector at TCA
    pub state_vector: CDMStateVector,
    /// RTN covariance (mandatory 6×6 core, optional extended)
    pub rtn_covariance: CDMRTNCovariance,
    /// XYZ covariance (conditional on ALT_COV_TYPE=XYZ)
    pub xyz_covariance: Option<CDMXYZCovariance>,
    /// Additional covariance metadata (v2.0)
    pub additional_covariance_metadata: Option<CDMAdditionalCovarianceMetadata>,
    /// CSIG3EIGVEC3 raw string (stored but not parsed into matrix)
    pub csig3eigvec3: Option<String>,
    /// Comments
    pub comments: Vec<String>,
}

// ---------------------------------------------------------------------------
// CDM Object (metadata + data)
// ---------------------------------------------------------------------------

/// One object in the conjunction (metadata + data).
#[derive(Debug, Clone)]
pub struct CDMObject {
    /// Object metadata
    pub metadata: CDMObjectMetadata,
    /// Object data
    pub data: CDMObjectData,
}

// ---------------------------------------------------------------------------
// Top-level CDM
// ---------------------------------------------------------------------------

/// A complete CCSDS Conjunction Data Message.
///
/// Contains a header, conjunction-level relative metadata, and exactly two
/// object sections (OBJECT1 and OBJECT2), each with their own metadata,
/// state vector, and covariance matrix.
#[derive(Debug, Clone)]
pub struct CDM {
    /// CDM header
    pub header: CDMHeader,
    /// Relative metadata/data (shared between objects)
    pub relative_metadata: CDMRelativeMetadata,
    /// Object 1 (metadata + data)
    pub object1: CDMObject,
    /// Object 2 (metadata + data)
    pub object2: CDMObject,
    /// Optional user-defined parameters
    pub user_defined: Option<CCSDSUserDefined>,
}

// ---------------------------------------------------------------------------
// Constructors and helpers
// ---------------------------------------------------------------------------

impl CDMRelativeMetadata {
    /// Create relative metadata with only the mandatory fields.
    pub fn new(tca: Epoch, miss_distance: f64) -> Self {
        Self {
            conjunction_id: None,
            tca,
            miss_distance,
            mahalanobis_distance: None,
            relative_speed: None,
            relative_position_r: None,
            relative_position_t: None,
            relative_position_n: None,
            relative_velocity_r: None,
            relative_velocity_t: None,
            relative_velocity_n: None,
            approach_angle: None,
            start_screen_period: None,
            stop_screen_period: None,
            screen_type: None,
            screen_volume_frame: None,
            screen_volume_shape: None,
            screen_volume_radius: None,
            screen_volume_x: None,
            screen_volume_y: None,
            screen_volume_z: None,
            screen_entry_time: None,
            screen_exit_time: None,
            screen_pc_threshold: None,
            collision_percentile: None,
            collision_probability: None,
            collision_probability_method: None,
            collision_max_probability: None,
            collision_max_pc_method: None,
            sefi_collision_probability: None,
            sefi_collision_probability_method: None,
            sefi_fragmentation_model: None,
            previous_message_id: None,
            previous_message_epoch: None,
            next_message_epoch: None,
            comments: Vec::new(),
        }
    }
}

impl CDMObjectMetadata {
    /// Create object metadata with only the mandatory fields.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        object: String,
        object_designator: String,
        catalog_name: String,
        object_name: String,
        international_designator: String,
        ephemeris_name: String,
        covariance_method: String,
        maneuverable: String,
        ref_frame: CCSDSRefFrame,
    ) -> Self {
        Self {
            object,
            object_designator,
            catalog_name,
            object_name,
            international_designator,
            object_type: None,
            ops_status: None,
            operator_contact_position: None,
            operator_organization: None,
            operator_phone: None,
            operator_email: None,
            ephemeris_name,
            odm_msg_link: None,
            adm_msg_link: None,
            obs_before_next_message: None,
            covariance_method,
            covariance_source: None,
            maneuverable,
            orbit_center: None,
            ref_frame,
            alt_cov_type: None,
            alt_cov_ref_frame: None,
            gravity_model: None,
            atmospheric_model: None,
            n_body_perturbations: None,
            solar_rad_pressure: None,
            earth_tides: None,
            intrack_thrust: None,
            comments: Vec::new(),
        }
    }
}

impl CDMStateVector {
    /// Create a new state vector.
    ///
    /// # Arguments
    ///
    /// * `position` - Position [x, y, z]. Units: meters
    /// * `velocity` - Velocity [vx, vy, vz]. Units: m/s
    pub fn new(position: [f64; 3], velocity: [f64; 3]) -> Self {
        Self {
            position,
            velocity,
            comments: Vec::new(),
        }
    }
}

impl CDMRTNCovariance {
    /// Create a 6×6 RTN covariance from a nalgebra 6×6 matrix.
    pub fn from_6x6(matrix: SMatrix<f64, 6, 6>) -> Self {
        let mut full = SMatrix::<f64, 9, 9>::zeros();
        for i in 0..6 {
            for j in 0..6 {
                full[(i, j)] = matrix[(i, j)];
            }
        }
        Self {
            matrix: full,
            dimension: CDMCovarianceDimension::SixBySix,
            comments: Vec::new(),
        }
    }

    /// Extract the 6×6 position/velocity submatrix.
    pub fn to_6x6(&self) -> SMatrix<f64, 6, 6> {
        self.matrix.fixed_view::<6, 6>(0, 0).into()
    }
}

impl CDMObject {
    /// Create a new CDM object with mandatory fields.
    pub fn new(
        metadata: CDMObjectMetadata,
        state_vector: CDMStateVector,
        rtn_covariance: CDMRTNCovariance,
    ) -> Self {
        Self {
            metadata,
            data: CDMObjectData {
                od_parameters: None,
                additional_parameters: None,
                state_vector,
                rtn_covariance,
                xyz_covariance: None,
                additional_covariance_metadata: None,
                csig3eigvec3: None,
                comments: Vec::new(),
            },
        }
    }
}

impl CDM {
    /// Create a new CDM message with required fields.
    ///
    /// # Arguments
    ///
    /// * `originator` - Originator of the message
    /// * `message_id` - Unique message identifier
    /// * `tca` - Time of closest approach
    /// * `miss_distance` - Miss distance in meters
    /// * `object1` - First object data
    /// * `object2` - Second object data
    pub fn new(
        originator: String,
        message_id: String,
        tca: Epoch,
        miss_distance: f64,
        object1: CDMObject,
        object2: CDMObject,
    ) -> Self {
        Self {
            header: CDMHeader {
                format_version: 1.0,
                classification: None,
                creation_date: Epoch::now(),
                originator,
                message_for: None,
                message_id,
                comments: Vec::new(),
            },
            relative_metadata: CDMRelativeMetadata::new(tca, miss_distance),
            object1,
            object2,
            user_defined: None,
        }
    }

    /// Parse a CDM message from a string, auto-detecting the format.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(content: &str) -> Result<Self, BraheError> {
        let format = crate::ccsds::common::detect_format(content);
        match format {
            CCSDSFormat::KVN => crate::ccsds::kvn::parse_cdm(content),
            CCSDSFormat::XML => crate::ccsds::xml::parse_cdm_xml(content),
            CCSDSFormat::JSON => crate::ccsds::json::parse_cdm_json(content),
        }
    }

    /// Parse a CDM message from a file, auto-detecting the format.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, BraheError> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| BraheError::IoError(format!("Failed to read CDM file: {}", e)))?;
        Self::from_str(&content)
    }

    /// Write the CDM message to a string in the specified format.
    pub fn to_string(&self, format: CCSDSFormat) -> Result<String, BraheError> {
        match format {
            CCSDSFormat::KVN => crate::ccsds::kvn::write_cdm(self),
            CCSDSFormat::XML => crate::ccsds::xml::write_cdm_xml(self),
            CCSDSFormat::JSON => crate::ccsds::json::write_cdm_json(self),
        }
    }

    /// Write the CDM message to a file in the specified format.
    pub fn to_file<P: AsRef<Path>>(&self, path: P, format: CCSDSFormat) -> Result<(), BraheError> {
        let content = self.to_string(format)?;
        std::fs::write(path.as_ref(), content)
            .map_err(|e| BraheError::IoError(format!("Failed to write CDM file: {}", e)))
    }

    // -----------------------------------------------------------------------
    // Convenience accessors
    // -----------------------------------------------------------------------

    /// Get the TCA epoch.
    pub fn tca(&self) -> &Epoch {
        &self.relative_metadata.tca
    }

    /// Get the miss distance in meters.
    pub fn miss_distance(&self) -> f64 {
        self.relative_metadata.miss_distance
    }

    /// Get collision probability if present.
    pub fn collision_probability(&self) -> Option<f64> {
        self.relative_metadata.collision_probability
    }

    /// Get object1 state vector as a 6-element vector [x, y, z, vx, vy, vz] in m and m/s.
    pub fn object1_state(&self) -> SVector<f64, 6> {
        let sv = &self.object1.data.state_vector;
        SVector::<f64, 6>::new(
            sv.position[0],
            sv.position[1],
            sv.position[2],
            sv.velocity[0],
            sv.velocity[1],
            sv.velocity[2],
        )
    }

    /// Get object2 state vector as a 6-element vector [x, y, z, vx, vy, vz] in m and m/s.
    pub fn object2_state(&self) -> SVector<f64, 6> {
        let sv = &self.object2.data.state_vector;
        SVector::<f64, 6>::new(
            sv.position[0],
            sv.position[1],
            sv.position[2],
            sv.velocity[0],
            sv.velocity[1],
            sv.velocity[2],
        )
    }

    /// Get the relative position vector [R, T, N] in meters, if all components present.
    pub fn relative_position_rtn(&self) -> Option<[f64; 3]> {
        let rm = &self.relative_metadata;
        match (
            rm.relative_position_r,
            rm.relative_position_t,
            rm.relative_position_n,
        ) {
            (Some(r), Some(t), Some(n)) => Some([r, t, n]),
            _ => None,
        }
    }

    /// Get the relative velocity vector [R, T, N] in m/s, if all components present.
    pub fn relative_velocity_rtn(&self) -> Option<[f64; 3]> {
        let rm = &self.relative_metadata;
        match (
            rm.relative_velocity_r,
            rm.relative_velocity_t,
            rm.relative_velocity_n,
        ) {
            (Some(r), Some(t), Some(n)) => Some([r, t, n]),
            _ => None,
        }
    }

    /// Get the 6×6 position/velocity submatrix of object1's RTN covariance.
    pub fn object1_rtn_covariance_6x6(&self) -> SMatrix<f64, 6, 6> {
        self.object1.data.rtn_covariance.to_6x6()
    }

    /// Get the 6×6 position/velocity submatrix of object2's RTN covariance.
    pub fn object2_rtn_covariance_6x6(&self) -> SMatrix<f64, 6, 6> {
        self.object2.data.rtn_covariance.to_6x6()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_cdm_new() {
        let sv1 = CDMStateVector::new([7000e3, 0.0, 0.0], [0.0, 7500.0, 0.0]);
        let sv2 = CDMStateVector::new([7001e3, 0.0, 0.0], [0.0, -7500.0, 0.0]);
        let cov1 = CDMRTNCovariance::from_6x6(SMatrix::<f64, 6, 6>::identity());
        let cov2 = CDMRTNCovariance::from_6x6(SMatrix::<f64, 6, 6>::identity());

        let meta1 = CDMObjectMetadata::new(
            "OBJECT1".to_string(),
            "12345".to_string(),
            "SATCAT".to_string(),
            "SAT A".to_string(),
            "2020-001A".to_string(),
            "NONE".to_string(),
            "CALCULATED".to_string(),
            "YES".to_string(),
            CCSDSRefFrame::EME2000,
        );
        let meta2 = CDMObjectMetadata::new(
            "OBJECT2".to_string(),
            "67890".to_string(),
            "SATCAT".to_string(),
            "SAT B".to_string(),
            "2021-002B".to_string(),
            "NONE".to_string(),
            "CALCULATED".to_string(),
            "NO".to_string(),
            CCSDSRefFrame::EME2000,
        );

        let obj1 = CDMObject::new(meta1, sv1, cov1);
        let obj2 = CDMObject::new(meta2, sv2, cov2);

        let tca = Epoch::from_datetime(2024, 1, 15, 12, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let cdm = CDM::new(
            "TEST_ORG".to_string(),
            "MSG001".to_string(),
            tca,
            715.0,
            obj1,
            obj2,
        );

        assert_eq!(cdm.header.originator, "TEST_ORG");
        assert_eq!(cdm.header.message_id, "MSG001");
        assert_eq!(cdm.miss_distance(), 715.0);
        assert_eq!(cdm.object1.metadata.object_name, "SAT A");
        assert_eq!(cdm.object2.metadata.object_name, "SAT B");

        let s1 = cdm.object1_state();
        assert_eq!(s1[0], 7000e3);
        assert_eq!(s1[4], 7500.0);

        assert!(cdm.collision_probability().is_none());
        assert!(cdm.relative_position_rtn().is_none());
    }

    #[test]
    fn test_cdm_kvn_parse_example1() {
        let cdm = CDM::from_file("test_assets/ccsds/cdm/CDMExample1.txt").unwrap();

        // Header
        assert_eq!(cdm.header.format_version, 1.0);
        assert_eq!(cdm.header.originator, "JSPOC");
        assert_eq!(cdm.header.message_id, "201113719185");
        assert!(cdm.header.message_for.is_none());

        // Relative metadata
        assert_eq!(cdm.miss_distance(), 715.0);
        assert!(cdm.collision_probability().is_none());
        assert!(cdm.relative_position_rtn().is_none());

        // Object 1
        assert_eq!(cdm.object1.metadata.object, "OBJECT1");
        assert_eq!(cdm.object1.metadata.object_designator, "12345");
        assert_eq!(cdm.object1.metadata.object_name, "SATELLITE A");
        assert_eq!(cdm.object1.metadata.ephemeris_name, "EPHEMERIS SATELLITE A");
        assert_eq!(cdm.object1.metadata.maneuverable, "YES");
        assert_eq!(cdm.object1.metadata.ref_frame, CCSDSRefFrame::EME2000);

        // Object 1 state vector (converted from km → m, km/s → m/s)
        let s1 = cdm.object1_state();
        assert!((s1[0] - 2570097.065).abs() < 0.01);
        assert!((s1[1] - 2244654.904).abs() < 0.01);
        assert!((s1[2] - 6281497.978).abs() < 0.01);
        assert!((s1[3] - 4418.769571).abs() < 0.0001);
        assert!((s1[4] - 4833.547743).abs() < 0.0001);
        assert!((s1[5] - (-3526.774282)).abs() < 0.0001);

        // Object 1 covariance (already in m², no conversion)
        let cov1 = cdm.object1_rtn_covariance_6x6();
        assert!((cov1[(0, 0)] - 4.142e+01).abs() < 1e-10);
        assert!((cov1[(1, 0)] - (-8.579e+00)).abs() < 1e-10);
        assert!((cov1[(1, 1)] - 2.533e+03).abs() < 1e-10);
        assert_eq!(
            cdm.object1.data.rtn_covariance.dimension,
            CDMCovarianceDimension::SixBySix
        );

        // Object 2
        assert_eq!(cdm.object2.metadata.object, "OBJECT2");
        assert_eq!(cdm.object2.metadata.object_designator, "30337");
        assert_eq!(cdm.object2.metadata.object_name, "FENGYUN 1C DEB");
        assert_eq!(cdm.object2.metadata.maneuverable, "NO");

        let s2 = cdm.object2_state();
        assert!((s2[0] - 2569540.800).abs() < 0.01);
        assert!((s2[3] - (-2888.6125)).abs() < 0.001);
    }

    #[test]
    fn test_cdm_kvn_parse_example2_extended_cov() {
        let cdm = CDM::from_file("test_assets/ccsds/cdm/CDMExample2.txt").unwrap();

        // Relative metadata
        assert!(cdm.header.message_for.is_some());
        assert_eq!(cdm.header.message_for.as_deref(), Some("SATELLITE A"));
        assert_eq!(cdm.relative_metadata.relative_speed, Some(14762.0));
        assert!((cdm.collision_probability().unwrap() - 4.835e-05).abs() < 1e-10);
        assert_eq!(
            cdm.relative_metadata
                .collision_probability_method
                .as_deref(),
            Some("FOSTER-1992")
        );

        // Relative position
        let rp = cdm.relative_position_rtn().unwrap();
        assert!((rp[0] - 27.4).abs() < 0.01);
        assert!((rp[1] - (-70.2)).abs() < 0.01);

        // Object 1 metadata
        assert_eq!(cdm.object1.metadata.object_type.as_deref(), Some("PAYLOAD"));
        assert_eq!(
            cdm.object1.metadata.operator_email.as_deref(),
            Some("JOHN.DOE@SOMEWHERE.NET")
        );
        assert_eq!(
            cdm.object1.metadata.gravity_model.as_deref(),
            Some("EGM-96: 36D 36O")
        );

        // OD parameters
        let od = cdm.object1.data.od_parameters.as_ref().unwrap();
        assert_eq!(od.obs_available, Some(592));
        assert_eq!(od.obs_used, Some(579));
        assert!((od.recommended_od_span.unwrap() - 7.88).abs() < 0.01);
        assert!((od.residuals_accepted.unwrap() - 97.8).abs() < 0.01);

        // Additional parameters
        let ap = cdm.object1.data.additional_parameters.as_ref().unwrap();
        assert!((ap.area_pc.unwrap() - 5.2).abs() < 0.01);
        assert!((ap.mass.unwrap() - 251.6).abs() < 0.01);
        assert!((ap.sedr.unwrap() - 4.54570e-05).abs() < 1e-10);

        // Extended covariance (8×8 with CDRG + CSRP)
        assert_eq!(
            cdm.object1.data.rtn_covariance.dimension,
            CDMCovarianceDimension::EightByEight
        );
        let cov = &cdm.object1.data.rtn_covariance.matrix;
        // CDRG_R = row 6, col 0
        assert!((cov[(6, 0)] - (-1.862e+00)).abs() < 1e-10);
        // CSRP_SRP = row 7, col 7
        assert!((cov[(7, 7)] - 1.593e-02).abs() < 1e-10);
    }

    #[test]
    fn test_cdm_kvn_parse_issue940_v2() {
        let cdm = CDM::from_file("test_assets/ccsds/cdm/CDMExample_issue_940.txt").unwrap();

        assert_eq!(cdm.header.format_version, 2.0);
        assert!(cdm.header.classification.is_some());
        assert_eq!(
            cdm.relative_metadata.conjunction_id.as_deref(),
            Some("20220708T10hz SATELLITEA SATELLITEB")
        );
        assert_eq!(cdm.relative_metadata.approach_angle, Some(180.0));
        assert_eq!(cdm.relative_metadata.screen_type.as_deref(), Some("SHAPE"));
        assert_eq!(cdm.relative_metadata.screen_pc_threshold, Some(1.0e-03));
        assert!(cdm.relative_metadata.collision_percentile.is_some());
        assert_eq!(
            cdm.relative_metadata.collision_percentile.as_ref().unwrap(),
            &[50, 51, 52]
        );
        assert!(cdm.relative_metadata.sefi_collision_probability.is_some());
        assert!(cdm.relative_metadata.previous_message_id.is_some());

        // Object 1 v2.0 fields
        assert_eq!(
            cdm.object1.metadata.odm_msg_link.as_deref(),
            Some("ODM_MSG_35132.txt")
        );
        assert_eq!(
            cdm.object1.metadata.covariance_source.as_deref(),
            Some("HAC Covariance")
        );
        assert_eq!(cdm.object1.metadata.alt_cov_type.as_deref(), Some("XYZ"));

        // OEB fields
        let ap = cdm.object1.data.additional_parameters.as_ref().unwrap();
        assert!((ap.oeb_q1.unwrap() - 0.03123).abs() < 1e-10);
        assert!((ap.hbr.unwrap() - 2.5).abs() < 0.01);
        assert!((ap.apoapsis_altitude.unwrap() - 800e3).abs() < 0.1); // km → m
        assert!((ap.inclination.unwrap() - 89.0).abs() < 0.01);

        // XYZ covariance
        assert!(cdm.object1.data.xyz_covariance.is_some());
        let xyz = cdm.object1.data.xyz_covariance.as_ref().unwrap();
        assert_eq!(xyz.dimension, CDMCovarianceDimension::NineByNine);
        assert!((xyz.matrix[(0, 0)] - 0.1).abs() < 1e-10);

        // Additional covariance metadata
        let acm = cdm
            .object1
            .data
            .additional_covariance_metadata
            .as_ref()
            .unwrap();
        assert!((acm.density_forecast_uncertainty.unwrap() - 2.5).abs() < 0.01);
        assert!((acm.cscale_factor.unwrap() - 1.0).abs() < 0.01);
        assert!(acm.dcp_sensitivity_vector_position.is_some());

        // Object 2 CSIG3EIGVEC3
        assert!(cdm.object2.data.csig3eigvec3.is_some());

        // User-defined parameters
        assert!(cdm.user_defined.is_some());
        let ud = cdm.user_defined.as_ref().unwrap();
        assert!(ud.parameters.contains_key("OBJ1_TIME_LASTOB_START"));
    }

    #[test]
    fn test_cdm_kvn_parse_issue942_maneuverable_na() {
        let cdm = CDM::from_file("test_assets/ccsds/cdm/CDMExample_issue942.txt").unwrap();
        assert_eq!(cdm.object1.metadata.maneuverable, "N/A");
    }

    #[test]
    fn test_cdm_kvn_parse_alfano01() {
        let cdm = CDM::from_file("test_assets/ccsds/cdm/AlfanoTestCase01.cdm").unwrap();
        assert!(cdm.miss_distance() > 0.0);
        // Verify state vectors and covariance parsed
        let s1 = cdm.object1_state();
        assert!(s1[0].abs() > 1.0); // Non-zero position
        // Alfano test cases have 8×8 covariance (6×6 + CDRG + CSRP)
        assert_eq!(
            cdm.object1.data.rtn_covariance.dimension,
            CDMCovarianceDimension::EightByEight
        );
    }

    #[test]
    fn test_cdm_kvn_parse_real_world() {
        let cdm = CDM::from_file("test_assets/ccsds/cdm/ION_SCV8_vs_STARLINK_1233.txt").unwrap();
        assert!(cdm.miss_distance() > 0.0);
        assert!(cdm.object1.data.od_parameters.is_some());
        assert!(cdm.object2.data.od_parameters.is_some());
    }

    #[test]
    fn test_cdm_kvn_missing_tca() {
        let result = CDM::from_file("test_assets/ccsds/cdm/CDM-missing-TCA.txt");
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("TCA"),
            "Error should mention TCA: {}",
            err_msg
        );
    }

    #[test]
    fn test_cdm_kvn_missing_obj2_state() {
        let result = CDM::from_file("test_assets/ccsds/cdm/CDM-missing-object2-state-vector.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_cdm_kvn_round_trip_example1() {
        let cdm1 = CDM::from_file("test_assets/ccsds/cdm/CDMExample1.txt").unwrap();
        let kvn = cdm1.to_string(CCSDSFormat::KVN).unwrap();
        let cdm2 = CDM::from_str(&kvn).unwrap();

        // Compare key fields
        assert_eq!(cdm1.header.format_version, cdm2.header.format_version);
        assert_eq!(cdm1.header.originator, cdm2.header.originator);
        assert_eq!(cdm1.header.message_id, cdm2.header.message_id);
        assert!((cdm1.miss_distance() - cdm2.miss_distance()).abs() < 1e-6);

        // State vectors
        for i in 0..6 {
            assert!(
                (cdm1.object1_state()[i] - cdm2.object1_state()[i]).abs() < 0.01,
                "Object1 state[{}] mismatch: {} vs {}",
                i,
                cdm1.object1_state()[i],
                cdm2.object1_state()[i]
            );
            assert!(
                (cdm1.object2_state()[i] - cdm2.object2_state()[i]).abs() < 0.01,
                "Object2 state[{}] mismatch: {} vs {}",
                i,
                cdm1.object2_state()[i],
                cdm2.object2_state()[i]
            );
        }

        // Covariance
        let c1 = cdm1.object1_rtn_covariance_6x6();
        let c2 = cdm2.object1_rtn_covariance_6x6();
        for i in 0..6 {
            for j in 0..6 {
                let rel = if c1[(i, j)].abs() > 1e-20 {
                    ((c1[(i, j)] - c2[(i, j)]) / c1[(i, j)]).abs()
                } else {
                    (c1[(i, j)] - c2[(i, j)]).abs()
                };
                assert!(
                    rel < 1e-4,
                    "Cov({},{}) mismatch: {} vs {}",
                    i,
                    j,
                    c1[(i, j)],
                    c2[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_cdm_xml_parse_example1() {
        let cdm = CDM::from_file("test_assets/ccsds/cdm/CDMExample1.xml").unwrap();

        assert_eq!(cdm.header.format_version, 1.0);
        assert_eq!(cdm.header.originator, "JSPOC");
        assert_eq!(cdm.header.message_for.as_deref(), Some("SATELLITE A"));
        assert_eq!(cdm.miss_distance(), 715.0);

        // Relative state vector
        let rp = cdm.relative_position_rtn().unwrap();
        assert!((rp[0] - 27.4).abs() < 0.01);

        // Object 1
        assert_eq!(cdm.object1.metadata.object_name, "SATELLITE A");
        assert_eq!(cdm.object1.metadata.ref_frame, CCSDSRefFrame::EME2000);

        let s1 = cdm.object1_state();
        assert!((s1[0] - 2570097.065).abs() < 0.01);

        // Covariance
        let cov1 = cdm.object1_rtn_covariance_6x6();
        assert!((cov1[(0, 0)] - 4.142e+01).abs() < 1e-10);

        // Object 2
        assert_eq!(cdm.object2.metadata.object_name, "FENGYUN 1C DEB");
    }

    #[test]
    fn test_cdm_xml_round_trip() {
        let cdm1 = CDM::from_file("test_assets/ccsds/cdm/CDMExample1.xml").unwrap();
        let xml = cdm1.to_string(CCSDSFormat::XML).unwrap();
        let cdm2 = CDM::from_str(&xml).unwrap();

        assert_eq!(cdm1.header.originator, cdm2.header.originator);
        assert!((cdm1.miss_distance() - cdm2.miss_distance()).abs() < 1e-6);

        for i in 0..6 {
            assert!((cdm1.object1_state()[i] - cdm2.object1_state()[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_cdm_json_round_trip() {
        let cdm1 = CDM::from_file("test_assets/ccsds/cdm/CDMExample1.txt").unwrap();
        let json = cdm1.to_string(CCSDSFormat::JSON).unwrap();
        let cdm2 = CDM::from_str(&json).unwrap();

        assert_eq!(cdm1.header.originator, cdm2.header.originator);
        assert!((cdm1.miss_distance() - cdm2.miss_distance()).abs() < 1e-6);
        for i in 0..6 {
            assert!(
                (cdm1.object1_state()[i] - cdm2.object1_state()[i]).abs() < 0.01,
                "Object1 state[{}]: {} vs {}",
                i,
                cdm1.object1_state()[i],
                cdm2.object1_state()[i]
            );
        }
    }

    #[test]
    fn test_cdm_kvn_to_xml_cross_format() {
        // Parse KVN
        let cdm_kvn = CDM::from_file("test_assets/ccsds/cdm/CDMExample1.txt").unwrap();
        // Write as XML
        let xml = cdm_kvn.to_string(CCSDSFormat::XML).unwrap();
        // Re-parse from XML
        let cdm_xml = CDM::from_str(&xml).unwrap();

        // Compare
        assert_eq!(cdm_kvn.header.originator, cdm_xml.header.originator);
        assert!((cdm_kvn.miss_distance() - cdm_xml.miss_distance()).abs() < 1e-6);
        for i in 0..6 {
            assert!((cdm_kvn.object1_state()[i] - cdm_xml.object1_state()[i]).abs() < 0.01);
            assert!((cdm_kvn.object2_state()[i] - cdm_xml.object2_state()[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_cdm_kvn_round_trip_example2() {
        let cdm1 = CDM::from_file("test_assets/ccsds/cdm/CDMExample2.txt").unwrap();
        let kvn = cdm1.to_string(CCSDSFormat::KVN).unwrap();
        let cdm2 = CDM::from_str(&kvn).unwrap();

        assert_eq!(cdm1.header.message_for, cdm2.header.message_for);
        assert!(
            (cdm1.collision_probability().unwrap() - cdm2.collision_probability().unwrap()).abs()
                < 1e-10
        );
        assert_eq!(
            cdm1.relative_metadata.collision_probability_method,
            cdm2.relative_metadata.collision_probability_method
        );

        // Extended covariance dimension preserved
        assert_eq!(
            cdm1.object1.data.rtn_covariance.dimension,
            cdm2.object1.data.rtn_covariance.dimension
        );

        // OD parameters
        let od1 = cdm1.object1.data.od_parameters.as_ref().unwrap();
        let od2 = cdm2.object1.data.od_parameters.as_ref().unwrap();
        assert_eq!(od1.obs_available, od2.obs_available);
        assert_eq!(od1.obs_used, od2.obs_used);
    }

    #[test]
    fn test_cdm_rtn_covariance_6x6() {
        let mut m6 = SMatrix::<f64, 6, 6>::zeros();
        m6[(0, 0)] = 41.42;
        m6[(1, 0)] = -8.579;
        m6[(0, 1)] = -8.579;
        m6[(1, 1)] = 2533.0;
        let cov = CDMRTNCovariance::from_6x6(m6);
        assert_eq!(cov.dimension, CDMCovarianceDimension::SixBySix);
        let extracted = cov.to_6x6();
        assert_eq!(extracted[(0, 0)], 41.42);
        assert_eq!(extracted[(1, 0)], -8.579);
        assert_eq!(extracted[(1, 1)], 2533.0);
        // Extended region should be zero
        assert_eq!(cov.matrix[(6, 0)], 0.0);
        assert_eq!(cov.matrix[(8, 8)], 0.0);
    }
}
