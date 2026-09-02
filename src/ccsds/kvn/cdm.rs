/*!
 * KVN reader and writer for the Conjunction Data Message (CDM).
 *
 * Reference: CCSDS 508.0-B-1 (Conjunction Data Message), section 3
 */

use std::collections::HashMap;

use crate::ccsds::common::{
    CCSDSRefFrame, CCSDSTimeSystem, CCSDSUserDefined, format_ccsds_datetime_in,
    parse_ccsds_datetime, strip_units,
};
use crate::ccsds::error::{ccsds_missing_field, ccsds_parse_error};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

/// The logical block a CDM keyword belongs to, per the CCSDS 508.0-B-1 tables.
/// Table 3-4 gives the object Data section a COMMENT row of its own and then a
/// further COMMENT row for each of its sub-blocks.
#[derive(Clone, Copy, PartialEq)]
enum CDMBlock {
    Metadata,
    ODParameters,
    AdditionalParameters,
    StateVector,
    RTNCovariance,
    XYZCovariance,
    AdditionalCovarianceMetadata,
}

/// The comment bucket a CDM keyword's section owns.
#[derive(Clone, Copy, PartialEq)]
enum CommentTarget {
    Header,
    RelativeMetadata,
    Object(CDMBlock),
}

/// Parse a CDM message from KVN format.
///
/// Uses a flat key-match approach with object context tracking. The `OBJECT`
/// keyword triggers transitions between Object1 and Object2. Within each
/// object, field names implicitly determine the subsection.
pub fn parse_cdm(content: &str) -> Result<crate::ccsds::cdm::CDM, BraheError> {
    use crate::ccsds::cdm::*;
    use crate::ccsds::common::covariance9x9_from_lower_triangular;

    // Track which object we're currently parsing
    #[derive(Clone, Copy, PartialEq)]
    enum CurrentObject {
        None,
        Object1,
        Object2,
    }
    let mut current_object = CurrentObject::None;

    // Comments accumulate until a keyword identifies the section they
    // introduce. CCSDS 508.0-B-1 subsection 6.3.1.9 fixes the keyword order to
    // that of tables 3-1 through 3-4, each of which opens its section with a
    // COMMENT row, so a comment belongs to the section of the keyword that
    // follows it rather than the one that precedes it.
    let mut pending_comments: Vec<String> = Vec::new();
    let mut last_target = CommentTarget::Header;

    // Header fields
    let mut format_version: Option<f64> = None;
    let mut classification: Option<String> = None;
    let mut creation_date: Option<Epoch> = None;
    let mut originator: Option<String> = None;
    let mut message_for: Option<String> = None;
    let mut message_id: Option<String> = None;
    let mut header_comments: Vec<String> = Vec::new();

    // Relative metadata fields
    let mut conjunction_id: Option<String> = None;
    let mut tca: Option<Epoch> = None;
    let mut miss_distance: Option<f64> = None;
    let mut mahalanobis_distance: Option<f64> = None;
    let mut relative_speed: Option<f64> = None;
    let mut rel_pos_r: Option<f64> = None;
    let mut rel_pos_t: Option<f64> = None;
    let mut rel_pos_n: Option<f64> = None;
    let mut rel_vel_r: Option<f64> = None;
    let mut rel_vel_t: Option<f64> = None;
    let mut rel_vel_n: Option<f64> = None;
    let mut approach_angle: Option<f64> = None;
    let mut start_screen_period: Option<Epoch> = None;
    let mut stop_screen_period: Option<Epoch> = None;
    let mut screen_type: Option<String> = None;
    let mut screen_volume_frame: Option<CCSDSRefFrame> = None;
    let mut screen_volume_shape: Option<String> = None;
    let mut screen_volume_radius: Option<f64> = None;
    let mut screen_volume_x: Option<f64> = None;
    let mut screen_volume_y: Option<f64> = None;
    let mut screen_volume_z: Option<f64> = None;
    let mut screen_entry_time: Option<Epoch> = None;
    let mut screen_exit_time: Option<Epoch> = None;
    let mut screen_pc_threshold: Option<f64> = None;
    let mut collision_percentile: Option<Vec<u32>> = None;
    let mut collision_probability: Option<f64> = None;
    let mut collision_probability_method: Option<String> = None;
    let mut collision_max_probability: Option<f64> = None;
    let mut collision_max_pc_method: Option<String> = None;
    let mut sefi_collision_probability: Option<f64> = None;
    let mut sefi_collision_probability_method: Option<String> = None;
    let mut sefi_fragmentation_model: Option<String> = None;
    let mut previous_message_id: Option<String> = None;
    let mut previous_message_epoch: Option<Epoch> = None;
    let mut next_message_epoch: Option<Epoch> = None;
    let mut rel_comments: Vec<String> = Vec::new();

    // Per-object data (index 0 = object1, 1 = object2)
    struct ObjectBuilder {
        // Metadata
        object: Option<String>,
        object_designator: Option<String>,
        catalog_name: Option<String>,
        object_name: Option<String>,
        international_designator: Option<String>,
        object_type: Option<String>,
        ops_status: Option<String>,
        operator_contact_position: Option<String>,
        operator_organization: Option<String>,
        operator_phone: Option<String>,
        operator_email: Option<String>,
        ephemeris_name: Option<String>,
        odm_msg_link: Option<String>,
        adm_msg_link: Option<String>,
        obs_before_next_message: Option<String>,
        covariance_method: Option<String>,
        covariance_source: Option<String>,
        maneuverable: Option<String>,
        orbit_center: Option<String>,
        ref_frame: Option<CCSDSRefFrame>,
        alt_cov_type: Option<String>,
        alt_cov_ref_frame: Option<CCSDSRefFrame>,
        gravity_model: Option<String>,
        atmospheric_model: Option<String>,
        n_body_perturbations: Option<String>,
        solar_rad_pressure: Option<String>,
        earth_tides: Option<String>,
        intrack_thrust: Option<String>,
        metadata_comments: Vec<String>,

        // OD parameters
        time_lastob_start: Option<Epoch>,
        time_lastob_end: Option<Epoch>,
        recommended_od_span: Option<f64>,
        actual_od_span: Option<f64>,
        obs_available: Option<u32>,
        obs_used: Option<u32>,
        tracks_available: Option<u32>,
        tracks_used: Option<u32>,
        residuals_accepted: Option<f64>,
        weighted_rms: Option<f64>,
        od_epoch: Option<Epoch>,
        od_comments: Vec<String>,
        has_od_params: bool,

        // Additional parameters
        area_pc: Option<f64>,
        area_pc_min: Option<f64>,
        area_pc_max: Option<f64>,
        area_drg: Option<f64>,
        area_srp: Option<f64>,
        oeb_parent_frame: Option<String>,
        oeb_parent_frame_epoch: Option<Epoch>,
        oeb_q1: Option<f64>,
        oeb_q2: Option<f64>,
        oeb_q3: Option<f64>,
        oeb_qc: Option<f64>,
        oeb_max: Option<f64>,
        oeb_int: Option<f64>,
        oeb_min: Option<f64>,
        area_along_oeb_max: Option<f64>,
        area_along_oeb_int: Option<f64>,
        area_along_oeb_min: Option<f64>,
        rcs: Option<f64>,
        rcs_min: Option<f64>,
        rcs_max: Option<f64>,
        vm_absolute: Option<f64>,
        vm_apparent_min: Option<f64>,
        vm_apparent: Option<f64>,
        vm_apparent_max: Option<f64>,
        reflectance: Option<f64>,
        mass: Option<f64>,
        hbr: Option<f64>,
        cd_area_over_mass: Option<f64>,
        cr_area_over_mass: Option<f64>,
        thrust_acceleration: Option<f64>,
        sedr: Option<f64>,
        min_dv: Option<[f64; 3]>,
        max_dv: Option<[f64; 3]>,
        lead_time_reqd_before_tca: Option<f64>,
        apoapsis_altitude: Option<f64>,
        periapsis_altitude: Option<f64>,
        inclination: Option<f64>,
        cov_confidence: Option<f64>,
        cov_confidence_method: Option<String>,
        add_comments: Vec<String>,
        has_add_params: bool,

        // State vector
        x: Option<f64>,
        y: Option<f64>,
        z: Option<f64>,
        x_dot: Option<f64>,
        y_dot: Option<f64>,
        z_dot: Option<f64>,
        sv_comments: Vec<String>,

        // RTN covariance (store as lower-triangular values)
        rtn_cov_values: Vec<f64>,
        rtn_cov_comments: Vec<String>,

        // XYZ covariance (store as lower-triangular values)
        xyz_cov_values: Vec<f64>,
        xyz_cov_comments: Vec<String>,

        // CSIG3EIGVEC3
        csig3eigvec3: Option<String>,

        // Additional covariance metadata
        density_forecast_uncertainty: Option<f64>,
        cscale_factor_min: Option<f64>,
        cscale_factor: Option<f64>,
        cscale_factor_max: Option<f64>,
        screening_data_source: Option<String>,
        dcp_sensitivity_vector_position: Option<[f64; 3]>,
        dcp_sensitivity_vector_velocity: Option<[f64; 3]>,
        acm_comments: Vec<String>,
        has_acm: bool,

        data_comments: Vec<String>,
        in_xyz_cov: bool,
    }

    impl ObjectBuilder {
        fn new() -> Self {
            Self {
                object: None,
                object_designator: None,
                catalog_name: None,
                object_name: None,
                international_designator: None,
                object_type: None,
                ops_status: None,
                operator_contact_position: None,
                operator_organization: None,
                operator_phone: None,
                operator_email: None,
                ephemeris_name: None,
                odm_msg_link: None,
                adm_msg_link: None,
                obs_before_next_message: None,
                covariance_method: None,
                covariance_source: None,
                maneuverable: None,
                orbit_center: None,
                ref_frame: None,
                alt_cov_type: None,
                alt_cov_ref_frame: None,
                gravity_model: None,
                atmospheric_model: None,
                n_body_perturbations: None,
                solar_rad_pressure: None,
                earth_tides: None,
                intrack_thrust: None,
                metadata_comments: Vec::new(),

                time_lastob_start: None,
                time_lastob_end: None,
                recommended_od_span: None,
                actual_od_span: None,
                obs_available: None,
                obs_used: None,
                tracks_available: None,
                tracks_used: None,
                residuals_accepted: None,
                weighted_rms: None,
                od_epoch: None,
                od_comments: Vec::new(),
                has_od_params: false,

                area_pc: None,
                area_pc_min: None,
                area_pc_max: None,
                area_drg: None,
                area_srp: None,
                oeb_parent_frame: None,
                oeb_parent_frame_epoch: None,
                oeb_q1: None,
                oeb_q2: None,
                oeb_q3: None,
                oeb_qc: None,
                oeb_max: None,
                oeb_int: None,
                oeb_min: None,
                area_along_oeb_max: None,
                area_along_oeb_int: None,
                area_along_oeb_min: None,
                rcs: None,
                rcs_min: None,
                rcs_max: None,
                vm_absolute: None,
                vm_apparent_min: None,
                vm_apparent: None,
                vm_apparent_max: None,
                reflectance: None,
                mass: None,
                hbr: None,
                cd_area_over_mass: None,
                cr_area_over_mass: None,
                thrust_acceleration: None,
                sedr: None,
                min_dv: None,
                max_dv: None,
                lead_time_reqd_before_tca: None,
                apoapsis_altitude: None,
                periapsis_altitude: None,
                inclination: None,
                cov_confidence: None,
                cov_confidence_method: None,
                add_comments: Vec::new(),
                has_add_params: false,

                x: None,
                y: None,
                z: None,
                x_dot: None,
                y_dot: None,
                z_dot: None,
                sv_comments: Vec::new(),

                rtn_cov_values: Vec::new(),
                rtn_cov_comments: Vec::new(),
                xyz_cov_values: Vec::new(),
                xyz_cov_comments: Vec::new(),
                csig3eigvec3: None,

                density_forecast_uncertainty: None,
                cscale_factor_min: None,
                cscale_factor: None,
                cscale_factor_max: None,
                screening_data_source: None,
                dcp_sensitivity_vector_position: None,
                dcp_sensitivity_vector_velocity: None,
                acm_comments: Vec::new(),
                has_acm: false,

                data_comments: Vec::new(),
                in_xyz_cov: false,
            }
        }
    }

    let mut obj1 = ObjectBuilder::new();
    let mut obj2 = ObjectBuilder::new();
    let mut user_defined: HashMap<String, String> = HashMap::new();

    let utc = CCSDSTimeSystem::UTC;

    // Helper: parse float from value with unit stripping
    let parse_f64 = |val: &str| -> Result<f64, BraheError> {
        strip_units(val)
            .parse()
            .map_err(|_| ccsds_parse_error("CDM", &format!("invalid numeric value '{}'", val)))
    };
    let parse_u32 = |val: &str| -> Result<u32, BraheError> {
        strip_units(val)
            .parse()
            .map_err(|_| ccsds_parse_error("CDM", &format!("invalid integer value '{}'", val)))
    };

    // RTN covariance field name → (row, col) mapping
    fn rtn_cov_index(key: &str) -> Option<(usize, usize)> {
        match key {
            "CR_R" => Some((0, 0)),
            "CT_R" => Some((1, 0)),
            "CT_T" => Some((1, 1)),
            "CN_R" => Some((2, 0)),
            "CN_T" => Some((2, 1)),
            "CN_N" => Some((2, 2)),
            "CRDOT_R" => Some((3, 0)),
            "CRDOT_T" => Some((3, 1)),
            "CRDOT_N" => Some((3, 2)),
            "CRDOT_RDOT" => Some((3, 3)),
            "CTDOT_R" => Some((4, 0)),
            "CTDOT_T" => Some((4, 1)),
            "CTDOT_N" => Some((4, 2)),
            "CTDOT_RDOT" => Some((4, 3)),
            "CTDOT_TDOT" => Some((4, 4)),
            "CNDOT_R" => Some((5, 0)),
            "CNDOT_T" => Some((5, 1)),
            "CNDOT_N" => Some((5, 2)),
            "CNDOT_RDOT" => Some((5, 3)),
            "CNDOT_TDOT" => Some((5, 4)),
            "CNDOT_NDOT" => Some((5, 5)),
            "CDRG_R" => Some((6, 0)),
            "CDRG_T" => Some((6, 1)),
            "CDRG_N" => Some((6, 2)),
            "CDRG_RDOT" => Some((6, 3)),
            "CDRG_TDOT" => Some((6, 4)),
            "CDRG_NDOT" => Some((6, 5)),
            "CDRG_DRG" => Some((6, 6)),
            "CSRP_R" => Some((7, 0)),
            "CSRP_T" => Some((7, 1)),
            "CSRP_N" => Some((7, 2)),
            "CSRP_RDOT" => Some((7, 3)),
            "CSRP_TDOT" => Some((7, 4)),
            "CSRP_NDOT" => Some((7, 5)),
            "CSRP_DRG" => Some((7, 6)),
            "CSRP_SRP" => Some((7, 7)),
            "CTHR_R" => Some((8, 0)),
            "CTHR_T" => Some((8, 1)),
            "CTHR_N" => Some((8, 2)),
            "CTHR_RDOT" => Some((8, 3)),
            "CTHR_TDOT" => Some((8, 4)),
            "CTHR_NDOT" => Some((8, 5)),
            "CTHR_DRG" => Some((8, 6)),
            "CTHR_SRP" => Some((8, 7)),
            "CTHR_THR" => Some((8, 8)),
            _ => None,
        }
    }

    // XYZ covariance field name → (row, col) mapping
    fn xyz_cov_index(key: &str) -> Option<(usize, usize)> {
        match key {
            "CX_X" => Some((0, 0)),
            "CY_X" => Some((1, 0)),
            "CY_Y" => Some((1, 1)),
            "CZ_X" => Some((2, 0)),
            "CZ_Y" => Some((2, 1)),
            "CZ_Z" => Some((2, 2)),
            "CXDOT_X" => Some((3, 0)),
            "CXDOT_Y" => Some((3, 1)),
            "CXDOT_Z" => Some((3, 2)),
            "CXDOT_XDOT" => Some((3, 3)),
            "CYDOT_X" => Some((4, 0)),
            "CYDOT_Y" => Some((4, 1)),
            "CYDOT_Z" => Some((4, 2)),
            "CYDOT_XDOT" => Some((4, 3)),
            "CYDOT_YDOT" => Some((4, 4)),
            "CZDOT_X" => Some((5, 0)),
            "CZDOT_Y" => Some((5, 1)),
            "CZDOT_Z" => Some((5, 2)),
            "CZDOT_XDOT" => Some((5, 3)),
            "CZDOT_YDOT" => Some((5, 4)),
            "CZDOT_ZDOT" => Some((5, 5)),
            "CDRG_X" => Some((6, 0)),
            "CDRG_Y" => Some((6, 1)),
            "CDRG_Z" => Some((6, 2)),
            "CDRG_XDOT" => Some((6, 3)),
            "CDRG_YDOT" => Some((6, 4)),
            "CDRG_ZDOT" => Some((6, 5)),
            // CDRG_DRG is shared between RTN and XYZ contexts
            "CSRP_X" => Some((7, 0)),
            "CSRP_Y" => Some((7, 1)),
            "CSRP_Z" => Some((7, 2)),
            "CSRP_XDOT" => Some((7, 3)),
            "CSRP_YDOT" => Some((7, 4)),
            "CSRP_ZDOT" => Some((7, 5)),
            // CSRP_DRG and CSRP_SRP shared
            "CTHR_X" => Some((8, 0)),
            "CTHR_Y" => Some((8, 1)),
            "CTHR_Z" => Some((8, 2)),
            "CTHR_XDOT" => Some((8, 3)),
            "CTHR_YDOT" => Some((8, 4)),
            "CTHR_ZDOT" => Some((8, 5)),
            // CTHR_DRG, CTHR_SRP, CTHR_THR shared
            _ => None,
        }
    }

    // The section a CDM keyword introduces, or `None` when the keyword is not
    // one this parser recognizes. The shared covariance keywords (`CDRG_DRG`,
    // `CSRP_SRP`, `CTHR_THR` and their neighbours) name an element of whichever
    // covariance block is open, so they take the same `in_xyz_cov` context the
    // value dispatch uses.
    let comment_target = |key: &str, in_xyz_cov: bool| -> Option<CommentTarget> {
        Some(match key {
            // Table 3-1: header
            "CCSDS_CDM_VERS" | "CLASSIFICATION" | "CREATION_DATE" | "ORIGINATOR"
            | "MESSAGE_FOR" | "MESSAGE_ID" => CommentTarget::Header,

            // Table 3-2: relative metadata/data
            "CONJUNCTION_ID"
            | "TCA"
            | "MISS_DISTANCE"
            | "MAHALANOBIS_DISTANCE"
            | "RELATIVE_SPEED"
            | "RELATIVE_POSITION_R"
            | "RELATIVE_POSITION_T"
            | "RELATIVE_POSITION_N"
            | "RELATIVE_VELOCITY_R"
            | "RELATIVE_VELOCITY_T"
            | "RELATIVE_VELOCITY_N"
            | "APPROACH_ANGLE"
            | "START_SCREEN_PERIOD"
            | "STOP_SCREEN_PERIOD"
            | "SCREEN_TYPE"
            | "SCREEN_VOLUME_FRAME"
            | "SCREEN_VOLUME_SHAPE"
            | "SCREEN_VOLUME_RADIUS"
            | "SCREEN_VOLUME_X"
            | "SCREEN_VOLUME_Y"
            | "SCREEN_VOLUME_Z"
            | "SCREEN_ENTRY_TIME"
            | "SCREEN_EXIT_TIME"
            | "SCREEN_PC_THRESHOLD"
            | "COLLISION_PERCENTILE"
            | "COLLISION_PROBABILITY"
            | "COLLISION_PROBABILITY_METHOD"
            | "COLLISION_MAX_PROBABILITY"
            | "COLLISION_MAX_PC_METHOD"
            | "SEFI_COLLISION_PROBABILITY"
            | "SEFI_COLLISION_PROBABILITY_METHOD"
            | "SEFI_FRAGMENTATION_MODEL"
            | "PREVIOUS_MESSAGE_ID"
            | "PREVIOUS_MESSAGE_EPOCH"
            | "NEXT_MESSAGE_EPOCH" => CommentTarget::RelativeMetadata,

            // Table 3-3: object metadata
            "OBJECT"
            | "OBJECT_DESIGNATOR"
            | "CATALOG_NAME"
            | "OBJECT_NAME"
            | "INTERNATIONAL_DESIGNATOR"
            | "OBJECT_TYPE"
            | "OPS_STATUS"
            | "OPERATOR_CONTACT_POSITION"
            | "OPERATOR_ORGANIZATION"
            | "OPERATOR_PHONE"
            | "OPERATOR_EMAIL"
            | "EPHEMERIS_NAME"
            | "ODM_MSG_LINK"
            | "ADM_MSG_LINK"
            | "OBS_BEFORE_NEXT_MESSAGE"
            | "COVARIANCE_METHOD"
            | "COVARIANCE_SOURCE"
            | "MANEUVERABLE"
            | "ORBIT_CENTER"
            | "REF_FRAME"
            | "ALT_COV_TYPE"
            | "ALT_COV_REF_FRAME"
            | "GRAVITY_MODEL"
            | "ATMOSPHERIC_MODEL"
            | "N_BODY_PERTURBATIONS"
            | "SOLAR_RAD_PRESSURE"
            | "EARTH_TIDES"
            | "INTRACK_THRUST" => CommentTarget::Object(CDMBlock::Metadata),

            // Table 3-4: OD parameters
            "TIME_LASTOB_START"
            | "TIME_LASTOB_END"
            | "RECOMMENDED_OD_SPAN"
            | "ACTUAL_OD_SPAN"
            | "OBS_AVAILABLE"
            | "OBS_USED"
            | "TRACKS_AVAILABLE"
            | "TRACKS_USED"
            | "RESIDUALS_ACCEPTED"
            | "WEIGHTED_RMS"
            | "OD_EPOCH" => CommentTarget::Object(CDMBlock::ODParameters),

            // Table 3-4: additional parameters
            "AREA_PC"
            | "AREA_PC_MIN"
            | "AREA_PC_MAX"
            | "AREA_DRG"
            | "AREA_SRP"
            | "OEB_PARENT_FRAME"
            | "OEB_PARENT_FRAME_EPOCH"
            | "OEB_Q1"
            | "OEB_Q2"
            | "OEB_Q3"
            | "OEB_QC"
            | "OEB_MAX"
            | "OEB_INT"
            | "OEB_MIN"
            | "AREA_ALONG_OEB_MAX"
            | "AREA_ALONG_OEB_INT"
            | "AREA_ALONG_OEB_MIN"
            | "RCS"
            | "RCS_MIN"
            | "RCS_MAX"
            | "VM_ABSOLUTE"
            | "VM_APPARENT_MIN"
            | "VM_APPARENT"
            | "VM_APPARENT_MAX"
            | "REFLECTANCE"
            | "MASS"
            | "HBR"
            | "CD_AREA_OVER_MASS"
            | "CR_AREA_OVER_MASS"
            | "THRUST_ACCELERATION"
            | "SEDR"
            | "MIN_DV"
            | "MAX_DV"
            | "LEAD_TIME_REQD_BEFORE_TCA"
            | "APOAPSIS_ALTITUDE"
            | "PERIAPSIS_ALTITUDE"
            | "INCLINATION"
            | "COV_CONFIDENCE"
            | "COV_CONFIDENCE_METHOD" => CommentTarget::Object(CDMBlock::AdditionalParameters),

            // Table 3-4: state vector
            "X" | "Y" | "Z" | "X_DOT" | "Y_DOT" | "Z_DOT" => {
                CommentTarget::Object(CDMBlock::StateVector)
            }

            // Table 3-4: additional covariance metadata
            "DENSITY_FORECAST_UNCERTAINTY"
            | "CSCALE_FACTOR_MIN"
            | "CSCALE_FACTOR"
            | "CSCALE_FACTOR_MAX"
            | "SCREENING_DATA_SOURCE"
            | "DCP_SENSITIVITY_VECTOR_POSITION"
            | "DCP_SENSITIVITY_VECTOR_VELOCITY"
            | "CSIG3EIGVEC3" => CommentTarget::Object(CDMBlock::AdditionalCovarianceMetadata),

            // Table 3-4: covariance matrices
            "CR_R" => CommentTarget::Object(CDMBlock::RTNCovariance),
            k if xyz_cov_index(k).is_some() => CommentTarget::Object(CDMBlock::XYZCovariance),
            k if rtn_cov_index(k).is_some() => CommentTarget::Object(if in_xyz_cov {
                CDMBlock::XYZCovariance
            } else {
                CDMBlock::RTNCovariance
            }),

            _ => return None,
        })
    };

    // Move the buffered comments into the bucket their section owns. An
    // `Object` target before either object has been opened cannot be filed
    // yet, so those comments stay buffered for the keyword after it.
    macro_rules! flush_comments {
        ($target:expr, $object:expr) => {{
            if !pending_comments.is_empty() {
                let sink: Option<&mut Vec<String>> = match ($target, $object) {
                    (CommentTarget::Header, _) => Some(&mut header_comments),
                    (CommentTarget::RelativeMetadata, _) => Some(&mut rel_comments),
                    (CommentTarget::Object(_), CurrentObject::None) => None,
                    (CommentTarget::Object(block), which) => {
                        let target_obj = match which {
                            CurrentObject::Object2 => &mut obj2,
                            _ => &mut obj1,
                        };
                        Some(match block {
                            CDMBlock::Metadata => &mut target_obj.metadata_comments,
                            CDMBlock::ODParameters => &mut target_obj.od_comments,
                            CDMBlock::AdditionalParameters => &mut target_obj.add_comments,
                            CDMBlock::StateVector => &mut target_obj.sv_comments,
                            CDMBlock::RTNCovariance => &mut target_obj.rtn_cov_comments,
                            CDMBlock::XYZCovariance => &mut target_obj.xyz_cov_comments,
                            CDMBlock::AdditionalCovarianceMetadata => &mut target_obj.acm_comments,
                        })
                    }
                };
                if let Some(sink) = sink {
                    sink.append(&mut pending_comments);
                }
            }
        }};
    }

    // Parse line-by-line
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Parse COMMENT lines
        if let Some(comment_text) = line.strip_prefix("COMMENT") {
            pending_comments.push(comment_text.trim().to_string());
            continue;
        }

        // Parse key=value
        let eq_pos = match line.find('=') {
            Some(pos) => pos,
            None => continue,
        };
        let key = line[..eq_pos].trim();
        let raw_val = line[eq_pos + 1..].trim();
        let val = strip_units(raw_val);

        // File the buffered comments under the section this keyword opens. The
        // OBJECT keyword switches objects, so the comments introducing it
        // belong to the object it names rather than the one before it.
        let switching = match (key, val) {
            ("OBJECT", "OBJECT1") => Some(CurrentObject::Object1),
            ("OBJECT", "OBJECT2") => Some(CurrentObject::Object2),
            _ => None,
        };
        let comment_object = switching.unwrap_or(current_object);
        let in_xyz_cov = match comment_object {
            CurrentObject::Object2 => obj2.in_xyz_cov,
            _ => obj1.in_xyz_cov,
        };
        if let Some(target) = comment_target(key, in_xyz_cov) {
            last_target = target;
            flush_comments!(target, comment_object);
        }

        // Get mutable reference to current object builder
        let obj = match current_object {
            CurrentObject::Object1 => &mut obj1,
            CurrentObject::Object2 => &mut obj2,
            CurrentObject::None => {
                // Header + relative metadata keys
                match key {
                    "CCSDS_CDM_VERS" => {
                        format_version = Some(parse_f64(val)?);
                    }
                    "CLASSIFICATION" => {
                        classification = Some(val.trim_matches('"').to_string());
                    }
                    "CREATION_DATE" => {
                        creation_date = Some(parse_ccsds_datetime(val, &utc)?);
                    }
                    "ORIGINATOR" => {
                        originator = Some(val.to_string());
                    }
                    "MESSAGE_FOR" => {
                        message_for = Some(val.to_string());
                    }
                    "MESSAGE_ID" => {
                        message_id = Some(val.to_string());
                    }
                    "CONJUNCTION_ID" => {
                        conjunction_id = Some(val.to_string());
                    }
                    "TCA" => {
                        tca = Some(parse_ccsds_datetime(val, &utc)?);
                    }
                    "MISS_DISTANCE" => {
                        miss_distance = Some(parse_f64(val)?);
                    }
                    "MAHALANOBIS_DISTANCE" => {
                        mahalanobis_distance = Some(parse_f64(val)?);
                    }
                    "RELATIVE_SPEED" => {
                        relative_speed = Some(parse_f64(val)?);
                    }
                    "RELATIVE_POSITION_R" => {
                        rel_pos_r = Some(parse_f64(val)?);
                    }
                    "RELATIVE_POSITION_T" => {
                        rel_pos_t = Some(parse_f64(val)?);
                    }
                    "RELATIVE_POSITION_N" => {
                        rel_pos_n = Some(parse_f64(val)?);
                    }
                    "RELATIVE_VELOCITY_R" => {
                        rel_vel_r = Some(parse_f64(val)?);
                    }
                    "RELATIVE_VELOCITY_T" => {
                        rel_vel_t = Some(parse_f64(val)?);
                    }
                    "RELATIVE_VELOCITY_N" => {
                        rel_vel_n = Some(parse_f64(val)?);
                    }
                    "APPROACH_ANGLE" => {
                        approach_angle = Some(parse_f64(val)?);
                    }
                    "START_SCREEN_PERIOD" => {
                        start_screen_period = Some(parse_ccsds_datetime(val, &utc)?);
                    }
                    "STOP_SCREEN_PERIOD" => {
                        stop_screen_period = Some(parse_ccsds_datetime(val, &utc)?);
                    }
                    "SCREEN_TYPE" => {
                        screen_type = Some(val.to_string());
                    }
                    "SCREEN_VOLUME_FRAME" => {
                        screen_volume_frame = Some(CCSDSRefFrame::parse(val));
                    }
                    "SCREEN_VOLUME_SHAPE" => {
                        screen_volume_shape = Some(val.to_string());
                    }
                    "SCREEN_VOLUME_RADIUS" => {
                        screen_volume_radius = Some(parse_f64(val)?);
                    }
                    "SCREEN_VOLUME_X" => {
                        screen_volume_x = Some(parse_f64(val)?);
                    }
                    "SCREEN_VOLUME_Y" => {
                        screen_volume_y = Some(parse_f64(val)?);
                    }
                    "SCREEN_VOLUME_Z" => {
                        screen_volume_z = Some(parse_f64(val)?);
                    }
                    "SCREEN_ENTRY_TIME" => {
                        screen_entry_time = Some(parse_ccsds_datetime(val, &utc)?);
                    }
                    "SCREEN_EXIT_TIME" => {
                        screen_exit_time = Some(parse_ccsds_datetime(val, &utc)?);
                    }
                    "SCREEN_PC_THRESHOLD" => {
                        screen_pc_threshold = Some(parse_f64(val)?);
                    }
                    "COLLISION_PERCENTILE" => {
                        let parts: Result<Vec<u32>, _> =
                            val.split_whitespace().map(|s| s.parse::<u32>()).collect();
                        collision_percentile = Some(parts.map_err(|_| {
                            ccsds_parse_error("CDM", "invalid COLLISION_PERCENTILE")
                        })?);
                    }
                    "COLLISION_PROBABILITY" => {
                        collision_probability = Some(parse_f64(val)?);
                    }
                    "COLLISION_PROBABILITY_METHOD" => {
                        collision_probability_method = Some(val.to_string());
                    }
                    "COLLISION_MAX_PROBABILITY" => {
                        collision_max_probability = Some(parse_f64(val)?);
                    }
                    "COLLISION_MAX_PC_METHOD" => {
                        collision_max_pc_method = Some(val.to_string());
                    }
                    "SEFI_COLLISION_PROBABILITY" => {
                        sefi_collision_probability = Some(parse_f64(val)?);
                    }
                    "SEFI_COLLISION_PROBABILITY_METHOD" => {
                        sefi_collision_probability_method = Some(val.to_string());
                    }
                    "SEFI_FRAGMENTATION_MODEL" => {
                        sefi_fragmentation_model = Some(val.to_string());
                    }
                    "PREVIOUS_MESSAGE_ID" => {
                        previous_message_id = Some(val.to_string());
                    }
                    "PREVIOUS_MESSAGE_EPOCH" => {
                        previous_message_epoch = Some(parse_ccsds_datetime(val, &utc)?);
                    }
                    "NEXT_MESSAGE_EPOCH" => {
                        next_message_epoch = Some(parse_ccsds_datetime(val, &utc)?);
                    }
                    "OBJECT" => match val {
                        "OBJECT1" => {
                            current_object = CurrentObject::Object1;
                            obj1.object = Some("OBJECT1".to_string());
                        }
                        "OBJECT2" => {
                            current_object = CurrentObject::Object2;
                            obj2.object = Some("OBJECT2".to_string());
                        }
                        _ => {
                            return Err(ccsds_parse_error(
                                "CDM",
                                &format!("unexpected OBJECT value '{}'", val),
                            ));
                        }
                    },
                    k if k.starts_with("USER_DEFINED_") => {
                        let ud_key = k.strip_prefix("USER_DEFINED_").unwrap_or(k);
                        user_defined.insert(ud_key.to_string(), val.to_string());
                    }
                    _ => {} // Ignore unknown keys in header/relative metadata
                }
                continue;
            }
        };

        // Object-level keyword dispatch
        match key {
            "OBJECT" => match val {
                "OBJECT1" => {
                    current_object = CurrentObject::Object1;
                    obj1.object = Some("OBJECT1".to_string());
                }
                "OBJECT2" => {
                    current_object = CurrentObject::Object2;
                    obj2.object = Some("OBJECT2".to_string());
                }
                _ => {
                    return Err(ccsds_parse_error(
                        "CDM",
                        &format!("unexpected OBJECT value '{}'", val),
                    ));
                }
            },

            // Metadata fields
            "OBJECT_DESIGNATOR" => {
                obj.object_designator = Some(val.to_string());
            }
            "CATALOG_NAME" => {
                obj.catalog_name = Some(val.to_string());
            }
            "OBJECT_NAME" => {
                obj.object_name = Some(val.to_string());
            }
            "INTERNATIONAL_DESIGNATOR" => {
                obj.international_designator = Some(val.to_string());
            }
            "OBJECT_TYPE" => {
                obj.object_type = Some(val.to_string());
            }
            "OPS_STATUS" => {
                obj.ops_status = Some(val.to_string());
            }
            "OPERATOR_CONTACT_POSITION" => {
                obj.operator_contact_position = Some(val.to_string());
            }
            "OPERATOR_ORGANIZATION" => {
                obj.operator_organization = Some(val.to_string());
            }
            "OPERATOR_PHONE" => {
                obj.operator_phone = Some(val.to_string());
            }
            "OPERATOR_EMAIL" => {
                obj.operator_email = Some(val.to_string());
            }
            "EPHEMERIS_NAME" => {
                obj.ephemeris_name = Some(val.to_string());
            }
            "ODM_MSG_LINK" => {
                obj.odm_msg_link = Some(val.to_string());
            }
            "ADM_MSG_LINK" => {
                obj.adm_msg_link = Some(val.to_string());
            }
            "OBS_BEFORE_NEXT_MESSAGE" => {
                obj.obs_before_next_message = Some(val.to_string());
            }
            "COVARIANCE_METHOD" => {
                obj.covariance_method = Some(val.to_string());
            }
            "COVARIANCE_SOURCE" => {
                obj.covariance_source = Some(val.to_string());
            }
            "MANEUVERABLE" => {
                obj.maneuverable = Some(val.to_string());
            }
            "ORBIT_CENTER" => {
                obj.orbit_center = Some(val.to_string());
            }
            "REF_FRAME" => {
                obj.ref_frame = Some(CCSDSRefFrame::parse(val));
            }
            "ALT_COV_TYPE" => {
                obj.alt_cov_type = Some(val.to_string());
            }
            "ALT_COV_REF_FRAME" => {
                obj.alt_cov_ref_frame = Some(CCSDSRefFrame::parse(val));
            }
            "GRAVITY_MODEL" => {
                obj.gravity_model = Some(val.to_string());
            }
            "ATMOSPHERIC_MODEL" => {
                obj.atmospheric_model = Some(val.to_string());
            }
            "N_BODY_PERTURBATIONS" => {
                obj.n_body_perturbations = Some(val.to_string());
            }
            "SOLAR_RAD_PRESSURE" => {
                obj.solar_rad_pressure = Some(val.to_string());
            }
            "EARTH_TIDES" => {
                obj.earth_tides = Some(val.to_string());
            }
            "INTRACK_THRUST" => {
                obj.intrack_thrust = Some(val.to_string());
            }

            // OD parameters
            "TIME_LASTOB_START" => {
                obj.time_lastob_start = Some(parse_ccsds_datetime(val, &utc)?);
                obj.has_od_params = true;
            }
            "TIME_LASTOB_END" => {
                obj.time_lastob_end = Some(parse_ccsds_datetime(val, &utc)?);
                obj.has_od_params = true;
            }
            "RECOMMENDED_OD_SPAN" => {
                obj.recommended_od_span = Some(parse_f64(val)?);
                obj.has_od_params = true;
            }
            "ACTUAL_OD_SPAN" => {
                obj.actual_od_span = Some(parse_f64(val)?);
                obj.has_od_params = true;
            }
            "OBS_AVAILABLE" => {
                obj.obs_available = Some(parse_u32(val)?);
                obj.has_od_params = true;
            }
            "OBS_USED" => {
                obj.obs_used = Some(parse_u32(val)?);
                obj.has_od_params = true;
            }
            "TRACKS_AVAILABLE" => {
                obj.tracks_available = Some(parse_u32(val)?);
                obj.has_od_params = true;
            }
            "TRACKS_USED" => {
                obj.tracks_used = Some(parse_u32(val)?);
                obj.has_od_params = true;
            }
            "RESIDUALS_ACCEPTED" => {
                obj.residuals_accepted = Some(parse_f64(val)?);
                obj.has_od_params = true;
            }
            "WEIGHTED_RMS" => {
                obj.weighted_rms = Some(parse_f64(val)?);
                obj.has_od_params = true;
            }
            "OD_EPOCH" => {
                obj.od_epoch = Some(parse_ccsds_datetime(val, &utc)?);
                obj.has_od_params = true;
            }

            // Additional parameters
            "AREA_PC" => {
                obj.area_pc = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "AREA_PC_MIN" => {
                obj.area_pc_min = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "AREA_PC_MAX" => {
                obj.area_pc_max = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "AREA_DRG" => {
                obj.area_drg = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "AREA_SRP" => {
                obj.area_srp = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "OEB_PARENT_FRAME" => {
                obj.oeb_parent_frame = Some(val.to_string());
                obj.has_add_params = true;
            }
            "OEB_PARENT_FRAME_EPOCH" => {
                obj.oeb_parent_frame_epoch = Some(parse_ccsds_datetime(val, &utc)?);
                obj.has_add_params = true;
            }
            "OEB_Q1" => {
                obj.oeb_q1 = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "OEB_Q2" => {
                obj.oeb_q2 = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "OEB_Q3" => {
                obj.oeb_q3 = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "OEB_QC" => {
                obj.oeb_qc = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "OEB_MAX" => {
                obj.oeb_max = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "OEB_INT" => {
                obj.oeb_int = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "OEB_MIN" => {
                obj.oeb_min = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "AREA_ALONG_OEB_MAX" => {
                obj.area_along_oeb_max = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "AREA_ALONG_OEB_INT" => {
                obj.area_along_oeb_int = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "AREA_ALONG_OEB_MIN" => {
                obj.area_along_oeb_min = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "RCS" => {
                obj.rcs = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "RCS_MIN" => {
                obj.rcs_min = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "RCS_MAX" => {
                obj.rcs_max = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "VM_ABSOLUTE" => {
                obj.vm_absolute = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "VM_APPARENT_MIN" => {
                obj.vm_apparent_min = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "VM_APPARENT" => {
                obj.vm_apparent = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "VM_APPARENT_MAX" => {
                obj.vm_apparent_max = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "REFLECTANCE" => {
                obj.reflectance = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "MASS" => {
                obj.mass = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "HBR" => {
                obj.hbr = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "CD_AREA_OVER_MASS" => {
                obj.cd_area_over_mass = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "CR_AREA_OVER_MASS" => {
                obj.cr_area_over_mass = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "THRUST_ACCELERATION" => {
                obj.thrust_acceleration = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "SEDR" => {
                obj.sedr = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "MIN_DV" => {
                let parts: Vec<f64> = val
                    .split_whitespace()
                    .map(|s| s.parse().unwrap_or(0.0))
                    .collect();
                if parts.len() == 3 {
                    obj.min_dv = Some([parts[0], parts[1], parts[2]]);
                }
                obj.has_add_params = true;
            }
            "MAX_DV" => {
                let parts: Vec<f64> = val
                    .split_whitespace()
                    .map(|s| s.parse().unwrap_or(0.0))
                    .collect();
                if parts.len() == 3 {
                    obj.max_dv = Some([parts[0], parts[1], parts[2]]);
                }
                obj.has_add_params = true;
            }
            "LEAD_TIME_REQD_BEFORE_TCA" => {
                obj.lead_time_reqd_before_tca = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "APOAPSIS_ALTITUDE" => {
                obj.apoapsis_altitude = Some(parse_f64(val)? * 1e3);
                obj.has_add_params = true;
            } // km → m
            "PERIAPSIS_ALTITUDE" => {
                obj.periapsis_altitude = Some(parse_f64(val)? * 1e3);
                obj.has_add_params = true;
            } // km → m
            "INCLINATION" => {
                obj.inclination = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "COV_CONFIDENCE" => {
                obj.cov_confidence = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "COV_CONFIDENCE_METHOD" => {
                obj.cov_confidence_method = Some(val.to_string());
                obj.has_add_params = true;
            }

            // State vector (km → m, km/s → m/s)
            "X" => {
                obj.x = Some(parse_f64(val)? * 1e3);
            }
            "Y" => {
                obj.y = Some(parse_f64(val)? * 1e3);
            }
            "Z" => {
                obj.z = Some(parse_f64(val)? * 1e3);
            }
            "X_DOT" => {
                obj.x_dot = Some(parse_f64(val)? * 1e3);
            }
            "Y_DOT" => {
                obj.y_dot = Some(parse_f64(val)? * 1e3);
            }
            "Z_DOT" => {
                obj.z_dot = Some(parse_f64(val)? * 1e3);
            }

            // Additional covariance metadata
            "DENSITY_FORECAST_UNCERTAINTY" => {
                obj.density_forecast_uncertainty = Some(parse_f64(val)?);
                obj.has_acm = true;
            }
            "CSCALE_FACTOR_MIN" => {
                obj.cscale_factor_min = Some(parse_f64(val)?);
                obj.has_acm = true;
            }
            "CSCALE_FACTOR" => {
                obj.cscale_factor = Some(parse_f64(val)?);
                obj.has_acm = true;
            }
            "CSCALE_FACTOR_MAX" => {
                obj.cscale_factor_max = Some(parse_f64(val)?);
                obj.has_acm = true;
            }
            "SCREENING_DATA_SOURCE" => {
                obj.screening_data_source = Some(val.to_string());
                obj.has_acm = true;
            }
            "DCP_SENSITIVITY_VECTOR_POSITION" => {
                let parts: Vec<f64> = val
                    .split_whitespace()
                    .map(|s| s.parse().unwrap_or(0.0))
                    .collect();
                if parts.len() == 3 {
                    obj.dcp_sensitivity_vector_position = Some([parts[0], parts[1], parts[2]]);
                }
                obj.has_acm = true;
            }
            "DCP_SENSITIVITY_VECTOR_VELOCITY" => {
                let parts: Vec<f64> = val
                    .split_whitespace()
                    .map(|s| s.parse().unwrap_or(0.0))
                    .collect();
                if parts.len() == 3 {
                    obj.dcp_sensitivity_vector_velocity = Some([parts[0], parts[1], parts[2]]);
                }
                obj.has_acm = true;
            }

            // CSIG3EIGVEC3 (stored as raw string)
            "CSIG3EIGVEC3" => {
                obj.csig3eigvec3 = Some(val.to_string());
            }

            // User-defined
            k if k.starts_with("USER_DEFINED_") => {
                let ud_key = k.strip_prefix("USER_DEFINED_").unwrap_or(k);
                user_defined.insert(ud_key.to_string(), val.to_string());
            }

            // Covariance fields — route to RTN or XYZ based on context
            k => {
                // Check if this is an XYZ-specific key (CX_X, CY_X, etc.)
                if let Some((_row, _col)) = xyz_cov_index(k) {
                    let v = parse_f64(val)?;
                    obj.xyz_cov_values.push(v);
                    obj.in_xyz_cov = true; // Switch to XYZ context
                } else if let Some((_row, _col)) = rtn_cov_index(k) {
                    let v = parse_f64(val)?;
                    // If this is a core RTN-only key (CR_R, CT_R, etc.), reset XYZ context
                    if k == "CR_R" {
                        obj.in_xyz_cov = false;
                    }
                    // Shared keys (CDRG_DRG, CSRP_DRG, CSRP_SRP, CTHR_DRG, CTHR_SRP, CTHR_THR)
                    // are routed based on which covariance context we're currently in
                    if obj.in_xyz_cov {
                        obj.xyz_cov_values.push(v);
                    } else {
                        obj.rtn_cov_values.push(v);
                    }
                }
                // Unknown keys are silently ignored
            }
        }
    }

    // Comments trailing the final keyword introduce no further section, so
    // they stay with the last one seen.
    flush_comments!(last_target, current_object);

    // Build the CDM struct from collected fields
    let build_object = |obj: ObjectBuilder, label: &str| -> Result<CDMObject, BraheError> {
        let obj_label = obj.object.clone().unwrap_or_else(|| label.to_string());

        let state_vector = CDMStateVector {
            position: [
                obj.x
                    .ok_or_else(|| ccsds_missing_field("CDM", &format!("{} X", obj_label)))?,
                obj.y
                    .ok_or_else(|| ccsds_missing_field("CDM", &format!("{} Y", obj_label)))?,
                obj.z
                    .ok_or_else(|| ccsds_missing_field("CDM", &format!("{} Z", obj_label)))?,
            ],
            velocity: [
                obj.x_dot
                    .ok_or_else(|| ccsds_missing_field("CDM", &format!("{} X_DOT", obj_label)))?,
                obj.y_dot
                    .ok_or_else(|| ccsds_missing_field("CDM", &format!("{} Y_DOT", obj_label)))?,
                obj.z_dot
                    .ok_or_else(|| ccsds_missing_field("CDM", &format!("{} Z_DOT", obj_label)))?,
            ],
            comments: obj.sv_comments,
        };

        // Build RTN covariance
        let rtn_covariance = if obj.rtn_cov_values.is_empty() {
            return Err(ccsds_missing_field(
                "CDM",
                &format!("{} RTN covariance", obj_label),
            ));
        } else {
            let (matrix, dim) = covariance9x9_from_lower_triangular(&obj.rtn_cov_values)?;
            CDMRTNCovariance {
                matrix,
                dimension: dim,
                comments: obj.rtn_cov_comments,
            }
        };

        // Build XYZ covariance (optional)
        let xyz_covariance = if obj.xyz_cov_values.is_empty() {
            None
        } else {
            let (matrix, dim) = covariance9x9_from_lower_triangular(&obj.xyz_cov_values)?;
            Some(CDMXYZCovariance {
                matrix,
                dimension: dim,
                comments: obj.xyz_cov_comments,
            })
        };

        let od_parameters = if obj.has_od_params {
            Some(CDMODParameters {
                time_lastob_start: obj.time_lastob_start,
                time_lastob_end: obj.time_lastob_end,
                recommended_od_span: obj.recommended_od_span,
                actual_od_span: obj.actual_od_span,
                obs_available: obj.obs_available,
                obs_used: obj.obs_used,
                tracks_available: obj.tracks_available,
                tracks_used: obj.tracks_used,
                residuals_accepted: obj.residuals_accepted,
                weighted_rms: obj.weighted_rms,
                od_epoch: obj.od_epoch,
                comments: obj.od_comments,
            })
        } else {
            None
        };

        let additional_parameters = if obj.has_add_params {
            Some(CDMAdditionalParameters {
                area_pc: obj.area_pc,
                area_pc_min: obj.area_pc_min,
                area_pc_max: obj.area_pc_max,
                area_drg: obj.area_drg,
                area_srp: obj.area_srp,
                oeb_parent_frame: obj.oeb_parent_frame,
                oeb_parent_frame_epoch: obj.oeb_parent_frame_epoch,
                oeb_q1: obj.oeb_q1,
                oeb_q2: obj.oeb_q2,
                oeb_q3: obj.oeb_q3,
                oeb_qc: obj.oeb_qc,
                oeb_max: obj.oeb_max,
                oeb_int: obj.oeb_int,
                oeb_min: obj.oeb_min,
                area_along_oeb_max: obj.area_along_oeb_max,
                area_along_oeb_int: obj.area_along_oeb_int,
                area_along_oeb_min: obj.area_along_oeb_min,
                rcs: obj.rcs,
                rcs_min: obj.rcs_min,
                rcs_max: obj.rcs_max,
                vm_absolute: obj.vm_absolute,
                vm_apparent_min: obj.vm_apparent_min,
                vm_apparent: obj.vm_apparent,
                vm_apparent_max: obj.vm_apparent_max,
                reflectance: obj.reflectance,
                mass: obj.mass,
                hbr: obj.hbr,
                cd_area_over_mass: obj.cd_area_over_mass,
                cr_area_over_mass: obj.cr_area_over_mass,
                thrust_acceleration: obj.thrust_acceleration,
                sedr: obj.sedr,
                min_dv: obj.min_dv,
                max_dv: obj.max_dv,
                lead_time_reqd_before_tca: obj.lead_time_reqd_before_tca,
                apoapsis_altitude: obj.apoapsis_altitude,
                periapsis_altitude: obj.periapsis_altitude,
                inclination: obj.inclination,
                cov_confidence: obj.cov_confidence,
                cov_confidence_method: obj.cov_confidence_method,
                comments: obj.add_comments,
            })
        } else {
            None
        };

        let additional_covariance_metadata = if obj.has_acm {
            Some(CDMAdditionalCovarianceMetadata {
                density_forecast_uncertainty: obj.density_forecast_uncertainty,
                cscale_factor_min: obj.cscale_factor_min,
                cscale_factor: obj.cscale_factor,
                cscale_factor_max: obj.cscale_factor_max,
                screening_data_source: obj.screening_data_source,
                dcp_sensitivity_vector_position: obj.dcp_sensitivity_vector_position,
                dcp_sensitivity_vector_velocity: obj.dcp_sensitivity_vector_velocity,
                comments: obj.acm_comments,
            })
        } else {
            None
        };

        let metadata = CDMObjectMetadata {
            object: obj.object.unwrap_or_else(|| label.to_string()),
            object_designator: obj.object_designator.ok_or_else(|| {
                ccsds_missing_field("CDM", &format!("{} OBJECT_DESIGNATOR", obj_label))
            })?,
            catalog_name: obj.catalog_name.ok_or_else(|| {
                ccsds_missing_field("CDM", &format!("{} CATALOG_NAME", obj_label))
            })?,
            object_name: obj
                .object_name
                .ok_or_else(|| ccsds_missing_field("CDM", &format!("{} OBJECT_NAME", obj_label)))?,
            international_designator: obj.international_designator.ok_or_else(|| {
                ccsds_missing_field("CDM", &format!("{} INTERNATIONAL_DESIGNATOR", obj_label))
            })?,
            object_type: obj.object_type,
            ops_status: obj.ops_status,
            operator_contact_position: obj.operator_contact_position,
            operator_organization: obj.operator_organization,
            operator_phone: obj.operator_phone,
            operator_email: obj.operator_email,
            ephemeris_name: obj.ephemeris_name.ok_or_else(|| {
                ccsds_missing_field("CDM", &format!("{} EPHEMERIS_NAME", obj_label))
            })?,
            odm_msg_link: obj.odm_msg_link,
            adm_msg_link: obj.adm_msg_link,
            obs_before_next_message: obj.obs_before_next_message,
            covariance_method: obj.covariance_method.ok_or_else(|| {
                ccsds_missing_field("CDM", &format!("{} COVARIANCE_METHOD", obj_label))
            })?,
            covariance_source: obj.covariance_source,
            maneuverable: obj.maneuverable.ok_or_else(|| {
                ccsds_missing_field("CDM", &format!("{} MANEUVERABLE", obj_label))
            })?,
            orbit_center: obj.orbit_center,
            ref_frame: obj
                .ref_frame
                .ok_or_else(|| ccsds_missing_field("CDM", &format!("{} REF_FRAME", obj_label)))?,
            alt_cov_type: obj.alt_cov_type,
            alt_cov_ref_frame: obj.alt_cov_ref_frame,
            gravity_model: obj.gravity_model,
            atmospheric_model: obj.atmospheric_model,
            n_body_perturbations: obj.n_body_perturbations,
            solar_rad_pressure: obj.solar_rad_pressure,
            earth_tides: obj.earth_tides,
            intrack_thrust: obj.intrack_thrust,
            comments: obj.metadata_comments,
        };

        Ok(CDMObject {
            metadata,
            data: CDMObjectData {
                od_parameters,
                additional_parameters,
                state_vector,
                rtn_covariance,
                xyz_covariance,
                additional_covariance_metadata,
                csig3eigvec3: obj.csig3eigvec3,
                comments: obj.data_comments,
            },
        })
    };

    // Validate mandatory header/relative metadata fields
    let format_version =
        format_version.ok_or_else(|| ccsds_missing_field("CDM", "CCSDS_CDM_VERS"))?;
    let creation_date = creation_date.ok_or_else(|| ccsds_missing_field("CDM", "CREATION_DATE"))?;
    let originator = originator.ok_or_else(|| ccsds_missing_field("CDM", "ORIGINATOR"))?;
    let message_id = message_id.unwrap_or_default();
    let tca = tca.ok_or_else(|| ccsds_missing_field("CDM", "TCA"))?;
    let miss_distance = miss_distance.ok_or_else(|| ccsds_missing_field("CDM", "MISS_DISTANCE"))?;

    let object1 = build_object(obj1, "OBJECT1")?;
    let object2 = build_object(obj2, "OBJECT2")?;

    let user_defined = if user_defined.is_empty() {
        None
    } else {
        Some(CCSDSUserDefined {
            parameters: user_defined,
        })
    };

    Ok(CDM {
        header: CDMHeader {
            format_version,
            classification,
            creation_date,
            originator,
            message_for,
            message_id,
            comments: header_comments,
        },
        relative_metadata: CDMRelativeMetadata {
            conjunction_id,
            tca,
            miss_distance,
            mahalanobis_distance,
            relative_speed,
            relative_position_r: rel_pos_r,
            relative_position_t: rel_pos_t,
            relative_position_n: rel_pos_n,
            relative_velocity_r: rel_vel_r,
            relative_velocity_t: rel_vel_t,
            relative_velocity_n: rel_vel_n,
            approach_angle,
            start_screen_period,
            stop_screen_period,
            screen_type,
            screen_volume_frame,
            screen_volume_shape,
            screen_volume_radius,
            screen_volume_x,
            screen_volume_y,
            screen_volume_z,
            screen_entry_time,
            screen_exit_time,
            screen_pc_threshold,
            collision_percentile,
            collision_probability,
            collision_probability_method,
            collision_max_probability,
            collision_max_pc_method,
            sefi_collision_probability,
            sefi_collision_probability_method,
            sefi_fragmentation_model,
            previous_message_id,
            previous_message_epoch,
            next_message_epoch,
            comments: rel_comments,
        },
        object1,
        object2,
        user_defined,
    })
}

/// Write a CDM message to KVN format.
///
/// Output follows CCSDS 508.0-P-1.1 field ordering with column-aligned
/// key=value formatting.
pub fn write_cdm(cdm: &crate::ccsds::cdm::CDM) -> Result<String, BraheError> {
    use crate::ccsds::cdm::*;
    use crate::ccsds::common::covariance9x9_to_lower_triangular;

    let mut out = String::new();

    // Formatting helper: write a key=value pair with consistent alignment
    let kw = |out: &mut String, key: &str, val: &str| {
        out.push_str(&format!("{:<34}= {}\n", key, val));
    };
    let kw_units = |out: &mut String, key: &str, val: &str, units: &str| {
        out.push_str(&format!("{:<34}= {:<40} [{}]\n", key, val, units));
    };
    // CCSDS 508.0-B-1 subsection 6.2.3.4: all CDM time tags are UTC.
    let utc = |e: &crate::time::Epoch| format_ccsds_datetime_in(e, &CCSDSTimeSystem::UTC);

    // Header
    kw(
        &mut out,
        "CCSDS_CDM_VERS",
        &format!("{:.1}", cdm.header.format_version),
    );
    for comment in &cdm.header.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }
    if let Some(ref class) = cdm.header.classification {
        kw(&mut out, "CLASSIFICATION", class);
    }
    kw(&mut out, "CREATION_DATE", &utc(&cdm.header.creation_date));
    kw(&mut out, "ORIGINATOR", &cdm.header.originator);
    if let Some(ref mf) = cdm.header.message_for {
        kw(&mut out, "MESSAGE_FOR", mf);
    }
    kw(&mut out, "MESSAGE_ID", &cdm.header.message_id);

    // Relative metadata
    let rm = &cdm.relative_metadata;
    for comment in &rm.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }
    if let Some(ref cid) = rm.conjunction_id {
        kw(&mut out, "CONJUNCTION_ID", cid);
    }
    kw(&mut out, "TCA", &utc(&rm.tca));
    kw_units(
        &mut out,
        "MISS_DISTANCE",
        &format!("{}", rm.miss_distance),
        "m",
    );
    if let Some(v) = rm.mahalanobis_distance {
        kw(&mut out, "MAHALANOBIS_DISTANCE", &format!("{}", v));
    }
    if let Some(v) = rm.relative_speed {
        kw_units(&mut out, "RELATIVE_SPEED", &format!("{}", v), "m/s");
    }
    if let Some(v) = rm.relative_position_r {
        kw_units(&mut out, "RELATIVE_POSITION_R", &format!("{}", v), "m");
    }
    if let Some(v) = rm.relative_position_t {
        kw_units(&mut out, "RELATIVE_POSITION_T", &format!("{}", v), "m");
    }
    if let Some(v) = rm.relative_position_n {
        kw_units(&mut out, "RELATIVE_POSITION_N", &format!("{}", v), "m");
    }
    if let Some(v) = rm.relative_velocity_r {
        kw_units(&mut out, "RELATIVE_VELOCITY_R", &format!("{}", v), "m/s");
    }
    if let Some(v) = rm.relative_velocity_t {
        kw_units(&mut out, "RELATIVE_VELOCITY_T", &format!("{}", v), "m/s");
    }
    if let Some(v) = rm.relative_velocity_n {
        kw_units(&mut out, "RELATIVE_VELOCITY_N", &format!("{}", v), "m/s");
    }
    if let Some(v) = rm.approach_angle {
        kw_units(&mut out, "APPROACH_ANGLE", &format!("{}", v), "deg");
    }
    if let Some(ref e) = rm.start_screen_period {
        kw(&mut out, "START_SCREEN_PERIOD", &utc(e));
    }
    if let Some(ref e) = rm.stop_screen_period {
        kw(&mut out, "STOP_SCREEN_PERIOD", &utc(e));
    }
    if let Some(ref s) = rm.screen_type {
        kw(&mut out, "SCREEN_TYPE", s);
    }
    if let Some(ref f) = rm.screen_volume_frame {
        kw(&mut out, "SCREEN_VOLUME_FRAME", &format!("{}", f));
    }
    if let Some(ref s) = rm.screen_volume_shape {
        kw(&mut out, "SCREEN_VOLUME_SHAPE", s);
    }
    if let Some(v) = rm.screen_volume_radius {
        kw_units(&mut out, "SCREEN_VOLUME_RADIUS", &format!("{}", v), "m");
    }
    if let Some(v) = rm.screen_volume_x {
        kw_units(&mut out, "SCREEN_VOLUME_X", &format!("{}", v), "m");
    }
    if let Some(v) = rm.screen_volume_y {
        kw_units(&mut out, "SCREEN_VOLUME_Y", &format!("{}", v), "m");
    }
    if let Some(v) = rm.screen_volume_z {
        kw_units(&mut out, "SCREEN_VOLUME_Z", &format!("{}", v), "m");
    }
    if let Some(ref e) = rm.screen_entry_time {
        kw(&mut out, "SCREEN_ENTRY_TIME", &utc(e));
    }
    if let Some(ref e) = rm.screen_exit_time {
        kw(&mut out, "SCREEN_EXIT_TIME", &utc(e));
    }
    if let Some(v) = rm.screen_pc_threshold {
        kw(&mut out, "SCREEN_PC_THRESHOLD", &format!("{:E}", v));
    }
    if let Some(ref cp) = rm.collision_percentile {
        let s: Vec<String> = cp.iter().map(|v| v.to_string()).collect();
        kw(&mut out, "COLLISION_PERCENTILE", &s.join(" "));
    }
    if let Some(v) = rm.collision_probability {
        kw(&mut out, "COLLISION_PROBABILITY", &format!("{:E}", v));
    }
    if let Some(ref s) = rm.collision_probability_method {
        kw(&mut out, "COLLISION_PROBABILITY_METHOD", s);
    }
    if let Some(v) = rm.collision_max_probability {
        kw(&mut out, "COLLISION_MAX_PROBABILITY", &format!("{:E}", v));
    }
    if let Some(ref s) = rm.collision_max_pc_method {
        kw(&mut out, "COLLISION_MAX_PC_METHOD", s);
    }
    if let Some(v) = rm.sefi_collision_probability {
        kw(&mut out, "SEFI_COLLISION_PROBABILITY", &format!("{:E}", v));
    }
    if let Some(ref s) = rm.sefi_collision_probability_method {
        kw(&mut out, "SEFI_COLLISION_PROBABILITY_METHOD", s);
    }
    if let Some(ref s) = rm.sefi_fragmentation_model {
        kw(&mut out, "SEFI_FRAGMENTATION_MODEL", s);
    }
    if let Some(ref s) = rm.previous_message_id {
        kw(&mut out, "PREVIOUS_MESSAGE_ID", s);
    }
    if let Some(ref e) = rm.previous_message_epoch {
        kw(&mut out, "PREVIOUS_MESSAGE_EPOCH", &utc(e));
    }
    if let Some(ref e) = rm.next_message_epoch {
        kw(&mut out, "NEXT_MESSAGE_EPOCH", &utc(e));
    }

    // Write object sections
    let write_object = |out: &mut String, obj: &CDMObject| {
        let m = &obj.metadata;
        let d = &obj.data;

        // Metadata
        for comment in &m.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        kw(out, "OBJECT", &m.object);
        kw(out, "OBJECT_DESIGNATOR", &m.object_designator);
        kw(out, "CATALOG_NAME", &m.catalog_name);
        kw(out, "OBJECT_NAME", &m.object_name);
        kw(out, "INTERNATIONAL_DESIGNATOR", &m.international_designator);
        if let Some(ref v) = m.object_type {
            kw(out, "OBJECT_TYPE", v);
        }
        if let Some(ref v) = m.ops_status {
            kw(out, "OPS_STATUS", v);
        }
        if let Some(ref v) = m.operator_contact_position {
            kw(out, "OPERATOR_CONTACT_POSITION", v);
        }
        if let Some(ref v) = m.operator_organization {
            kw(out, "OPERATOR_ORGANIZATION", v);
        }
        if let Some(ref v) = m.operator_phone {
            kw(out, "OPERATOR_PHONE", v);
        }
        if let Some(ref v) = m.operator_email {
            kw(out, "OPERATOR_EMAIL", v);
        }
        kw(out, "EPHEMERIS_NAME", &m.ephemeris_name);
        if let Some(ref v) = m.odm_msg_link {
            kw(out, "ODM_MSG_LINK", v);
        }
        if let Some(ref v) = m.adm_msg_link {
            kw(out, "ADM_MSG_LINK", v);
        }
        if let Some(ref v) = m.obs_before_next_message {
            kw(out, "OBS_BEFORE_NEXT_MESSAGE", v);
        }
        kw(out, "COVARIANCE_METHOD", &m.covariance_method);
        if let Some(ref v) = m.covariance_source {
            kw(out, "COVARIANCE_SOURCE", v);
        }
        kw(out, "MANEUVERABLE", &m.maneuverable);
        if let Some(ref v) = m.orbit_center {
            kw(out, "ORBIT_CENTER", v);
        }
        kw(out, "REF_FRAME", &format!("{}", m.ref_frame));
        if let Some(ref v) = m.alt_cov_type {
            kw(out, "ALT_COV_TYPE", v);
        }
        if let Some(ref v) = m.alt_cov_ref_frame {
            kw(out, "ALT_COV_REF_FRAME", &format!("{}", v));
        }
        if let Some(ref v) = m.gravity_model {
            kw(out, "GRAVITY_MODEL", v);
        }
        if let Some(ref v) = m.atmospheric_model {
            kw(out, "ATMOSPHERIC_MODEL", v);
        }
        if let Some(ref v) = m.n_body_perturbations {
            kw(out, "N_BODY_PERTURBATIONS", v);
        }
        if let Some(ref v) = m.solar_rad_pressure {
            kw(out, "SOLAR_RAD_PRESSURE", v);
        }
        if let Some(ref v) = m.earth_tides {
            kw(out, "EARTH_TIDES", v);
        }
        if let Some(ref v) = m.intrack_thrust {
            kw(out, "INTRACK_THRUST", v);
        }

        // Data-section comments lead the data section: table 3-4 lists the
        // Data COMMENT row ahead of the OD Parameters sub-block.
        for comment in &d.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }

        // OD parameters
        if let Some(ref od) = d.od_parameters {
            for comment in &od.comments {
                out.push_str(&format!("COMMENT {}\n", comment));
            }
            if let Some(ref e) = od.time_lastob_start {
                kw(out, "TIME_LASTOB_START", &utc(e));
            }
            if let Some(ref e) = od.time_lastob_end {
                kw(out, "TIME_LASTOB_END", &utc(e));
            }
            if let Some(v) = od.recommended_od_span {
                kw_units(out, "RECOMMENDED_OD_SPAN", &format!("{:.2}", v), "d");
            }
            if let Some(v) = od.actual_od_span {
                kw_units(out, "ACTUAL_OD_SPAN", &format!("{:.2}", v), "d");
            }
            if let Some(v) = od.obs_available {
                kw(out, "OBS_AVAILABLE", &format!("{}", v));
            }
            if let Some(v) = od.obs_used {
                kw(out, "OBS_USED", &format!("{}", v));
            }
            if let Some(v) = od.tracks_available {
                kw(out, "TRACKS_AVAILABLE", &format!("{}", v));
            }
            if let Some(v) = od.tracks_used {
                kw(out, "TRACKS_USED", &format!("{}", v));
            }
            if let Some(v) = od.residuals_accepted {
                kw_units(out, "RESIDUALS_ACCEPTED", &format!("{}", v), "%");
            }
            if let Some(v) = od.weighted_rms {
                kw(out, "WEIGHTED_RMS", &format!("{}", v));
            }
            if let Some(ref e) = od.od_epoch {
                kw(out, "OD_EPOCH", &utc(e));
            }
        }

        // Additional parameters
        if let Some(ref ap) = d.additional_parameters {
            for comment in &ap.comments {
                out.push_str(&format!("COMMENT {}\n", comment));
            }
            if let Some(v) = ap.area_pc {
                kw_units(out, "AREA_PC", &format!("{}", v), "m**2");
            }
            if let Some(v) = ap.area_pc_min {
                kw_units(out, "AREA_PC_MIN", &format!("{}", v), "m**2");
            }
            if let Some(v) = ap.area_pc_max {
                kw_units(out, "AREA_PC_MAX", &format!("{}", v), "m**2");
            }
            if let Some(v) = ap.area_drg {
                kw_units(out, "AREA_DRG", &format!("{}", v), "m**2");
            }
            if let Some(v) = ap.area_srp {
                kw_units(out, "AREA_SRP", &format!("{}", v), "m**2");
            }
            if let Some(ref v) = ap.oeb_parent_frame {
                kw(out, "OEB_PARENT_FRAME", v);
            }
            if let Some(ref e) = ap.oeb_parent_frame_epoch {
                kw(out, "OEB_PARENT_FRAME_EPOCH", &utc(e));
            }
            if let Some(v) = ap.oeb_q1 {
                kw(out, "OEB_Q1", &format!("{}", v));
            }
            if let Some(v) = ap.oeb_q2 {
                kw(out, "OEB_Q2", &format!("{}", v));
            }
            if let Some(v) = ap.oeb_q3 {
                kw(out, "OEB_Q3", &format!("{}", v));
            }
            if let Some(v) = ap.oeb_qc {
                kw(out, "OEB_QC", &format!("{}", v));
            }
            if let Some(v) = ap.oeb_max {
                kw_units(out, "OEB_MAX", &format!("{}", v), "m");
            }
            if let Some(v) = ap.oeb_int {
                kw_units(out, "OEB_INT", &format!("{}", v), "m");
            }
            if let Some(v) = ap.oeb_min {
                kw_units(out, "OEB_MIN", &format!("{}", v), "m");
            }
            if let Some(v) = ap.area_along_oeb_max {
                kw_units(out, "AREA_ALONG_OEB_MAX", &format!("{}", v), "m**2");
            }
            if let Some(v) = ap.area_along_oeb_int {
                kw_units(out, "AREA_ALONG_OEB_INT", &format!("{}", v), "m**2");
            }
            if let Some(v) = ap.area_along_oeb_min {
                kw_units(out, "AREA_ALONG_OEB_MIN", &format!("{}", v), "m**2");
            }
            if let Some(v) = ap.rcs {
                kw_units(out, "RCS", &format!("{}", v), "m**2");
            }
            if let Some(v) = ap.rcs_min {
                kw_units(out, "RCS_MIN", &format!("{}", v), "m**2");
            }
            if let Some(v) = ap.rcs_max {
                kw_units(out, "RCS_MAX", &format!("{}", v), "m**2");
            }
            if let Some(v) = ap.vm_absolute {
                kw(out, "VM_ABSOLUTE", &format!("{}", v));
            }
            if let Some(v) = ap.vm_apparent_min {
                kw(out, "VM_APPARENT_MIN", &format!("{}", v));
            }
            if let Some(v) = ap.vm_apparent {
                kw(out, "VM_APPARENT", &format!("{}", v));
            }
            if let Some(v) = ap.vm_apparent_max {
                kw(out, "VM_APPARENT_MAX", &format!("{}", v));
            }
            if let Some(v) = ap.reflectance {
                kw(out, "REFLECTANCE", &format!("{}", v));
            }
            if let Some(v) = ap.mass {
                kw_units(out, "MASS", &format!("{}", v), "kg");
            }
            if let Some(v) = ap.hbr {
                kw_units(out, "HBR", &format!("{}", v), "m");
            }
            if let Some(v) = ap.cd_area_over_mass {
                kw_units(out, "CD_AREA_OVER_MASS", &format!("{}", v), "m**2/kg");
            }
            if let Some(v) = ap.cr_area_over_mass {
                kw_units(out, "CR_AREA_OVER_MASS", &format!("{}", v), "m**2/kg");
            }
            if let Some(v) = ap.thrust_acceleration {
                kw_units(out, "THRUST_ACCELERATION", &format!("{}", v), "m/s**2");
            }
            if let Some(v) = ap.sedr {
                kw_units(out, "SEDR", &format!("{:E}", v), "W/kg");
            }
            // Space-delimited RTN triples, per CCSDS 508.0-B-1 subsection
            // 6.3.2.x; the parser already reads them.
            if let Some(ref v) = ap.min_dv {
                kw_units(out, "MIN_DV", &format!("{} {} {}", v[0], v[1], v[2]), "m/s");
            }
            if let Some(ref v) = ap.max_dv {
                kw_units(out, "MAX_DV", &format!("{} {} {}", v[0], v[1], v[2]), "m/s");
            }
            if let Some(v) = ap.lead_time_reqd_before_tca {
                kw_units(out, "LEAD_TIME_REQD_BEFORE_TCA", &format!("{}", v), "h");
            }
            if let Some(v) = ap.apoapsis_altitude {
                kw_units(out, "APOAPSIS_ALTITUDE", &format!("{}", v / 1e3), "km");
            }
            if let Some(v) = ap.periapsis_altitude {
                kw_units(out, "PERIAPSIS_ALTITUDE", &format!("{}", v / 1e3), "km");
            }
            if let Some(v) = ap.inclination {
                kw_units(out, "INCLINATION", &format!("{}", v), "deg");
            }
            if let Some(v) = ap.cov_confidence {
                kw(out, "COV_CONFIDENCE", &format!("{}", v));
            }
            if let Some(ref v) = ap.cov_confidence_method {
                kw(out, "COV_CONFIDENCE_METHOD", v);
            }
        }

        // State vector (m → km, m/s → km/s)
        for comment in &d.state_vector.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        kw_units(
            out,
            "X",
            &format!("{:.6}", d.state_vector.position[0] / 1e3),
            "km",
        );
        kw_units(
            out,
            "Y",
            &format!("{:.6}", d.state_vector.position[1] / 1e3),
            "km",
        );
        kw_units(
            out,
            "Z",
            &format!("{:.6}", d.state_vector.position[2] / 1e3),
            "km",
        );
        kw_units(
            out,
            "X_DOT",
            &format!("{:.9}", d.state_vector.velocity[0] / 1e3),
            "km/s",
        );
        kw_units(
            out,
            "Y_DOT",
            &format!("{:.9}", d.state_vector.velocity[1] / 1e3),
            "km/s",
        );
        kw_units(
            out,
            "Z_DOT",
            &format!("{:.9}", d.state_vector.velocity[2] / 1e3),
            "km/s",
        );

        // RTN covariance (already in SI units, no conversion)
        let rtn_names_6x6: &[&str] = &[
            "CR_R",
            "CT_R",
            "CT_T",
            "CN_R",
            "CN_T",
            "CN_N",
            "CRDOT_R",
            "CRDOT_T",
            "CRDOT_N",
            "CRDOT_RDOT",
            "CTDOT_R",
            "CTDOT_T",
            "CTDOT_N",
            "CTDOT_RDOT",
            "CTDOT_TDOT",
            "CNDOT_R",
            "CNDOT_T",
            "CNDOT_N",
            "CNDOT_RDOT",
            "CNDOT_TDOT",
            "CNDOT_NDOT",
        ];
        let rtn_names_drg: &[&str] = &[
            "CDRG_R",
            "CDRG_T",
            "CDRG_N",
            "CDRG_RDOT",
            "CDRG_TDOT",
            "CDRG_NDOT",
            "CDRG_DRG",
        ];
        let rtn_names_srp: &[&str] = &[
            "CSRP_R",
            "CSRP_T",
            "CSRP_N",
            "CSRP_RDOT",
            "CSRP_TDOT",
            "CSRP_NDOT",
            "CSRP_DRG",
            "CSRP_SRP",
        ];
        let rtn_names_thr: &[&str] = &[
            "CTHR_R",
            "CTHR_T",
            "CTHR_N",
            "CTHR_RDOT",
            "CTHR_TDOT",
            "CTHR_NDOT",
            "CTHR_DRG",
            "CTHR_SRP",
            "CTHR_THR",
        ];

        // Covariance units by position
        #[allow(clippy::manual_range_contains)]
        let cov_unit = |row: usize, col: usize| -> &str {
            match (row, col) {
                (r, c) if r < 3 && c < 3 => "m**2",
                (r, c) if (r < 3 && c >= 3 && c < 6) || (r >= 3 && r < 6 && c < 3) => "m**2/s",
                (r, c) if r >= 3 && r < 6 && c >= 3 && c < 6 => "m**2/s**2",
                (6, c) if c < 3 => "m**3/kg",
                (6, c) if c >= 3 && c < 6 => "m**3/(kg*s)",
                (6, 6) => "m**4/kg**2",
                (7, c) if c < 3 => "m**3/kg",
                (7, c) if c >= 3 && c < 6 => "m**3/(kg*s)",
                (7, 6) | (7, 7) => "m**4/kg**2",
                (8, c) if c < 3 => "m**2/s**2",
                (8, c) if c >= 3 && c < 6 => "m**2/s**3",
                (8, 6) | (8, 7) => "m**3/(kg*s**2)",
                (8, 8) => "m**2/s**4",
                _ => "",
            }
        };

        for comment in &d.rtn_covariance.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        let rtn_vals =
            covariance9x9_to_lower_triangular(&d.rtn_covariance.matrix, d.rtn_covariance.dimension);
        let dim = d.rtn_covariance.dimension.size();
        let mut idx = 0;
        for row in 0..dim {
            for col in 0..=row {
                let name = match row {
                    0..=5 => rtn_names_6x6[idx],
                    6 => rtn_names_drg[col],
                    7 => rtn_names_srp[col],
                    8 => rtn_names_thr[col],
                    _ => unreachable!(),
                };
                let unit = cov_unit(row, col);
                kw_units(out, name, &format!("{:E}", rtn_vals[idx]), unit);
                idx += 1;
            }
        }

        // XYZ covariance (if present)
        if let Some(ref xyz) = d.xyz_covariance {
            let xyz_names_6x6: &[&str] = &[
                "CX_X",
                "CY_X",
                "CY_Y",
                "CZ_X",
                "CZ_Y",
                "CZ_Z",
                "CXDOT_X",
                "CXDOT_Y",
                "CXDOT_Z",
                "CXDOT_XDOT",
                "CYDOT_X",
                "CYDOT_Y",
                "CYDOT_Z",
                "CYDOT_XDOT",
                "CYDOT_YDOT",
                "CZDOT_X",
                "CZDOT_Y",
                "CZDOT_Z",
                "CZDOT_XDOT",
                "CZDOT_YDOT",
                "CZDOT_ZDOT",
            ];
            let xyz_names_drg: &[&str] = &[
                "CDRG_X",
                "CDRG_Y",
                "CDRG_Z",
                "CDRG_XDOT",
                "CDRG_YDOT",
                "CDRG_ZDOT",
                "CDRG_DRG",
            ];
            let xyz_names_srp: &[&str] = &[
                "CSRP_X",
                "CSRP_Y",
                "CSRP_Z",
                "CSRP_XDOT",
                "CSRP_YDOT",
                "CSRP_ZDOT",
                "CSRP_DRG",
                "CSRP_SRP",
            ];
            let xyz_names_thr: &[&str] = &[
                "CTHR_X",
                "CTHR_Y",
                "CTHR_Z",
                "CTHR_XDOT",
                "CTHR_YDOT",
                "CTHR_ZDOT",
                "CTHR_DRG",
                "CTHR_SRP",
                "CTHR_THR",
            ];

            for comment in &xyz.comments {
                out.push_str(&format!("COMMENT {}\n", comment));
            }
            let xyz_vals = covariance9x9_to_lower_triangular(&xyz.matrix, xyz.dimension);
            let xdim = xyz.dimension.size();
            let mut xidx = 0;
            for row in 0..xdim {
                for col in 0..=row {
                    let name = match row {
                        0..=5 => xyz_names_6x6[xidx],
                        6 => xyz_names_drg[col],
                        7 => xyz_names_srp[col],
                        8 => xyz_names_thr[col],
                        _ => unreachable!(),
                    };
                    let unit = cov_unit(row, col);
                    kw_units(out, name, &format!("{:E}", xyz_vals[xidx]), unit);
                    xidx += 1;
                }
            }
        }

        // CSIG3EIGVEC3
        if let Some(ref s) = d.csig3eigvec3 {
            kw(out, "CSIG3EIGVEC3", s);
        }

        // Additional covariance metadata
        if let Some(ref acm) = d.additional_covariance_metadata {
            for comment in &acm.comments {
                out.push_str(&format!("COMMENT {}\n", comment));
            }
            if let Some(v) = acm.density_forecast_uncertainty {
                kw(out, "DENSITY_FORECAST_UNCERTAINTY", &format!("{}", v));
            }
            if let Some(v) = acm.cscale_factor_min {
                kw(out, "CSCALE_FACTOR_MIN", &format!("{}", v));
            }
            if let Some(v) = acm.cscale_factor {
                kw(out, "CSCALE_FACTOR", &format!("{}", v));
            }
            if let Some(v) = acm.cscale_factor_max {
                kw(out, "CSCALE_FACTOR_MAX", &format!("{}", v));
            }
            if let Some(ref s) = acm.screening_data_source {
                kw(out, "SCREENING_DATA_SOURCE", s);
            }
            if let Some(ref v) = acm.dcp_sensitivity_vector_position {
                kw(
                    out,
                    "DCP_SENSITIVITY_VECTOR_POSITION",
                    &format!("{} {} {}", v[0], v[1], v[2]),
                );
            }
            if let Some(ref v) = acm.dcp_sensitivity_vector_velocity {
                kw(
                    out,
                    "DCP_SENSITIVITY_VECTOR_VELOCITY",
                    &format!("{} {} {}", v[0], v[1], v[2]),
                );
            }
        }
    };

    write_object(&mut out, &cdm.object1);
    write_object(&mut out, &cdm.object2);

    // User-defined parameters
    if let Some(ref ud) = cdm.user_defined {
        for (k, v) in &ud.parameters {
            kw(&mut out, &format!("USER_DEFINED_{}", k), v);
        }
    }

    Ok(out)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::ccsds::common::CDMCovarianceDimension;
    use crate::ccsds::kvn::parse_cdm;

    use serial_test::parallel;
    #[test]
    #[serial_test::parallel]
    fn test_parse_cdm_attributes_comments_to_the_block_they_introduce() {
        let content = std::fs::read_to_string("test_assets/ccsds/cdm/CDMExample2.txt").unwrap();
        let cdm = parse_cdm(&content).unwrap();

        // The section-introducing comments of the CCSDS example land in the
        // section each one names, not in the one that precedes it.
        assert!(cdm.header.comments.is_empty());
        assert_eq!(
            cdm.relative_metadata.comments,
            vec!["Relative Metadata/Data"]
        );
        assert_eq!(cdm.object1.metadata.comments, vec!["Object1 Metadata"]);
        assert_eq!(cdm.object2.metadata.comments, vec!["Object2 Metadata"]);
        assert_eq!(
            cdm.object1.data.state_vector.comments,
            vec!["Object1 State Vector"]
        );
        assert_eq!(
            cdm.object1.data.rtn_covariance.comments,
            vec!["Object1 Covariance in the RTN Coordinate Frame"]
        );
        assert_eq!(
            cdm.object1
                .data
                .additional_parameters
                .as_ref()
                .unwrap()
                .comments,
            vec![
                "Object1 Additional Parameters",
                "Apogee Altitude=779 km",
                "Perigee Altitude=765 km",
                "Inclination=86.4 deg",
            ]
        );
        // KVN puts nothing between the Data comment and the OD Parameters
        // comment, so the run resolves to the innermost block the next keyword
        // opens.
        assert_eq!(
            cdm.object1.data.od_parameters.as_ref().unwrap().comments,
            vec!["Object1 Data", "Object1 OD Parameters"]
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_cdm_comment_buckets_survive_a_kvn_round_trip() {
        use crate::ccsds::common::CCSDSFormat;

        for name in [
            "CDMExample1.txt",
            "CDMExample2.txt",
            "CDMExample3.txt",
            "CDMExample4.txt",
            "CDMExample_issue_940.txt",
        ] {
            let content =
                std::fs::read_to_string(format!("test_assets/ccsds/cdm/{}", name)).unwrap();
            let written = parse_cdm(&content)
                .unwrap()
                .to_string(CCSDSFormat::KVN)
                .unwrap();
            let rewritten = parse_cdm(&written)
                .unwrap()
                .to_string(CCSDSFormat::KVN)
                .unwrap();
            assert_eq!(rewritten, written, "{} does not round-trip", name);
        }
    }

    // ------------------------------------------------------------------
    // CDM writer: round-trip tests with test assets
    // ------------------------------------------------------------------

    #[test]
    #[parallel]
    fn test_cdm_kvn_round_trip_example2() {
        // CDMExample2 has many optional fields: MESSAGE_FOR, OBJECT_TYPE,
        // OPERATOR_*, ORBIT_CENTER, OD params, additional params, 8x8 covariance
        let content = std::fs::read_to_string("test_assets/ccsds/cdm/CDMExample2.txt").unwrap();
        let cdm = crate::ccsds::kvn::parse_cdm(&content).unwrap();

        let written = write_cdm(&cdm).unwrap();

        // Verify header optional fields
        assert!(written.contains("MESSAGE_FOR"));
        assert!(written.contains("SATELLITE A"));

        // Verify relative metadata screening volume fields
        assert!(written.contains("START_SCREEN_PERIOD"));
        assert!(written.contains("STOP_SCREEN_PERIOD"));
        assert!(written.contains("SCREEN_VOLUME_FRAME"));
        assert!(written.contains("SCREEN_VOLUME_SHAPE"));
        assert!(written.contains("SCREEN_VOLUME_X"));
        assert!(written.contains("SCREEN_VOLUME_Y"));
        assert!(written.contains("SCREEN_VOLUME_Z"));
        assert!(written.contains("SCREEN_ENTRY_TIME"));
        assert!(written.contains("SCREEN_EXIT_TIME"));
        assert!(written.contains("COLLISION_PROBABILITY"));
        assert!(written.contains("COLLISION_PROBABILITY_METHOD"));

        // Verify relative speed and positions
        assert!(written.contains("RELATIVE_SPEED"));
        assert!(written.contains("RELATIVE_POSITION_R"));
        assert!(written.contains("RELATIVE_POSITION_T"));
        assert!(written.contains("RELATIVE_POSITION_N"));
        assert!(written.contains("RELATIVE_VELOCITY_R"));
        assert!(written.contains("RELATIVE_VELOCITY_T"));
        assert!(written.contains("RELATIVE_VELOCITY_N"));

        // Verify object metadata optional fields
        assert!(written.contains("OBJECT_TYPE"));
        assert!(written.contains("OPERATOR_CONTACT_POSITION"));
        assert!(written.contains("OPERATOR_ORGANIZATION"));
        assert!(written.contains("OPERATOR_PHONE"));
        assert!(written.contains("OPERATOR_EMAIL"));
        assert!(written.contains("ORBIT_CENTER"));

        // Verify physical model fields
        assert!(written.contains("GRAVITY_MODEL"));
        assert!(written.contains("ATMOSPHERIC_MODEL"));
        assert!(written.contains("N_BODY_PERTURBATIONS"));
        assert!(written.contains("SOLAR_RAD_PRESSURE"));
        assert!(written.contains("EARTH_TIDES"));
        assert!(written.contains("INTRACK_THRUST"));

        // Verify OD parameters block
        assert!(written.contains("TIME_LASTOB_START"));
        assert!(written.contains("TIME_LASTOB_END"));
        assert!(written.contains("RECOMMENDED_OD_SPAN"));
        assert!(written.contains("ACTUAL_OD_SPAN"));
        assert!(written.contains("OBS_AVAILABLE"));
        assert!(written.contains("OBS_USED"));
        assert!(written.contains("TRACKS_AVAILABLE"));
        assert!(written.contains("TRACKS_USED"));
        assert!(written.contains("RESIDUALS_ACCEPTED"));
        assert!(written.contains("WEIGHTED_RMS"));

        // Verify additional parameters
        assert!(written.contains("AREA_PC"));
        assert!(written.contains("MASS"));
        assert!(written.contains("CD_AREA_OVER_MASS"));
        assert!(written.contains("CR_AREA_OVER_MASS"));
        assert!(written.contains("THRUST_ACCELERATION"));
        assert!(written.contains("SEDR"));

        // Verify 8x8 covariance (has DRG and SRP rows)
        assert!(written.contains("CDRG_R"));
        assert!(written.contains("CDRG_DRG"));
        assert!(written.contains("CSRP_R"));
        assert!(written.contains("CSRP_SRP"));

        // Round-trip: re-parse and verify structure
        let cdm2 = crate::ccsds::kvn::parse_cdm(&written).unwrap();
        assert_eq!(cdm2.header.originator, "JSPOC");
        assert_eq!(cdm2.header.message_for.as_deref(), Some("SATELLITE A"));
        assert!(cdm2.relative_metadata.collision_probability.is_some());
        assert!(cdm2.object1.data.od_parameters.is_some());
        assert!(cdm2.object1.data.additional_parameters.is_some());
        assert!(cdm2.object2.data.od_parameters.is_some());
        assert!(cdm2.object2.data.additional_parameters.is_some());
    }

    #[test]
    #[parallel]
    fn test_cdm_write_programmatic_all_optional_fields() {
        use crate::ccsds::cdm::*;
        use crate::ccsds::common::{CCSDSRefFrame, CCSDSUserDefined, CDMCovarianceDimension};
        use crate::time::Epoch;
        use nalgebra::SMatrix;
        use std::collections::HashMap;

        let tca = Epoch::from_datetime(2024, 1, 15, 12, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let creation =
            Epoch::from_datetime(2024, 1, 14, 8, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let screen_start =
            Epoch::from_datetime(2024, 1, 14, 0, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let screen_stop =
            Epoch::from_datetime(2024, 1, 16, 0, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let screen_entry =
            Epoch::from_datetime(2024, 1, 15, 11, 59, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let screen_exit =
            Epoch::from_datetime(2024, 1, 15, 12, 1, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let prev_epoch =
            Epoch::from_datetime(2024, 1, 13, 12, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let next_epoch =
            Epoch::from_datetime(2024, 1, 16, 12, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let od_start =
            Epoch::from_datetime(2024, 1, 10, 0, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let od_end =
            Epoch::from_datetime(2024, 1, 14, 0, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let od_epoch =
            Epoch::from_datetime(2024, 1, 13, 0, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let oeb_epoch =
            Epoch::from_datetime(2024, 1, 15, 0, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);

        // Build relative metadata with all optional fields
        let mut rm = CDMRelativeMetadata::new(tca, 500.0);
        rm.conjunction_id = Some("CONJ-2024-001".to_string());
        rm.mahalanobis_distance = Some(3.5);
        rm.relative_speed = Some(14000.0);
        rm.relative_position_r = Some(25.0);
        rm.relative_position_t = Some(-60.0);
        rm.relative_position_n = Some(490.0);
        rm.relative_velocity_r = Some(-5.0);
        rm.relative_velocity_t = Some(-13900.0);
        rm.relative_velocity_n = Some(-1200.0);
        rm.approach_angle = Some(45.0);
        rm.start_screen_period = Some(screen_start);
        rm.stop_screen_period = Some(screen_stop);
        rm.screen_type = Some("PC".to_string());
        rm.screen_volume_frame = Some(CCSDSRefFrame::RTN);
        rm.screen_volume_shape = Some("ELLIPSOID".to_string());
        rm.screen_volume_radius = Some(100.0);
        rm.screen_volume_x = Some(200.0);
        rm.screen_volume_y = Some(1000.0);
        rm.screen_volume_z = Some(1000.0);
        rm.screen_entry_time = Some(screen_entry);
        rm.screen_exit_time = Some(screen_exit);
        rm.screen_pc_threshold = Some(1e-7);
        rm.collision_percentile = Some(vec![25, 50, 75]);
        rm.collision_probability = Some(5.0e-5);
        rm.collision_probability_method = Some("FOSTER-1992".to_string());
        rm.collision_max_probability = Some(1.0e-4);
        rm.collision_max_pc_method = Some("ALFANO-2005".to_string());
        rm.sefi_collision_probability = Some(2.0e-5);
        rm.sefi_collision_probability_method = Some("SEFI-FOSTER".to_string());
        rm.sefi_fragmentation_model = Some("NASA-SBM".to_string());
        rm.previous_message_id = Some("PREV-MSG-001".to_string());
        rm.previous_message_epoch = Some(prev_epoch);
        rm.next_message_epoch = Some(next_epoch);
        rm.comments = vec!["Relative metadata comment".to_string()];

        // Build object metadata with all optional fields
        let mut meta1 = CDMObjectMetadata::new(
            "OBJECT1".to_string(),
            "12345".to_string(),
            "SATCAT".to_string(),
            "SAT-A".to_string(),
            "2020-001A".to_string(),
            "EPHEMERIS SAT-A".to_string(),
            "CALCULATED".to_string(),
            "YES".to_string(),
            CCSDSRefFrame::EME2000,
        );
        meta1.object_type = Some("PAYLOAD".to_string());
        meta1.ops_status = Some("+/- FON".to_string());
        meta1.operator_contact_position = Some("OSA".to_string());
        meta1.operator_organization = Some("EUMETSAT".to_string());
        meta1.operator_phone = Some("+49123456789".to_string());
        meta1.operator_email = Some("ops@example.com".to_string());
        meta1.odm_msg_link = Some("ccsds.org/msg/12345".to_string());
        meta1.adm_msg_link = Some("ccsds.org/adm/67890".to_string());
        meta1.obs_before_next_message = Some("YES".to_string());
        meta1.covariance_source = Some("ASW".to_string());
        meta1.orbit_center = Some("EARTH".to_string());
        meta1.alt_cov_type = Some("XYZ".to_string());
        meta1.alt_cov_ref_frame = Some(CCSDSRefFrame::ITRF2000);
        meta1.gravity_model = Some("EGM-96: 36D 36O".to_string());
        meta1.atmospheric_model = Some("JACCHIA 70 DCA".to_string());
        meta1.n_body_perturbations = Some("MOON, SUN".to_string());
        meta1.solar_rad_pressure = Some("YES".to_string());
        meta1.earth_tides = Some("NO".to_string());
        meta1.intrack_thrust = Some("NO".to_string());
        meta1.comments = vec!["Object1 metadata comment".to_string()];

        // Build OD parameters
        let od_params = CDMODParameters {
            time_lastob_start: Some(od_start),
            time_lastob_end: Some(od_end),
            recommended_od_span: Some(7.5),
            actual_od_span: Some(5.0),
            obs_available: Some(500),
            obs_used: Some(480),
            tracks_available: Some(100),
            tracks_used: Some(95),
            residuals_accepted: Some(98.5),
            weighted_rms: Some(0.95),
            od_epoch: Some(od_epoch),
            comments: vec!["OD parameters comment".to_string()],
        };

        // Build additional parameters with many optional fields
        let ap = CDMAdditionalParameters {
            area_pc: Some(5.0),
            area_pc_min: Some(3.0),
            area_pc_max: Some(7.0),
            area_drg: Some(10.0),
            area_srp: Some(12.0),
            oeb_parent_frame: Some("EME2000".to_string()),
            oeb_parent_frame_epoch: Some(oeb_epoch),
            oeb_q1: Some(0.1),
            oeb_q2: Some(0.2),
            oeb_q3: Some(0.3),
            oeb_qc: Some(0.927),
            oeb_max: Some(2.0),
            oeb_int: Some(1.5),
            oeb_min: Some(1.0),
            area_along_oeb_max: Some(4.0),
            area_along_oeb_int: Some(3.0),
            area_along_oeb_min: Some(2.0),
            rcs: Some(1.5),
            rcs_min: Some(0.5),
            rcs_max: Some(2.5),
            vm_absolute: Some(20.0),
            vm_apparent_min: Some(18.0),
            vm_apparent: Some(19.0),
            vm_apparent_max: Some(21.0),
            reflectance: Some(0.3),
            mass: Some(250.0),
            hbr: Some(1.0),
            cd_area_over_mass: Some(0.05),
            cr_area_over_mass: Some(0.01),
            thrust_acceleration: Some(0.001),
            sedr: Some(4.5e-5),
            min_dv: None,
            max_dv: None,
            lead_time_reqd_before_tca: Some(24.0),
            apoapsis_altitude: Some(800e3),
            periapsis_altitude: Some(750e3),
            inclination: Some(98.0),
            cov_confidence: Some(0.95),
            cov_confidence_method: Some("EIGENVALUE".to_string()),
            comments: vec!["Additional params comment".to_string()],
        };

        // Build state vector
        let sv1 = CDMStateVector::new(
            [2570097.065, 2244654.904, 6281497.978],
            [4418.769571, 4833.547743, -3526.774282],
        );

        // Build 6x6 RTN covariance
        let mut rtn_matrix = SMatrix::<f64, 9, 9>::zeros();
        rtn_matrix[(0, 0)] = 4.142e+01;
        rtn_matrix[(1, 0)] = -8.579e+00;
        rtn_matrix[(0, 1)] = -8.579e+00;
        rtn_matrix[(1, 1)] = 2.533e+03;
        rtn_matrix[(2, 2)] = 7.098e+01;
        rtn_matrix[(3, 3)] = 5.744e-03;
        rtn_matrix[(4, 4)] = 1.049e-05;
        rtn_matrix[(5, 5)] = 5.529e-05;

        let rtn_cov = CDMRTNCovariance {
            matrix: rtn_matrix,
            dimension: CDMCovarianceDimension::SixBySix,
            comments: vec!["RTN covariance comment".to_string()],
        };

        // Build XYZ covariance (optional)
        let mut xyz_matrix = SMatrix::<f64, 9, 9>::zeros();
        xyz_matrix[(0, 0)] = 1.0e+02;
        xyz_matrix[(1, 1)] = 2.0e+02;
        xyz_matrix[(2, 2)] = 3.0e+02;
        xyz_matrix[(3, 3)] = 1.0e-03;
        xyz_matrix[(4, 4)] = 2.0e-03;
        xyz_matrix[(5, 5)] = 3.0e-03;

        let xyz_cov = CDMXYZCovariance {
            matrix: xyz_matrix,
            dimension: CDMCovarianceDimension::SixBySix,
            comments: vec!["XYZ covariance comment".to_string()],
        };

        // Build additional covariance metadata
        let acm = CDMAdditionalCovarianceMetadata {
            density_forecast_uncertainty: Some(0.5),
            cscale_factor_min: Some(0.8),
            cscale_factor: Some(1.0),
            cscale_factor_max: Some(1.2),
            screening_data_source: Some("ASTAT".to_string()),
            dcp_sensitivity_vector_position: Some([1.0, 2.0, 3.0]),
            dcp_sensitivity_vector_velocity: Some([0.01, 0.02, 0.03]),
            comments: vec!["Additional covariance comment".to_string()],
        };

        let obj1 = CDMObject {
            metadata: meta1,
            data: CDMObjectData {
                od_parameters: Some(od_params),
                additional_parameters: Some(ap),
                state_vector: sv1,
                rtn_covariance: rtn_cov,
                xyz_covariance: Some(xyz_cov),
                additional_covariance_metadata: Some(acm),
                csig3eigvec3: Some("1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0".to_string()),
                comments: vec!["Data section comment".to_string()],
            },
        };

        // Minimal object2
        let meta2 = CDMObjectMetadata::new(
            "OBJECT2".to_string(),
            "67890".to_string(),
            "SATCAT".to_string(),
            "DEBRIS-B".to_string(),
            "1999-025AA".to_string(),
            "NONE".to_string(),
            "CALCULATED".to_string(),
            "NO".to_string(),
            CCSDSRefFrame::EME2000,
        );
        let sv2 = CDMStateVector::new(
            [2569540.800, 2245093.614, 6281599.946],
            [-2888.612500, -6007.247516, 3328.770172],
        );
        let mut rtn_matrix2 = SMatrix::<f64, 9, 9>::zeros();
        rtn_matrix2[(0, 0)] = 1.337e+03;
        rtn_matrix2[(1, 1)] = 2.492e+06;
        rtn_matrix2[(2, 2)] = 7.105e+01;
        rtn_matrix2[(3, 3)] = 6.886e-05;
        rtn_matrix2[(4, 4)] = 1.059e-05;
        rtn_matrix2[(5, 5)] = 5.178e-05;
        let rtn_cov2 = CDMRTNCovariance {
            matrix: rtn_matrix2,
            dimension: CDMCovarianceDimension::SixBySix,
            comments: Vec::new(),
        };
        let obj2 = CDMObject::new(meta2, sv2, rtn_cov2);

        // Build CDM with all optional fields
        let cdm = CDM {
            header: CDMHeader {
                format_version: 1.0,
                classification: Some("RESTRICTED".to_string()),
                creation_date: creation,
                originator: "TEST_ORG".to_string(),
                message_for: Some("SAT-A".to_string()),
                message_id: "MSG-2024-001".to_string(),
                comments: vec!["Header comment".to_string()],
            },
            relative_metadata: rm,
            object1: obj1,
            object2: obj2,
            user_defined: Some(CCSDSUserDefined {
                parameters: {
                    let mut m = HashMap::new();
                    m.insert("PARAM_A".to_string(), "VALUE_A".to_string());
                    m.insert("PARAM_B".to_string(), "42".to_string());
                    m
                },
            }),
        };

        let written = write_cdm(&cdm).unwrap();

        // Verify header
        assert!(written.contains("CLASSIFICATION"));
        assert!(written.contains("RESTRICTED"));
        assert!(written.contains("MESSAGE_FOR"));
        assert!(written.contains("SAT-A"));
        assert!(written.contains("COMMENT Header comment"));

        // Verify relative metadata
        assert!(written.contains("CONJUNCTION_ID"));
        assert!(written.contains("MAHALANOBIS_DISTANCE"));
        assert!(written.contains("APPROACH_ANGLE"));
        assert!(written.contains("SCREEN_TYPE"));
        assert!(written.contains("SCREEN_VOLUME_RADIUS"));
        assert!(written.contains("SCREEN_PC_THRESHOLD"));
        assert!(written.contains("COLLISION_PERCENTILE"));
        assert!(written.contains("COLLISION_MAX_PROBABILITY"));
        assert!(written.contains("COLLISION_MAX_PC_METHOD"));
        assert!(written.contains("SEFI_COLLISION_PROBABILITY"));
        assert!(written.contains("SEFI_COLLISION_PROBABILITY_METHOD"));
        assert!(written.contains("SEFI_FRAGMENTATION_MODEL"));
        assert!(written.contains("PREVIOUS_MESSAGE_ID"));
        assert!(written.contains("PREVIOUS_MESSAGE_EPOCH"));
        assert!(written.contains("NEXT_MESSAGE_EPOCH"));

        // Verify object metadata optional fields
        assert!(written.contains("OPS_STATUS"));
        assert!(written.contains("ODM_MSG_LINK"));
        assert!(written.contains("ADM_MSG_LINK"));
        assert!(written.contains("OBS_BEFORE_NEXT_MESSAGE"));
        assert!(written.contains("COVARIANCE_SOURCE"));
        assert!(written.contains("ALT_COV_TYPE"));
        assert!(written.contains("ALT_COV_REF_FRAME"));

        // Verify OD parameters
        assert!(written.contains("OD_EPOCH"));

        // Verify additional parameters
        assert!(written.contains("AREA_PC_MIN"));
        assert!(written.contains("AREA_PC_MAX"));
        assert!(written.contains("AREA_DRG"));
        assert!(written.contains("AREA_SRP"));
        assert!(written.contains("OEB_PARENT_FRAME"));
        assert!(written.contains("OEB_PARENT_FRAME_EPOCH"));
        assert!(written.contains("OEB_Q1"));
        assert!(written.contains("OEB_Q2"));
        assert!(written.contains("OEB_Q3"));
        assert!(written.contains("OEB_QC"));
        assert!(written.contains("OEB_MAX"));
        assert!(written.contains("OEB_INT"));
        assert!(written.contains("OEB_MIN"));
        assert!(written.contains("AREA_ALONG_OEB_MAX"));
        assert!(written.contains("AREA_ALONG_OEB_INT"));
        assert!(written.contains("AREA_ALONG_OEB_MIN"));
        assert!(written.contains("RCS"));
        assert!(written.contains("RCS_MIN"));
        assert!(written.contains("RCS_MAX"));
        assert!(written.contains("VM_ABSOLUTE"));
        assert!(written.contains("VM_APPARENT_MIN"));
        assert!(written.contains("VM_APPARENT"));
        assert!(written.contains("VM_APPARENT_MAX"));
        assert!(written.contains("REFLECTANCE"));
        assert!(written.contains("HBR"));
        assert!(written.contains("LEAD_TIME_REQD_BEFORE_TCA"));
        assert!(written.contains("APOAPSIS_ALTITUDE"));
        assert!(written.contains("PERIAPSIS_ALTITUDE"));
        assert!(written.contains("INCLINATION"));
        assert!(written.contains("COV_CONFIDENCE"));
        assert!(written.contains("COV_CONFIDENCE_METHOD"));

        // Verify XYZ covariance block
        assert!(written.contains("CX_X"));
        assert!(written.contains("CY_Y"));
        assert!(written.contains("CZ_Z"));
        assert!(written.contains("CXDOT_XDOT"));
        assert!(written.contains("CYDOT_YDOT"));
        assert!(written.contains("CZDOT_ZDOT"));
        assert!(written.contains("COMMENT XYZ covariance comment"));

        // Verify additional covariance metadata
        assert!(written.contains("DENSITY_FORECAST_UNCERTAINTY"));
        assert!(written.contains("CSCALE_FACTOR_MIN"));
        assert!(written.contains("CSCALE_FACTOR_MAX"));
        assert!(written.contains("SCREENING_DATA_SOURCE"));
        assert!(written.contains("DCP_SENSITIVITY_VECTOR_POSITION"));
        assert!(written.contains("DCP_SENSITIVITY_VECTOR_VELOCITY"));

        // Verify CSIG3EIGVEC3
        assert!(written.contains("CSIG3EIGVEC3"));

        // Verify data comments
        assert!(written.contains("COMMENT Data section comment"));
        assert!(written.contains("COMMENT RTN covariance comment"));
        assert!(written.contains("COMMENT Additional covariance comment"));
        assert!(written.contains("COMMENT OD parameters comment"));
        assert!(written.contains("COMMENT Additional params comment"));

        // Verify user-defined parameters
        assert!(written.contains("USER_DEFINED_PARAM_A"));
        assert!(written.contains("USER_DEFINED_PARAM_B"));
    }

    #[test]
    #[parallel]
    fn test_cdm_kvn_round_trip_example4_9x9_covariance() {
        // CDMExample4 has 9x9 covariance (THR row)
        let content = std::fs::read_to_string("test_assets/ccsds/cdm/CDMExample4.txt").unwrap();
        let cdm = crate::ccsds::kvn::parse_cdm(&content).unwrap();

        let written = write_cdm(&cdm).unwrap();

        // Verify 9x9 covariance fields (thrust row)
        assert!(written.contains("CTHR_R"));
        assert!(written.contains("CTHR_THR"));
        assert!(written.contains("CDRG_DRG"));
        assert!(written.contains("CSRP_SRP"));

        // Round-trip
        let cdm2 = crate::ccsds::kvn::parse_cdm(&written).unwrap();
        assert_eq!(
            cdm2.object1.data.rtn_covariance.dimension,
            CDMCovarianceDimension::NineByNine
        );
    }

    #[test]
    #[parallel]
    fn test_cdm_write_ion_starlink_round_trip() {
        // ION_SCV8_vs_STARLINK_1233 has operator fields and ITRF ref frame
        let content =
            std::fs::read_to_string("test_assets/ccsds/cdm/ION_SCV8_vs_STARLINK_1233.txt").unwrap();
        let cdm = crate::ccsds::kvn::parse_cdm(&content).unwrap();

        let written = write_cdm(&cdm).unwrap();
        assert!(written.contains("OPERATOR_CONTACT_POSITION"));
        assert!(written.contains("OPERATOR_ORGANIZATION"));
        assert!(written.contains("OPERATOR_PHONE"));
        assert!(written.contains("OPERATOR_EMAIL"));
        assert!(written.contains("MESSAGE_FOR"));

        // Round-trip verify
        let cdm2 = crate::ccsds::kvn::parse_cdm(&written).unwrap();
        assert_eq!(cdm2.header.originator, cdm.header.originator);
        assert!(cdm2.header.message_for.is_some());
        assert!(cdm2.object1.metadata.operator_email.is_some());
    }

    #[test]
    #[parallel]
    fn test_cdm_write_minimal_round_trip() {
        // CDMExample5 is a minimal CDM (only mandatory fields)
        let content = std::fs::read_to_string("test_assets/ccsds/cdm/CDMExample5.txt").unwrap();
        let cdm = crate::ccsds::kvn::parse_cdm(&content).unwrap();
        let written = write_cdm(&cdm).unwrap();
        let cdm2 = crate::ccsds::kvn::parse_cdm(&written).unwrap();

        assert_eq!(cdm2.header.message_id, cdm.header.message_id);
        assert_eq!(cdm2.object1.metadata.object_name, "SATELLITE A");
        assert_eq!(cdm2.object2.metadata.object_name, "FENGYUN 1C DEB");
    }

    #[test]
    #[parallel]
    fn test_cdm_write_state_vector_comments() {
        use crate::ccsds::cdm::*;
        use crate::ccsds::common::{CCSDSRefFrame, CDMCovarianceDimension};
        use crate::time::Epoch;
        use nalgebra::SMatrix;

        let tca = Epoch::from_datetime(2024, 1, 15, 12, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);

        let mut sv = CDMStateVector::new([7000e3, 0.0, 0.0], [0.0, 7500.0, 0.0]);
        sv.comments = vec!["State vector comment".to_string()];

        let meta = CDMObjectMetadata::new(
            "OBJECT1".to_string(),
            "99999".to_string(),
            "SATCAT".to_string(),
            "TEST-SAT".to_string(),
            "2024-999A".to_string(),
            "NONE".to_string(),
            "CALCULATED".to_string(),
            "NO".to_string(),
            CCSDSRefFrame::EME2000,
        );
        let rtn = CDMRTNCovariance {
            matrix: SMatrix::<f64, 9, 9>::identity(),
            dimension: CDMCovarianceDimension::SixBySix,
            comments: Vec::new(),
        };
        let obj1 = CDMObject::new(meta.clone(), sv, rtn.clone());

        let sv2 = CDMStateVector::new([6000e3, 1000e3, 0.0], [0.0, 6500.0, 1000.0]);
        let mut meta2 = meta;
        meta2.object = "OBJECT2".to_string();
        meta2.object_name = "TEST-DEB".to_string();
        let obj2 = CDMObject::new(meta2, sv2, rtn);

        let cdm = CDM::new(
            "TEST".to_string(),
            "MSG-001".to_string(),
            tca,
            500.0,
            obj1,
            obj2,
        );

        let written = write_cdm(&cdm).unwrap();
        assert!(written.contains("COMMENT State vector comment"));
    }
}
