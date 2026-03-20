/*!
 * XML writer for CCSDS OEM, OMM, and OPM messages.
 *
 * Stub — implemented in Stage 3 (OEM), Stage 5 (OMM), Stage 6 (OPM).
 */

use crate::utils::errors::BraheError;

/// Write an OEM message to XML format.
pub fn write_oem_xml(_oem: &crate::ccsds::oem::OEM) -> Result<String, BraheError> {
    Err(BraheError::Error(
        "OEM XML writer not yet implemented".to_string(),
    ))
}

/// Write an OMM message to XML format.
pub fn write_omm_xml(_omm: &crate::ccsds::omm::OMM) -> Result<String, BraheError> {
    Err(BraheError::Error(
        "OMM XML writer not yet implemented".to_string(),
    ))
}

/// Write an OPM message to XML format.
pub fn write_opm_xml(_opm: &crate::ccsds::opm::OPM) -> Result<String, BraheError> {
    Err(BraheError::Error(
        "OPM XML writer not yet implemented".to_string(),
    ))
}

/// Write a CDM message to XML format.
pub fn write_cdm_xml(cdm: &crate::ccsds::cdm::CDM) -> Result<String, BraheError> {
    use crate::ccsds::cdm::*;
    use crate::ccsds::common::{covariance9x9_to_lower_triangular, format_ccsds_datetime};

    let mut out = String::new();
    let i1 = "  ";
    let i2 = "    ";
    let i3 = "      ";
    let i4 = "        ";

    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str(&format!(
        "<cdm id=\"CCSDS_CDM_VERS\" version=\"{:.1}\">\n",
        cdm.header.format_version
    ));

    // Header
    out.push_str(&format!("{}<header>\n", i1));
    for c in &cdm.header.comments {
        out.push_str(&format!("{}<COMMENT>{}</COMMENT>\n", i2, c));
    }
    if let Some(ref cl) = cdm.header.classification {
        out.push_str(&format!("{}<CLASSIFICATION>{}</CLASSIFICATION>\n", i2, cl));
    }
    out.push_str(&format!(
        "{}<CREATION_DATE>{}</CREATION_DATE>\n",
        i2,
        format_ccsds_datetime(&cdm.header.creation_date)
    ));
    out.push_str(&format!(
        "{}<ORIGINATOR>{}</ORIGINATOR>\n",
        i2, cdm.header.originator
    ));
    if let Some(ref mf) = cdm.header.message_for {
        out.push_str(&format!("{}<MESSAGE_FOR>{}</MESSAGE_FOR>\n", i2, mf));
    }
    out.push_str(&format!(
        "{}<MESSAGE_ID>{}</MESSAGE_ID>\n",
        i2, cdm.header.message_id
    ));
    out.push_str(&format!("{}</header>\n", i1));

    // Body
    out.push_str(&format!("{}<body>\n", i1));

    // Relative metadata/data
    out.push_str(&format!("{}<relativeMetadataData>\n", i2));
    let rm = &cdm.relative_metadata;
    for c in &rm.comments {
        out.push_str(&format!("{}<COMMENT>{}</COMMENT>\n", i3, c));
    }
    if let Some(ref cid) = rm.conjunction_id {
        out.push_str(&format!("{}<CONJUNCTION_ID>{}</CONJUNCTION_ID>\n", i3, cid));
    }
    out.push_str(&format!(
        "{}<TCA>{}</TCA>\n",
        i3,
        format_ccsds_datetime(&rm.tca)
    ));
    out.push_str(&format!(
        "{}<MISS_DISTANCE units=\"m\">{}</MISS_DISTANCE>\n",
        i3, rm.miss_distance
    ));
    if let Some(v) = rm.relative_speed {
        out.push_str(&format!(
            "{}<RELATIVE_SPEED units=\"m/s\">{}</RELATIVE_SPEED>\n",
            i3, v
        ));
    }

    // Relative state vector
    let has_rel_state = rm.relative_position_r.is_some() || rm.relative_velocity_r.is_some();
    if has_rel_state {
        out.push_str(&format!("{}<relativeStateVector>\n", i3));
        if let Some(v) = rm.relative_position_r {
            out.push_str(&format!(
                "{}<RELATIVE_POSITION_R units=\"m\">{}</RELATIVE_POSITION_R>\n",
                i4, v
            ));
        }
        if let Some(v) = rm.relative_position_t {
            out.push_str(&format!(
                "{}<RELATIVE_POSITION_T units=\"m\">{}</RELATIVE_POSITION_T>\n",
                i4, v
            ));
        }
        if let Some(v) = rm.relative_position_n {
            out.push_str(&format!(
                "{}<RELATIVE_POSITION_N units=\"m\">{}</RELATIVE_POSITION_N>\n",
                i4, v
            ));
        }
        if let Some(v) = rm.relative_velocity_r {
            out.push_str(&format!(
                "{}<RELATIVE_VELOCITY_R units=\"m/s\">{}</RELATIVE_VELOCITY_R>\n",
                i4, v
            ));
        }
        if let Some(v) = rm.relative_velocity_t {
            out.push_str(&format!(
                "{}<RELATIVE_VELOCITY_T units=\"m/s\">{}</RELATIVE_VELOCITY_T>\n",
                i4, v
            ));
        }
        if let Some(v) = rm.relative_velocity_n {
            out.push_str(&format!(
                "{}<RELATIVE_VELOCITY_N units=\"m/s\">{}</RELATIVE_VELOCITY_N>\n",
                i4, v
            ));
        }
        out.push_str(&format!("{}</relativeStateVector>\n", i3));
    }

    // Screening and collision probability fields
    if let Some(v) = rm.mahalanobis_distance {
        out.push_str(&format!(
            "{}<MAHALANOBIS_DISTANCE>{}</MAHALANOBIS_DISTANCE>\n",
            i3, v
        ));
    }
    if let Some(v) = rm.approach_angle {
        out.push_str(&format!(
            "{}<APPROACH_ANGLE units=\"deg\">{}</APPROACH_ANGLE>\n",
            i3, v
        ));
    }
    if let Some(ref e) = rm.start_screen_period {
        out.push_str(&format!(
            "{}<START_SCREEN_PERIOD>{}</START_SCREEN_PERIOD>\n",
            i3,
            format_ccsds_datetime(e)
        ));
    }
    if let Some(ref e) = rm.stop_screen_period {
        out.push_str(&format!(
            "{}<STOP_SCREEN_PERIOD>{}</STOP_SCREEN_PERIOD>\n",
            i3,
            format_ccsds_datetime(e)
        ));
    }
    if let Some(ref s) = rm.screen_type {
        out.push_str(&format!("{}<SCREEN_TYPE>{}</SCREEN_TYPE>\n", i3, s));
    }
    if let Some(ref s) = rm.screen_volume_frame {
        out.push_str(&format!(
            "{}<SCREEN_VOLUME_FRAME>{}</SCREEN_VOLUME_FRAME>\n",
            i3, s
        ));
    }
    if let Some(ref s) = rm.screen_volume_shape {
        out.push_str(&format!(
            "{}<SCREEN_VOLUME_SHAPE>{}</SCREEN_VOLUME_SHAPE>\n",
            i3, s
        ));
    }
    if let Some(v) = rm.screen_volume_radius {
        out.push_str(&format!(
            "{}<SCREEN_VOLUME_RADIUS units=\"m\">{}</SCREEN_VOLUME_RADIUS>\n",
            i3, v
        ));
    }
    if let Some(v) = rm.screen_volume_x {
        out.push_str(&format!(
            "{}<SCREEN_VOLUME_X units=\"m\">{}</SCREEN_VOLUME_X>\n",
            i3, v
        ));
    }
    if let Some(v) = rm.screen_volume_y {
        out.push_str(&format!(
            "{}<SCREEN_VOLUME_Y units=\"m\">{}</SCREEN_VOLUME_Y>\n",
            i3, v
        ));
    }
    if let Some(v) = rm.screen_volume_z {
        out.push_str(&format!(
            "{}<SCREEN_VOLUME_Z units=\"m\">{}</SCREEN_VOLUME_Z>\n",
            i3, v
        ));
    }
    if let Some(ref e) = rm.screen_entry_time {
        out.push_str(&format!(
            "{}<SCREEN_ENTRY_TIME>{}</SCREEN_ENTRY_TIME>\n",
            i3,
            format_ccsds_datetime(e)
        ));
    }
    if let Some(ref e) = rm.screen_exit_time {
        out.push_str(&format!(
            "{}<SCREEN_EXIT_TIME>{}</SCREEN_EXIT_TIME>\n",
            i3,
            format_ccsds_datetime(e)
        ));
    }
    if let Some(v) = rm.screen_pc_threshold {
        out.push_str(&format!(
            "{}<SCREEN_PC_THRESHOLD>{:E}</SCREEN_PC_THRESHOLD>\n",
            i3, v
        ));
    }
    if let Some(ref cp) = rm.collision_percentile {
        let s: Vec<String> = cp.iter().map(|v| v.to_string()).collect();
        out.push_str(&format!(
            "{}<COLLISION_PERCENTILE>{}</COLLISION_PERCENTILE>\n",
            i3,
            s.join(" ")
        ));
    }
    if let Some(v) = rm.collision_probability {
        out.push_str(&format!(
            "{}<COLLISION_PROBABILITY>{:E}</COLLISION_PROBABILITY>\n",
            i3, v
        ));
    }
    if let Some(ref s) = rm.collision_probability_method {
        out.push_str(&format!(
            "{}<COLLISION_PROBABILITY_METHOD>{}</COLLISION_PROBABILITY_METHOD>\n",
            i3, s
        ));
    }
    if let Some(v) = rm.collision_max_probability {
        out.push_str(&format!(
            "{}<COLLISION_MAX_PROBABILITY>{:E}</COLLISION_MAX_PROBABILITY>\n",
            i3, v
        ));
    }
    if let Some(ref s) = rm.collision_max_pc_method {
        out.push_str(&format!(
            "{}<COLLISION_MAX_PC_METHOD>{}</COLLISION_MAX_PC_METHOD>\n",
            i3, s
        ));
    }
    if let Some(v) = rm.sefi_collision_probability {
        out.push_str(&format!(
            "{}<SEFI_COLLISION_PROBABILITY>{:E}</SEFI_COLLISION_PROBABILITY>\n",
            i3, v
        ));
    }
    if let Some(ref s) = rm.sefi_collision_probability_method {
        out.push_str(&format!(
            "{}<SEFI_COLLISION_PROBABILITY_METHOD>{}</SEFI_COLLISION_PROBABILITY_METHOD>\n",
            i3, s
        ));
    }
    if let Some(ref s) = rm.sefi_fragmentation_model {
        out.push_str(&format!(
            "{}<SEFI_FRAGMENTATION_MODEL>{}</SEFI_FRAGMENTATION_MODEL>\n",
            i3, s
        ));
    }
    if let Some(ref s) = rm.previous_message_id {
        out.push_str(&format!(
            "{}<PREVIOUS_MESSAGE_ID>{}</PREVIOUS_MESSAGE_ID>\n",
            i3, s
        ));
    }
    if let Some(ref e) = rm.previous_message_epoch {
        out.push_str(&format!(
            "{}<PREVIOUS_MESSAGE_EPOCH>{}</PREVIOUS_MESSAGE_EPOCH>\n",
            i3,
            format_ccsds_datetime(e)
        ));
    }
    if let Some(ref e) = rm.next_message_epoch {
        out.push_str(&format!(
            "{}<NEXT_MESSAGE_EPOCH>{}</NEXT_MESSAGE_EPOCH>\n",
            i3,
            format_ccsds_datetime(e)
        ));
    }

    out.push_str(&format!("{}</relativeMetadataData>\n", i2));

    // Write object segments
    let write_segment = |out: &mut String, obj: &CDMObject| {
        let m = &obj.metadata;
        let d = &obj.data;

        out.push_str(&format!("{}<segment>\n", i2));

        // Metadata
        out.push_str(&format!("{}<metadata>\n", i3));
        for c in &m.comments {
            out.push_str(&format!("{}<COMMENT>{}</COMMENT>\n", i4, c));
        }
        out.push_str(&format!("{}<OBJECT>{}</OBJECT>\n", i4, m.object));
        out.push_str(&format!(
            "{}<OBJECT_DESIGNATOR>{}</OBJECT_DESIGNATOR>\n",
            i4, m.object_designator
        ));
        out.push_str(&format!(
            "{}<CATALOG_NAME>{}</CATALOG_NAME>\n",
            i4, m.catalog_name
        ));
        out.push_str(&format!(
            "{}<OBJECT_NAME>{}</OBJECT_NAME>\n",
            i4, m.object_name
        ));
        out.push_str(&format!(
            "{}<INTERNATIONAL_DESIGNATOR>{}</INTERNATIONAL_DESIGNATOR>\n",
            i4, m.international_designator
        ));
        if let Some(ref v) = m.object_type {
            out.push_str(&format!("{}<OBJECT_TYPE>{}</OBJECT_TYPE>\n", i4, v));
        }
        if let Some(ref v) = m.ops_status {
            out.push_str(&format!("{}<OPS_STATUS>{}</OPS_STATUS>\n", i4, v));
        }
        if let Some(ref v) = m.operator_contact_position {
            out.push_str(&format!(
                "{}<OPERATOR_CONTACT_POSITION>{}</OPERATOR_CONTACT_POSITION>\n",
                i4, v
            ));
        }
        if let Some(ref v) = m.operator_organization {
            out.push_str(&format!(
                "{}<OPERATOR_ORGANIZATION>{}</OPERATOR_ORGANIZATION>\n",
                i4, v
            ));
        }
        if let Some(ref v) = m.operator_phone {
            out.push_str(&format!("{}<OPERATOR_PHONE>{}</OPERATOR_PHONE>\n", i4, v));
        }
        if let Some(ref v) = m.operator_email {
            out.push_str(&format!("{}<OPERATOR_EMAIL>{}</OPERATOR_EMAIL>\n", i4, v));
        }
        out.push_str(&format!(
            "{}<EPHEMERIS_NAME>{}</EPHEMERIS_NAME>\n",
            i4, m.ephemeris_name
        ));
        if let Some(ref v) = m.odm_msg_link {
            out.push_str(&format!("{}<ODM_MSG_LINK>{}</ODM_MSG_LINK>\n", i4, v));
        }
        if let Some(ref v) = m.adm_msg_link {
            out.push_str(&format!("{}<ADM_MSG_LINK>{}</ADM_MSG_LINK>\n", i4, v));
        }
        if let Some(ref v) = m.obs_before_next_message {
            out.push_str(&format!(
                "{}<OBS_BEFORE_NEXT_MESSAGE>{}</OBS_BEFORE_NEXT_MESSAGE>\n",
                i4, v
            ));
        }
        out.push_str(&format!(
            "{}<COVARIANCE_METHOD>{}</COVARIANCE_METHOD>\n",
            i4, m.covariance_method
        ));
        if let Some(ref v) = m.covariance_source {
            out.push_str(&format!(
                "{}<COVARIANCE_SOURCE>{}</COVARIANCE_SOURCE>\n",
                i4, v
            ));
        }
        out.push_str(&format!(
            "{}<MANEUVERABLE>{}</MANEUVERABLE>\n",
            i4, m.maneuverable
        ));
        if let Some(ref v) = m.orbit_center {
            out.push_str(&format!("{}<ORBIT_CENTER>{}</ORBIT_CENTER>\n", i4, v));
        }
        out.push_str(&format!("{}<REF_FRAME>{}</REF_FRAME>\n", i4, m.ref_frame));
        if let Some(ref v) = m.alt_cov_type {
            out.push_str(&format!("{}<ALT_COV_TYPE>{}</ALT_COV_TYPE>\n", i4, v));
        }
        if let Some(ref v) = m.alt_cov_ref_frame {
            out.push_str(&format!(
                "{}<ALT_COV_REF_FRAME>{}</ALT_COV_REF_FRAME>\n",
                i4, v
            ));
        }
        if let Some(ref v) = m.gravity_model {
            out.push_str(&format!("{}<GRAVITY_MODEL>{}</GRAVITY_MODEL>\n", i4, v));
        }
        if let Some(ref v) = m.atmospheric_model {
            out.push_str(&format!(
                "{}<ATMOSPHERIC_MODEL>{}</ATMOSPHERIC_MODEL>\n",
                i4, v
            ));
        }
        if let Some(ref v) = m.n_body_perturbations {
            out.push_str(&format!(
                "{}<N_BODY_PERTURBATIONS>{}</N_BODY_PERTURBATIONS>\n",
                i4, v
            ));
        }
        if let Some(ref v) = m.solar_rad_pressure {
            out.push_str(&format!(
                "{}<SOLAR_RAD_PRESSURE>{}</SOLAR_RAD_PRESSURE>\n",
                i4, v
            ));
        }
        if let Some(ref v) = m.earth_tides {
            out.push_str(&format!("{}<EARTH_TIDES>{}</EARTH_TIDES>\n", i4, v));
        }
        if let Some(ref v) = m.intrack_thrust {
            out.push_str(&format!("{}<INTRACK_THRUST>{}</INTRACK_THRUST>\n", i4, v));
        }
        out.push_str(&format!("{}</metadata>\n", i3));

        // Data
        out.push_str(&format!("{}<data>\n", i3));
        for c in &d.comments {
            out.push_str(&format!("{}<COMMENT>{}</COMMENT>\n", i4, c));
        }

        // State vector (m → km)
        out.push_str(&format!("{}<stateVector>\n", i4));
        for c in &d.state_vector.comments {
            out.push_str(&format!("        <COMMENT>{}</COMMENT>\n", c));
        }
        out.push_str(&format!(
            "        <X units=\"km\">{:.6}</X>\n",
            d.state_vector.position[0] / 1e3
        ));
        out.push_str(&format!(
            "        <Y units=\"km\">{:.6}</Y>\n",
            d.state_vector.position[1] / 1e3
        ));
        out.push_str(&format!(
            "        <Z units=\"km\">{:.6}</Z>\n",
            d.state_vector.position[2] / 1e3
        ));
        out.push_str(&format!(
            "        <X_DOT units=\"km/s\">{:.9}</X_DOT>\n",
            d.state_vector.velocity[0] / 1e3
        ));
        out.push_str(&format!(
            "        <Y_DOT units=\"km/s\">{:.9}</Y_DOT>\n",
            d.state_vector.velocity[1] / 1e3
        ));
        out.push_str(&format!(
            "        <Z_DOT units=\"km/s\">{:.9}</Z_DOT>\n",
            d.state_vector.velocity[2] / 1e3
        ));
        out.push_str(&format!("{}</stateVector>\n", i4));

        // RTN covariance
        let rtn_names: &[&str] = &[
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
            "CDRG_R",
            "CDRG_T",
            "CDRG_N",
            "CDRG_RDOT",
            "CDRG_TDOT",
            "CDRG_NDOT",
            "CDRG_DRG",
            "CSRP_R",
            "CSRP_T",
            "CSRP_N",
            "CSRP_RDOT",
            "CSRP_TDOT",
            "CSRP_NDOT",
            "CSRP_DRG",
            "CSRP_SRP",
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
        let rtn_vals =
            covariance9x9_to_lower_triangular(&d.rtn_covariance.matrix, d.rtn_covariance.dimension);
        out.push_str(&format!("{}<covarianceMatrix>\n", i4));
        for (i, v) in rtn_vals.iter().enumerate() {
            out.push_str(&format!(
                "        <{}>{:E}</{}>\n",
                rtn_names[i], v, rtn_names[i]
            ));
        }
        out.push_str(&format!("{}</covarianceMatrix>\n", i4));

        // OD parameters
        if let Some(ref od) = d.od_parameters {
            out.push_str(&format!("{}<odParameters>\n", i4));
            for c in &od.comments {
                out.push_str(&format!("        <COMMENT>{}</COMMENT>\n", c));
            }
            if let Some(ref e) = od.time_lastob_start {
                out.push_str(&format!(
                    "        <TIME_LASTOB_START>{}</TIME_LASTOB_START>\n",
                    format_ccsds_datetime(e)
                ));
            }
            if let Some(ref e) = od.time_lastob_end {
                out.push_str(&format!(
                    "        <TIME_LASTOB_END>{}</TIME_LASTOB_END>\n",
                    format_ccsds_datetime(e)
                ));
            }
            if let Some(v) = od.recommended_od_span {
                out.push_str(&format!(
                    "        <RECOMMENDED_OD_SPAN units=\"d\">{:.2}</RECOMMENDED_OD_SPAN>\n",
                    v
                ));
            }
            if let Some(v) = od.actual_od_span {
                out.push_str(&format!(
                    "        <ACTUAL_OD_SPAN units=\"d\">{:.2}</ACTUAL_OD_SPAN>\n",
                    v
                ));
            }
            if let Some(v) = od.obs_available {
                out.push_str(&format!("        <OBS_AVAILABLE>{}</OBS_AVAILABLE>\n", v));
            }
            if let Some(v) = od.obs_used {
                out.push_str(&format!("        <OBS_USED>{}</OBS_USED>\n", v));
            }
            if let Some(v) = od.tracks_available {
                out.push_str(&format!(
                    "        <TRACKS_AVAILABLE>{}</TRACKS_AVAILABLE>\n",
                    v
                ));
            }
            if let Some(v) = od.tracks_used {
                out.push_str(&format!("        <TRACKS_USED>{}</TRACKS_USED>\n", v));
            }
            if let Some(v) = od.residuals_accepted {
                out.push_str(&format!(
                    "        <RESIDUALS_ACCEPTED units=\"%\">{}</RESIDUALS_ACCEPTED>\n",
                    v
                ));
            }
            if let Some(v) = od.weighted_rms {
                out.push_str(&format!("        <WEIGHTED_RMS>{}</WEIGHTED_RMS>\n", v));
            }
            if let Some(ref e) = od.od_epoch {
                out.push_str(&format!(
                    "        <OD_EPOCH>{}</OD_EPOCH>\n",
                    format_ccsds_datetime(e)
                ));
            }
            out.push_str(&format!("{}</odParameters>\n", i4));
        }

        // Additional parameters
        if let Some(ref ap) = d.additional_parameters {
            out.push_str(&format!("{}<additionalParameters>\n", i4));
            for c in &ap.comments {
                out.push_str(&format!("        <COMMENT>{}</COMMENT>\n", c));
            }
            if let Some(v) = ap.area_pc {
                out.push_str(&format!(
                    "        <AREA_PC units=\"m**2\">{}</AREA_PC>\n",
                    v
                ));
            }
            if let Some(v) = ap.area_pc_min {
                out.push_str(&format!(
                    "        <AREA_PC_MIN units=\"m**2\">{}</AREA_PC_MIN>\n",
                    v
                ));
            }
            if let Some(v) = ap.area_pc_max {
                out.push_str(&format!(
                    "        <AREA_PC_MAX units=\"m**2\">{}</AREA_PC_MAX>\n",
                    v
                ));
            }
            if let Some(v) = ap.area_drg {
                out.push_str(&format!(
                    "        <AREA_DRG units=\"m**2\">{}</AREA_DRG>\n",
                    v
                ));
            }
            if let Some(v) = ap.area_srp {
                out.push_str(&format!(
                    "        <AREA_SRP units=\"m**2\">{}</AREA_SRP>\n",
                    v
                ));
            }
            if let Some(ref v) = ap.oeb_parent_frame {
                out.push_str(&format!(
                    "        <OEB_PARENT_FRAME>{}</OEB_PARENT_FRAME>\n",
                    v
                ));
            }
            if let Some(ref e) = ap.oeb_parent_frame_epoch {
                out.push_str(&format!(
                    "        <OEB_PARENT_FRAME_EPOCH>{}</OEB_PARENT_FRAME_EPOCH>\n",
                    format_ccsds_datetime(e)
                ));
            }
            if let Some(v) = ap.oeb_q1 {
                out.push_str(&format!("        <OEB_Q1>{}</OEB_Q1>\n", v));
            }
            if let Some(v) = ap.oeb_q2 {
                out.push_str(&format!("        <OEB_Q2>{}</OEB_Q2>\n", v));
            }
            if let Some(v) = ap.oeb_q3 {
                out.push_str(&format!("        <OEB_Q3>{}</OEB_Q3>\n", v));
            }
            if let Some(v) = ap.oeb_qc {
                out.push_str(&format!("        <OEB_QC>{}</OEB_QC>\n", v));
            }
            if let Some(v) = ap.oeb_max {
                out.push_str(&format!("        <OEB_MAX units=\"m\">{}</OEB_MAX>\n", v));
            }
            if let Some(v) = ap.oeb_int {
                out.push_str(&format!("        <OEB_INT units=\"m\">{}</OEB_INT>\n", v));
            }
            if let Some(v) = ap.oeb_min {
                out.push_str(&format!("        <OEB_MIN units=\"m\">{}</OEB_MIN>\n", v));
            }
            if let Some(v) = ap.area_along_oeb_max {
                out.push_str(&format!(
                    "        <AREA_ALONG_OEB_MAX units=\"m**2\">{}</AREA_ALONG_OEB_MAX>\n",
                    v
                ));
            }
            if let Some(v) = ap.area_along_oeb_int {
                out.push_str(&format!(
                    "        <AREA_ALONG_OEB_INT units=\"m**2\">{}</AREA_ALONG_OEB_INT>\n",
                    v
                ));
            }
            if let Some(v) = ap.area_along_oeb_min {
                out.push_str(&format!(
                    "        <AREA_ALONG_OEB_MIN units=\"m**2\">{}</AREA_ALONG_OEB_MIN>\n",
                    v
                ));
            }
            if let Some(v) = ap.rcs {
                out.push_str(&format!("        <RCS units=\"m**2\">{}</RCS>\n", v));
            }
            if let Some(v) = ap.rcs_min {
                out.push_str(&format!(
                    "        <RCS_MIN units=\"m**2\">{}</RCS_MIN>\n",
                    v
                ));
            }
            if let Some(v) = ap.rcs_max {
                out.push_str(&format!(
                    "        <RCS_MAX units=\"m**2\">{}</RCS_MAX>\n",
                    v
                ));
            }
            if let Some(v) = ap.vm_absolute {
                out.push_str(&format!("        <VM_ABSOLUTE>{}</VM_ABSOLUTE>\n", v));
            }
            if let Some(v) = ap.vm_apparent_min {
                out.push_str(&format!(
                    "        <VM_APPARENT_MIN>{}</VM_APPARENT_MIN>\n",
                    v
                ));
            }
            if let Some(v) = ap.vm_apparent {
                out.push_str(&format!("        <VM_APPARENT>{}</VM_APPARENT>\n", v));
            }
            if let Some(v) = ap.vm_apparent_max {
                out.push_str(&format!(
                    "        <VM_APPARENT_MAX>{}</VM_APPARENT_MAX>\n",
                    v
                ));
            }
            if let Some(v) = ap.reflectance {
                out.push_str(&format!("        <REFLECTANCE>{}</REFLECTANCE>\n", v));
            }
            if let Some(v) = ap.mass {
                out.push_str(&format!("        <MASS units=\"kg\">{}</MASS>\n", v));
            }
            if let Some(v) = ap.hbr {
                out.push_str(&format!("        <HBR units=\"m\">{}</HBR>\n", v));
            }
            if let Some(v) = ap.cd_area_over_mass {
                out.push_str(&format!(
                    "        <CD_AREA_OVER_MASS units=\"m**2/kg\">{}</CD_AREA_OVER_MASS>\n",
                    v
                ));
            }
            if let Some(v) = ap.cr_area_over_mass {
                out.push_str(&format!(
                    "        <CR_AREA_OVER_MASS units=\"m**2/kg\">{}</CR_AREA_OVER_MASS>\n",
                    v
                ));
            }
            if let Some(v) = ap.thrust_acceleration {
                out.push_str(&format!(
                    "        <THRUST_ACCELERATION units=\"m/s**2\">{}</THRUST_ACCELERATION>\n",
                    v
                ));
            }
            if let Some(v) = ap.sedr {
                out.push_str(&format!("        <SEDR units=\"W/kg\">{:E}</SEDR>\n", v));
            }
            if let Some(v) = ap.lead_time_reqd_before_tca {
                out.push_str(&format!("        <LEAD_TIME_REQD_BEFORE_TCA units=\"h\">{}</LEAD_TIME_REQD_BEFORE_TCA>\n", v));
            }
            if let Some(v) = ap.apoapsis_altitude {
                out.push_str(&format!(
                    "        <APOAPSIS_ALTITUDE units=\"km\">{}</APOAPSIS_ALTITUDE>\n",
                    v / 1e3
                ));
            }
            if let Some(v) = ap.periapsis_altitude {
                out.push_str(&format!(
                    "        <PERIAPSIS_ALTITUDE units=\"km\">{}</PERIAPSIS_ALTITUDE>\n",
                    v / 1e3
                ));
            }
            if let Some(v) = ap.inclination {
                out.push_str(&format!(
                    "        <INCLINATION units=\"deg\">{}</INCLINATION>\n",
                    v
                ));
            }
            if let Some(v) = ap.cov_confidence {
                out.push_str(&format!("        <COV_CONFIDENCE>{}</COV_CONFIDENCE>\n", v));
            }
            if let Some(ref v) = ap.cov_confidence_method {
                out.push_str(&format!(
                    "        <COV_CONFIDENCE_METHOD>{}</COV_CONFIDENCE_METHOD>\n",
                    v
                ));
            }
            out.push_str(&format!("{}</additionalParameters>\n", i4));
        }

        // XYZ covariance
        if let Some(ref xyz) = d.xyz_covariance {
            let xyz_names: &[&str] = &[
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
                "CDRG_X",
                "CDRG_Y",
                "CDRG_Z",
                "CDRG_XDOT",
                "CDRG_YDOT",
                "CDRG_ZDOT",
                "CDRG_DRG",
                "CSRP_X",
                "CSRP_Y",
                "CSRP_Z",
                "CSRP_XDOT",
                "CSRP_YDOT",
                "CSRP_ZDOT",
                "CSRP_DRG",
                "CSRP_SRP",
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
            let xyz_vals = covariance9x9_to_lower_triangular(&xyz.matrix, xyz.dimension);
            out.push_str(&format!("{}<xyzCovarianceMatrix>\n", i4));
            for (i, v) in xyz_vals.iter().enumerate() {
                out.push_str(&format!(
                    "        <{}>{:E}</{}>\n",
                    xyz_names[i], v, xyz_names[i]
                ));
            }
            out.push_str(&format!("{}</xyzCovarianceMatrix>\n", i4));
        }

        // CSIG3EIGVEC3
        if let Some(ref s) = d.csig3eigvec3 {
            out.push_str(&format!("        <CSIG3EIGVEC3>{}</CSIG3EIGVEC3>\n", s));
        }

        // Additional covariance metadata
        if let Some(ref acm) = d.additional_covariance_metadata {
            out.push_str(&format!("{}<additionalCovarianceMetadata>\n", i4));
            for c in &acm.comments {
                out.push_str(&format!("        <COMMENT>{}</COMMENT>\n", c));
            }
            if let Some(v) = acm.density_forecast_uncertainty {
                out.push_str(&format!(
                    "        <DENSITY_FORECAST_UNCERTAINTY>{}</DENSITY_FORECAST_UNCERTAINTY>\n",
                    v
                ));
            }
            if let Some(v) = acm.cscale_factor_min {
                out.push_str(&format!(
                    "        <CSCALE_FACTOR_MIN>{}</CSCALE_FACTOR_MIN>\n",
                    v
                ));
            }
            if let Some(v) = acm.cscale_factor {
                out.push_str(&format!("        <CSCALE_FACTOR>{}</CSCALE_FACTOR>\n", v));
            }
            if let Some(v) = acm.cscale_factor_max {
                out.push_str(&format!(
                    "        <CSCALE_FACTOR_MAX>{}</CSCALE_FACTOR_MAX>\n",
                    v
                ));
            }
            if let Some(ref s) = acm.screening_data_source {
                out.push_str(&format!(
                    "        <SCREENING_DATA_SOURCE>{}</SCREENING_DATA_SOURCE>\n",
                    s
                ));
            }
            if let Some(ref v) = acm.dcp_sensitivity_vector_position {
                out.push_str(&format!("        <DCP_SENSITIVITY_VECTOR_POSITION>{} {} {}</DCP_SENSITIVITY_VECTOR_POSITION>\n", v[0], v[1], v[2]));
            }
            if let Some(ref v) = acm.dcp_sensitivity_vector_velocity {
                out.push_str(&format!("        <DCP_SENSITIVITY_VECTOR_VELOCITY>{} {} {}</DCP_SENSITIVITY_VECTOR_VELOCITY>\n", v[0], v[1], v[2]));
            }
            out.push_str(&format!("{}</additionalCovarianceMetadata>\n", i4));
        }

        out.push_str(&format!("{}</data>\n", i3));
        out.push_str(&format!("{}</segment>\n", i2));
    };

    write_segment(&mut out, &cdm.object1);
    write_segment(&mut out, &cdm.object2);

    // User-defined parameters
    if let Some(ref ud) = cdm.user_defined {
        out.push_str(&format!("{}<userDefinedParameters>\n", i2));
        for (k, v) in &ud.parameters {
            out.push_str(&format!("{}<USER_DEFINED_{} value=\"{}\"/>\n", i3, k, v));
        }
        out.push_str(&format!("{}</userDefinedParameters>\n", i2));
    }

    out.push_str(&format!("{}</body>\n", i1));
    out.push_str("</cdm>\n");

    Ok(out)
}
