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
        out.push_str(&format!(
            "{}<EPHEMERIS_NAME>{}</EPHEMERIS_NAME>\n",
            i4, m.ephemeris_name
        ));
        out.push_str(&format!(
            "{}<COVARIANCE_METHOD>{}</COVARIANCE_METHOD>\n",
            i4, m.covariance_method
        ));
        out.push_str(&format!(
            "{}<MANEUVERABLE>{}</MANEUVERABLE>\n",
            i4, m.maneuverable
        ));
        out.push_str(&format!("{}<REF_FRAME>{}</REF_FRAME>\n", i4, m.ref_frame));
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

        out.push_str(&format!("{}</data>\n", i3));
        out.push_str(&format!("{}</segment>\n", i2));
    };

    write_segment(&mut out, &cdm.object1);
    write_segment(&mut out, &cdm.object2);

    out.push_str(&format!("{}</body>\n", i1));
    out.push_str("</cdm>\n");

    Ok(out)
}
