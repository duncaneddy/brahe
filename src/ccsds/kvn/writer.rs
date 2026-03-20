/*!
 * KVN writer for CCSDS OEM, OMM, and OPM messages.
 */

use crate::ccsds::common::{covariance_to_lower_triangular, format_ccsds_datetime};
use crate::ccsds::oem::OEM;
use crate::utils::errors::BraheError;

/// Write an OEM message to KVN format.
pub fn write_oem(oem: &OEM) -> Result<String, BraheError> {
    let mut out = String::new();

    // Header
    out.push_str(&format!(
        "CCSDS_OEM_VERS = {:.1}\n",
        oem.header.format_version
    ));
    if let Some(ref class) = oem.header.classification {
        out.push_str(&format!("CLASSIFICATION = {}\n", class));
    }
    for comment in &oem.header.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }
    out.push_str(&format!(
        "CREATION_DATE = {}\n",
        format_ccsds_datetime(&oem.header.creation_date)
    ));
    out.push_str(&format!("ORIGINATOR = {}\n", oem.header.originator));
    if let Some(ref msg_id) = oem.header.message_id {
        out.push_str(&format!("MESSAGE_ID = {}\n", msg_id));
    }

    // Segments
    for segment in &oem.segments {
        out.push('\n');

        // Metadata
        out.push_str("META_START\n");
        for comment in &segment.metadata.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        out.push_str(&format!("OBJECT_NAME = {}\n", segment.metadata.object_name));
        out.push_str(&format!("OBJECT_ID = {}\n", segment.metadata.object_id));
        out.push_str(&format!("CENTER_NAME = {}\n", segment.metadata.center_name));
        out.push_str(&format!("REF_FRAME = {}\n", segment.metadata.ref_frame));
        if let Some(ref epoch) = segment.metadata.ref_frame_epoch {
            out.push_str(&format!(
                "REF_FRAME_EPOCH = {}\n",
                format_ccsds_datetime(epoch)
            ));
        }
        out.push_str(&format!("TIME_SYSTEM = {}\n", segment.metadata.time_system));
        out.push_str(&format!(
            "START_TIME = {}\n",
            format_ccsds_datetime(&segment.metadata.start_time)
        ));
        if let Some(ref t) = segment.metadata.useable_start_time {
            out.push_str(&format!(
                "USEABLE_START_TIME = {}\n",
                format_ccsds_datetime(t)
            ));
        }
        if let Some(ref t) = segment.metadata.useable_stop_time {
            out.push_str(&format!(
                "USEABLE_STOP_TIME = {}\n",
                format_ccsds_datetime(t)
            ));
        }
        out.push_str(&format!(
            "STOP_TIME = {}\n",
            format_ccsds_datetime(&segment.metadata.stop_time)
        ));
        if let Some(ref interp) = segment.metadata.interpolation {
            out.push_str(&format!("INTERPOLATION = {}\n", interp));
        }
        if let Some(deg) = segment.metadata.interpolation_degree {
            out.push_str(&format!("INTERPOLATION_DEGREE = {}\n", deg));
        }
        out.push_str("META_STOP\n");

        // Data comments
        for comment in &segment.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }

        // Ephemeris data lines
        for sv in &segment.states {
            let epoch_str = format_ccsds_datetime(&sv.epoch);
            // Convert m → km, m/s → km/s
            let x = sv.position[0] / 1000.0;
            let y = sv.position[1] / 1000.0;
            let z = sv.position[2] / 1000.0;
            let vx = sv.velocity[0] / 1000.0;
            let vy = sv.velocity[1] / 1000.0;
            let vz = sv.velocity[2] / 1000.0;

            if let Some(ref acc) = sv.acceleration {
                let ax = acc[0] / 1000.0;
                let ay = acc[1] / 1000.0;
                let az = acc[2] / 1000.0;
                out.push_str(&format!(
                    "{} {:15.6} {:15.6} {:15.6} {:15.9} {:15.9} {:15.9} {:15.9} {:15.9} {:15.9}\n",
                    epoch_str, x, y, z, vx, vy, vz, ax, ay, az
                ));
            } else {
                out.push_str(&format!(
                    "{} {:15.6} {:15.6} {:15.6} {:15.9} {:15.9} {:15.9}\n",
                    epoch_str, x, y, z, vx, vy, vz
                ));
            }
        }

        // Covariance blocks
        if !segment.covariances.is_empty() {
            out.push_str("\nCOVARIANCE_START\n");
            for cov in &segment.covariances {
                if let Some(ref epoch) = cov.epoch {
                    out.push_str(&format!("EPOCH = {}\n", format_ccsds_datetime(epoch)));
                }
                if let Some(ref frame) = cov.cov_ref_frame {
                    out.push_str(&format!("COV_REF_FRAME = {}\n", frame));
                }
                for comment in &cov.comments {
                    out.push_str(&format!("COMMENT {}\n", comment));
                }

                // Convert m² → km² (factor 1e-6)
                let values = covariance_to_lower_triangular(&cov.matrix, 1e-6);
                let mut idx = 0;
                for row in 0..6 {
                    let line: Vec<String> = (0..=row)
                        .map(|_| {
                            let v = values[idx];
                            idx += 1;
                            format!("{:.10e}", v)
                        })
                        .collect();
                    out.push_str(&line.join(" "));
                    out.push('\n');
                }
            }
            out.push_str("COVARIANCE_STOP\n");
        }
    }

    Ok(out)
}

/// Write an OMM message to KVN format.
///
/// Stub — implemented in Stage 5.
pub fn write_omm(_omm: &crate::ccsds::omm::OMM) -> Result<String, BraheError> {
    Err(BraheError::Error(
        "OMM KVN writer not yet implemented".to_string(),
    ))
}

/// Write an OPM message to KVN format.
///
/// Stub — implemented in Stage 6.
pub fn write_opm(_opm: &crate::ccsds::opm::OPM) -> Result<String, BraheError> {
    Err(BraheError::Error(
        "OPM KVN writer not yet implemented".to_string(),
    ))
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
    kw(
        &mut out,
        "CREATION_DATE",
        &format_ccsds_datetime(&cdm.header.creation_date),
    );
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
    kw(&mut out, "TCA", &format_ccsds_datetime(&rm.tca));
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
        kw(&mut out, "START_SCREEN_PERIOD", &format_ccsds_datetime(e));
    }
    if let Some(ref e) = rm.stop_screen_period {
        kw(&mut out, "STOP_SCREEN_PERIOD", &format_ccsds_datetime(e));
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
        kw(&mut out, "SCREEN_ENTRY_TIME", &format_ccsds_datetime(e));
    }
    if let Some(ref e) = rm.screen_exit_time {
        kw(&mut out, "SCREEN_EXIT_TIME", &format_ccsds_datetime(e));
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
        kw(
            &mut out,
            "PREVIOUS_MESSAGE_EPOCH",
            &format_ccsds_datetime(e),
        );
    }
    if let Some(ref e) = rm.next_message_epoch {
        kw(&mut out, "NEXT_MESSAGE_EPOCH", &format_ccsds_datetime(e));
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

        // OD parameters
        if let Some(ref od) = d.od_parameters {
            for comment in &od.comments {
                out.push_str(&format!("COMMENT {}\n", comment));
            }
            if let Some(ref e) = od.time_lastob_start {
                kw(out, "TIME_LASTOB_START", &format_ccsds_datetime(e));
            }
            if let Some(ref e) = od.time_lastob_end {
                kw(out, "TIME_LASTOB_END", &format_ccsds_datetime(e));
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
                kw(out, "OD_EPOCH", &format_ccsds_datetime(e));
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
                kw(out, "OEB_PARENT_FRAME_EPOCH", &format_ccsds_datetime(e));
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

        // Data comments
        for comment in &d.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
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
    use crate::ccsds::kvn::parse_oem;

    #[test]
    fn test_oem_kvn_round_trip_example1() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let oem = parse_oem(&content).unwrap();

        // Write
        let written = write_oem(&oem).unwrap();

        // Re-parse
        let oem2 = parse_oem(&written).unwrap();

        // Verify
        assert_eq!(oem.segments.len(), oem2.segments.len());
        assert_eq!(oem.header.originator, oem2.header.originator);

        for (seg1, seg2) in oem.segments.iter().zip(oem2.segments.iter()) {
            assert_eq!(seg1.metadata.object_name, seg2.metadata.object_name);
            assert_eq!(seg1.metadata.ref_frame, seg2.metadata.ref_frame);
            assert_eq!(seg1.states.len(), seg2.states.len());
            assert_eq!(seg1.covariances.len(), seg2.covariances.len());

            // Check state vectors are close
            for (s1, s2) in seg1.states.iter().zip(seg2.states.iter()) {
                for i in 0..3 {
                    assert!(
                        (s1.position[i] - s2.position[i]).abs() < 1.0,
                        "Position mismatch: {} vs {}",
                        s1.position[i],
                        s2.position[i]
                    );
                    assert!(
                        (s1.velocity[i] - s2.velocity[i]).abs() < 0.001,
                        "Velocity mismatch: {} vs {}",
                        s1.velocity[i],
                        s2.velocity[i]
                    );
                }
            }

            // Check covariance matrices
            for (c1, c2) in seg1.covariances.iter().zip(seg2.covariances.iter()) {
                assert_eq!(c1.cov_ref_frame, c2.cov_ref_frame);
                for i in 0..6 {
                    for j in 0..6 {
                        let rel_err = if c1.matrix[(i, j)].abs() > 1e-20 {
                            ((c1.matrix[(i, j)] - c2.matrix[(i, j)]) / c1.matrix[(i, j)]).abs()
                        } else {
                            (c1.matrix[(i, j)] - c2.matrix[(i, j)]).abs()
                        };
                        assert!(
                            rel_err < 1e-4,
                            "Covariance mismatch at ({},{}): {} vs {} (rel_err: {})",
                            i,
                            j,
                            c1.matrix[(i, j)],
                            c2.matrix[(i, j)],
                            rel_err
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_oem_kvn_round_trip_example5() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample5.txt").unwrap();
        let oem = parse_oem(&content).unwrap();
        let written = write_oem(&oem).unwrap();
        let oem2 = parse_oem(&written).unwrap();

        assert_eq!(oem.segments[0].states.len(), oem2.segments[0].states.len());
        assert_eq!(
            oem.segments[0].metadata.object_name,
            oem2.segments[0].metadata.object_name
        );
    }
}
