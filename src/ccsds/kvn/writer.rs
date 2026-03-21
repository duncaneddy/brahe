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
pub fn write_omm(omm: &crate::ccsds::omm::OMM) -> Result<String, BraheError> {
    let mut out = String::new();

    // Header
    out.push_str(&format!(
        "CCSDS_OMM_VERS = {:.1}\n",
        omm.header.format_version
    ));
    if let Some(ref class) = omm.header.classification {
        out.push_str(&format!("CLASSIFICATION = {}\n", class));
    }
    for comment in &omm.header.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }
    out.push_str(&format!(
        "CREATION_DATE = {}\n",
        format_ccsds_datetime(&omm.header.creation_date)
    ));
    out.push_str(&format!("ORIGINATOR = {}\n", omm.header.originator));
    if let Some(ref msg_id) = omm.header.message_id {
        out.push_str(&format!("MESSAGE_ID = {}\n", msg_id));
    }
    out.push('\n');

    // Metadata comments
    for comment in &omm.metadata.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }
    out.push_str(&format!("OBJECT_NAME = {}\n", omm.metadata.object_name));
    out.push_str(&format!("OBJECT_ID = {}\n", omm.metadata.object_id));
    out.push_str(&format!("CENTER_NAME = {}\n", omm.metadata.center_name));
    out.push_str(&format!("REF_FRAME = {}\n", omm.metadata.ref_frame));
    if let Some(ref epoch) = omm.metadata.ref_frame_epoch {
        out.push_str(&format!(
            "REF_FRAME_EPOCH = {}\n",
            format_ccsds_datetime(epoch)
        ));
    }
    out.push_str(&format!("TIME_SYSTEM = {}\n", omm.metadata.time_system));
    out.push_str(&format!(
        "MEAN_ELEMENT_THEORY = {}\n",
        omm.metadata.mean_element_theory
    ));
    out.push('\n');

    // Mean elements
    for comment in &omm.mean_elements.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }
    out.push_str(&format!(
        "EPOCH = {}\n",
        format_ccsds_datetime(&omm.mean_elements.epoch)
    ));
    if let Some(mm) = omm.mean_elements.mean_motion {
        out.push_str(&format!("MEAN_MOTION = {}\n", mm));
    }
    if let Some(sma) = omm.mean_elements.semi_major_axis {
        out.push_str(&format!("SEMI_MAJOR_AXIS = {}\n", sma));
    }
    out.push_str(&format!(
        "ECCENTRICITY = {}\n",
        omm.mean_elements.eccentricity
    ));
    out.push_str(&format!(
        "INCLINATION = {}\n",
        omm.mean_elements.inclination
    ));
    out.push_str(&format!(
        "RA_OF_ASC_NODE = {}\n",
        omm.mean_elements.ra_of_asc_node
    ));
    out.push_str(&format!(
        "ARG_OF_PERICENTER = {}\n",
        omm.mean_elements.arg_of_pericenter
    ));
    out.push_str(&format!(
        "MEAN_ANOMALY = {}\n",
        omm.mean_elements.mean_anomaly
    ));
    if let Some(gm) = omm.mean_elements.gm {
        // Internal m³/s² → CCSDS km³/s²
        out.push_str(&format!("GM = {}\n", gm / 1e9));
    }

    // TLE parameters
    if let Some(ref tle) = omm.tle_parameters {
        out.push('\n');
        for comment in &tle.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        if let Some(et) = tle.ephemeris_type {
            out.push_str(&format!("EPHEMERIS_TYPE = {}\n", et));
        }
        if let Some(ct) = tle.classification_type {
            out.push_str(&format!("CLASSIFICATION_TYPE = {}\n", ct));
        }
        if let Some(id) = tle.norad_cat_id {
            out.push_str(&format!("NORAD_CAT_ID = {}\n", id));
        }
        if let Some(esn) = tle.element_set_no {
            out.push_str(&format!("ELEMENT_SET_NO = {}\n", esn));
        }
        if let Some(rev) = tle.rev_at_epoch {
            out.push_str(&format!("REV_AT_EPOCH = {}\n", rev));
        }
        if let Some(bs) = tle.bstar {
            out.push_str(&format!("BSTAR = {}\n", bs));
        }
        if let Some(bt) = tle.bterm {
            out.push_str(&format!("BTERM = {}\n", bt));
        }
        if let Some(mmd) = tle.mean_motion_dot {
            out.push_str(&format!("MEAN_MOTION_DOT = {}\n", mmd));
        }
        if let Some(mmdd) = tle.mean_motion_ddot {
            out.push_str(&format!("MEAN_MOTION_DDOT = {}\n", mmdd));
        }
        if let Some(ag) = tle.agom {
            out.push_str(&format!("AGOM = {}\n", ag));
        }
    }

    // Spacecraft parameters
    write_kvn_spacecraft_params(&mut out, &omm.spacecraft_parameters);

    // Covariance (OMM uses flat key=value pairs, no COVARIANCE_START/STOP)
    if let Some(ref cov) = omm.covariance {
        out.push('\n');
        for comment in &cov.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        if let Some(ref epoch) = cov.epoch {
            out.push_str(&format!("EPOCH = {}\n", format_ccsds_datetime(epoch)));
        }
        if let Some(ref frame) = cov.cov_ref_frame {
            out.push_str(&format!("COV_REF_FRAME = {}\n", frame));
        }
        write_kvn_covariance_elements(&mut out, &cov.matrix);
    }

    // User-defined parameters
    write_kvn_user_defined(&mut out, &omm.user_defined);

    // Comments at message level
    for comment in &omm.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }

    Ok(out)
}

/// Write an OPM message to KVN format.
pub fn write_opm(opm: &crate::ccsds::opm::OPM) -> Result<String, BraheError> {
    let mut out = String::new();

    // Header
    out.push_str(&format!(
        "CCSDS_OPM_VERS = {:.1}\n",
        opm.header.format_version
    ));
    if let Some(ref class) = opm.header.classification {
        out.push_str(&format!("CLASSIFICATION = {}\n", class));
    }
    for comment in &opm.header.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }
    out.push_str(&format!(
        "CREATION_DATE = {}\n",
        format_ccsds_datetime(&opm.header.creation_date)
    ));
    out.push_str(&format!("ORIGINATOR = {}\n", opm.header.originator));
    if let Some(ref msg_id) = opm.header.message_id {
        out.push_str(&format!("MESSAGE_ID = {}\n", msg_id));
    }
    out.push('\n');

    // Metadata comments
    for comment in &opm.metadata.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }
    out.push_str(&format!("OBJECT_NAME = {}\n", opm.metadata.object_name));
    out.push_str(&format!("OBJECT_ID = {}\n", opm.metadata.object_id));
    out.push_str(&format!("CENTER_NAME = {}\n", opm.metadata.center_name));
    out.push_str(&format!("REF_FRAME = {}\n", opm.metadata.ref_frame));
    if let Some(ref epoch) = opm.metadata.ref_frame_epoch {
        out.push_str(&format!(
            "REF_FRAME_EPOCH = {}\n",
            format_ccsds_datetime(epoch)
        ));
    }
    out.push_str(&format!("TIME_SYSTEM = {}\n", opm.metadata.time_system));

    // State vector
    for comment in &opm.state_vector.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }
    out.push_str(&format!(
        "EPOCH = {}\n",
        format_ccsds_datetime(&opm.state_vector.epoch)
    ));
    // Position: m → km
    out.push_str(&format!("X = {:.6}\n", opm.state_vector.position[0] / 1e3));
    out.push_str(&format!("Y = {:.6}\n", opm.state_vector.position[1] / 1e3));
    out.push_str(&format!("Z = {:.6}\n", opm.state_vector.position[2] / 1e3));
    // Velocity: m/s → km/s
    out.push_str(&format!(
        "X_DOT = {:.6}\n",
        opm.state_vector.velocity[0] / 1e3
    ));
    out.push_str(&format!(
        "Y_DOT = {:.6}\n",
        opm.state_vector.velocity[1] / 1e3
    ));
    out.push_str(&format!(
        "Z_DOT = {:.6}\n",
        opm.state_vector.velocity[2] / 1e3
    ));

    // Keplerian elements
    if let Some(ref ke) = opm.keplerian_elements {
        for comment in &ke.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        // Semi-major axis: m → km
        out.push_str(&format!(
            "SEMI_MAJOR_AXIS = {:.6}\n",
            ke.semi_major_axis / 1e3
        ));
        out.push_str(&format!("ECCENTRICITY = {}\n", ke.eccentricity));
        out.push_str(&format!("INCLINATION = {}\n", ke.inclination));
        out.push_str(&format!("RA_OF_ASC_NODE = {}\n", ke.ra_of_asc_node));
        out.push_str(&format!("ARG_OF_PERICENTER = {}\n", ke.arg_of_pericenter));
        if let Some(ta) = ke.true_anomaly {
            out.push_str(&format!("TRUE_ANOMALY = {}\n", ta));
        }
        if let Some(ma) = ke.mean_anomaly {
            out.push_str(&format!("MEAN_ANOMALY = {}\n", ma));
        }
        if let Some(gm) = ke.gm {
            // m³/s² → km³/s²
            out.push_str(&format!("GM = {}\n", gm / 1e9));
        }
    }

    // Spacecraft parameters
    write_kvn_spacecraft_params(&mut out, &opm.spacecraft_parameters);

    // Covariance (OPM uses COVARIANCE_START/STOP like OEM)
    if let Some(ref cov) = opm.covariance {
        out.push_str("\nCOVARIANCE_START\n");
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
        out.push_str("COVARIANCE_STOP\n");
    }

    // Maneuvers
    for man in &opm.maneuvers {
        out.push('\n');
        for comment in &man.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        out.push_str(&format!(
            "MAN_EPOCH_IGNITION = {}\n",
            format_ccsds_datetime(&man.epoch_ignition)
        ));
        out.push_str(&format!("MAN_DURATION = {:.2}\n", man.duration));
        if let Some(dm) = man.delta_mass {
            out.push_str(&format!("MAN_DELTA_MASS = {:.3}\n", dm));
        }
        out.push_str(&format!("MAN_REF_FRAME = {}\n", man.ref_frame));
        // DV: m/s → km/s
        out.push_str(&format!("MAN_DV_1 = {:.8}\n", man.dv[0] / 1e3));
        out.push_str(&format!("MAN_DV_2 = {:.8}\n", man.dv[1] / 1e3));
        out.push_str(&format!("MAN_DV_3 = {:.8}\n", man.dv[2] / 1e3));
    }

    // User-defined parameters
    write_kvn_user_defined(&mut out, &opm.user_defined);

    Ok(out)
}

/// Shared helper: write spacecraft parameters to KVN output.
fn write_kvn_spacecraft_params(
    out: &mut String,
    sp: &Option<crate::ccsds::common::CCSDSSpacecraftParameters>,
) {
    if let Some(sp) = sp {
        for comment in &sp.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        if let Some(mass) = sp.mass {
            out.push_str(&format!("MASS = {:.6}\n", mass));
        }
        if let Some(sra) = sp.solar_rad_area {
            out.push_str(&format!("SOLAR_RAD_AREA = {:.6}\n", sra));
        }
        if let Some(src) = sp.solar_rad_coeff {
            out.push_str(&format!("SOLAR_RAD_COEFF = {:.6}\n", src));
        }
        if let Some(da) = sp.drag_area {
            out.push_str(&format!("DRAG_AREA = {:.6}\n", da));
        }
        if let Some(dc) = sp.drag_coeff {
            out.push_str(&format!("DRAG_COEFF = {:.6}\n", dc));
        }
    }
}

/// Shared helper: write covariance as named key=value elements (for OMM KVN).
fn write_kvn_covariance_elements(out: &mut String, matrix: &nalgebra::SMatrix<f64, 6, 6>) {
    // Convert m² → km² (factor 1e-6)
    let values = covariance_to_lower_triangular(matrix, 1e-6);
    let names = [
        "CX_X",
        "CY_X",
        "CY_Y",
        "CZ_X",
        "CZ_Y",
        "CZ_Z",
        "CX_DOT_X",
        "CX_DOT_Y",
        "CX_DOT_Z",
        "CX_DOT_X_DOT",
        "CY_DOT_X",
        "CY_DOT_Y",
        "CY_DOT_Z",
        "CY_DOT_X_DOT",
        "CY_DOT_Y_DOT",
        "CZ_DOT_X",
        "CZ_DOT_Y",
        "CZ_DOT_Z",
        "CZ_DOT_X_DOT",
        "CZ_DOT_Y_DOT",
        "CZ_DOT_Z_DOT",
    ];
    for (i, name) in names.iter().enumerate() {
        out.push_str(&format!("{} = {:.15e}\n", name, values[i]));
    }
}

/// Shared helper: write user-defined parameters to KVN output.
fn write_kvn_user_defined(out: &mut String, ud: &Option<crate::ccsds::common::CCSDSUserDefined>) {
    if let Some(ud) = ud {
        out.push('\n');
        for (k, v) in &ud.parameters {
            out.push_str(&format!("USER_DEFINED_{} = {}\n", k, v));
        }
    }
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
    use crate::ccsds::common::CDMCovarianceDimension;
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

    // ------------------------------------------------------------------
    // OEM writer: optional field coverage
    // ------------------------------------------------------------------

    #[test]
    fn test_oem_write_header_classification() {
        // OEMExample1 has CLASSIFICATION = public, test-data
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let oem = parse_oem(&content).unwrap();
        assert!(oem.header.classification.is_some());

        let written = write_oem(&oem).unwrap();
        assert!(written.contains("CLASSIFICATION = public, test-data"));
    }

    #[test]
    fn test_oem_write_header_message_id() {
        // OEMExample3 has MESSAGE_ID
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample3.txt").unwrap();
        let oem = parse_oem(&content).unwrap();
        assert!(oem.header.message_id.is_some());

        let written = write_oem(&oem).unwrap();
        assert!(written.contains("MESSAGE_ID = OEM 201113719185"));
    }

    #[test]
    fn test_oem_write_ref_frame_epoch() {
        use crate::ccsds::common::{CCSDSRefFrame, CCSDSTimeSystem, ODMHeader};
        use crate::ccsds::oem::{OEMMetadata, OEMSegment, OEMStateVector};
        use crate::time::Epoch;

        let epoch = Epoch::from_datetime(2024, 6, 1, 0, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let ref_epoch =
            Epoch::from_datetime(2000, 1, 1, 12, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let mut metadata = OEMMetadata::new(
            "REF_SAT".to_string(),
            "2024-001A".to_string(),
            "EARTH".to_string(),
            CCSDSRefFrame::TOD,
            CCSDSTimeSystem::UTC,
            epoch,
            epoch,
        );
        metadata.ref_frame_epoch = Some(ref_epoch);
        let mut seg = OEMSegment::new(metadata);
        seg.push_state(OEMStateVector::new(
            epoch,
            [7000e3, 0.0, 0.0],
            [0.0, 7500.0, 0.0],
        ));
        let oem = OEM {
            header: ODMHeader {
                format_version: 3.0,
                classification: None,
                creation_date: epoch,
                originator: "TEST".to_string(),
                message_id: None,
                comments: Vec::new(),
            },
            segments: vec![seg],
        };
        let written = write_oem(&oem).unwrap();
        assert!(written.contains("REF_FRAME_EPOCH ="));
        assert!(written.contains("2000-01-01"));
        let oem2 = parse_oem(&written).unwrap();
        assert!(oem2.segments[0].metadata.ref_frame_epoch.is_some());
    }

    #[test]
    fn test_oem_write_useable_times() {
        // OEMExample1 has USEABLE_START_TIME and USEABLE_STOP_TIME
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let oem = parse_oem(&content).unwrap();
        assert!(oem.segments[0].metadata.useable_start_time.is_some());
        assert!(oem.segments[0].metadata.useable_stop_time.is_some());

        let written = write_oem(&oem).unwrap();
        assert!(written.contains("USEABLE_START_TIME ="));
        assert!(written.contains("USEABLE_STOP_TIME ="));
    }

    #[test]
    fn test_oem_write_interpolation() {
        // OEMExample1 has INTERPOLATION and INTERPOLATION_DEGREE
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let oem = parse_oem(&content).unwrap();
        assert!(oem.segments[0].metadata.interpolation.is_some());
        assert!(oem.segments[0].metadata.interpolation_degree.is_some());

        let written = write_oem(&oem).unwrap();
        assert!(written.contains("INTERPOLATION = HERMITE"));
        assert!(written.contains("INTERPOLATION_DEGREE = 7"));
    }

    #[test]
    fn test_oem_write_acceleration() {
        use crate::ccsds::common::{CCSDSRefFrame, CCSDSTimeSystem, ODMHeader};
        use crate::ccsds::oem::{OEMMetadata, OEMSegment, OEMStateVector};
        use crate::time::Epoch;

        let epoch = Epoch::from_datetime(2024, 6, 1, 0, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let sv = OEMStateVector::new(epoch, [7000e3, 0.0, 0.0], [0.0, 7500.0, 0.0])
            .with_acceleration([0.001, -0.002, 0.003]);

        let metadata = OEMMetadata::new(
            "ACCEL_SAT".to_string(),
            "2024-001A".to_string(),
            "EARTH".to_string(),
            CCSDSRefFrame::J2000,
            CCSDSTimeSystem::UTC,
            epoch,
            epoch,
        );
        let mut seg = OEMSegment::new(metadata);
        seg.push_state(sv);

        let oem = OEM {
            header: ODMHeader {
                format_version: 3.0,
                classification: None,
                creation_date: epoch,
                originator: "TEST".to_string(),
                message_id: None,
                comments: Vec::new(),
            },
            segments: vec![seg],
        };

        let written = write_oem(&oem).unwrap();
        // Acceleration columns produce 9 space-separated values after epoch
        // Verify the line has 9 numeric columns (position, velocity, acceleration)
        let data_lines: Vec<&str> = written.lines().filter(|l| l.starts_with("2024")).collect();
        assert_eq!(data_lines.len(), 1);
        let cols: Vec<&str> = data_lines[0].split_whitespace().collect();
        // epoch + 3 pos + 3 vel + 3 acc = 10
        assert_eq!(
            cols.len(),
            10,
            "Expected 10 columns for state with acceleration"
        );

        // Round-trip: re-parse and verify acceleration survives
        let oem2 = parse_oem(&written).unwrap();
        let sv2 = &oem2.segments[0].states[0];
        assert!(sv2.acceleration.is_some());
        let acc = sv2.acceleration.unwrap();
        // Units are m/s², written as km/s², so after round-trip converted back
        assert!((acc[0] - 0.001).abs() < 1e-6);
        assert!((acc[1] - (-0.002)).abs() < 1e-6);
        assert!((acc[2] - 0.003).abs() < 1e-6);
    }

    #[test]
    fn test_oem_write_covariance_with_epoch_and_frame() {
        use crate::ccsds::common::{CCSDSCovariance, CCSDSRefFrame, CCSDSTimeSystem, ODMHeader};
        use crate::ccsds::oem::{OEMMetadata, OEMSegment, OEMStateVector};
        use crate::time::Epoch;
        use nalgebra::SMatrix;

        let epoch = Epoch::from_datetime(2024, 6, 1, 0, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);

        let metadata = OEMMetadata::new(
            "COV_SAT".to_string(),
            "2024-002A".to_string(),
            "EARTH".to_string(),
            CCSDSRefFrame::J2000,
            CCSDSTimeSystem::UTC,
            epoch,
            epoch,
        );
        let mut seg = OEMSegment::new(metadata);
        seg.push_state(OEMStateVector::new(
            epoch,
            [7000e3, 0.0, 0.0],
            [0.0, 7500.0, 0.0],
        ));

        // Add covariance with optional epoch and cov_ref_frame
        let mut matrix = SMatrix::<f64, 6, 6>::zeros();
        matrix[(0, 0)] = 1.0e6; // 1 km^2 in m^2
        matrix[(1, 1)] = 2.0e6;
        matrix[(2, 2)] = 3.0e6;
        matrix[(3, 3)] = 1.0;
        matrix[(4, 4)] = 2.0;
        matrix[(5, 5)] = 3.0;

        let cov_epoch =
            Epoch::from_datetime(2024, 6, 1, 0, 30, 0.0, 0.0, crate::time::TimeSystem::UTC);
        seg.covariances.push(CCSDSCovariance {
            epoch: Some(cov_epoch),
            cov_ref_frame: Some(CCSDSRefFrame::RTN),
            matrix,
            comments: vec!["Test covariance comment".to_string()],
        });

        let oem = OEM {
            header: ODMHeader {
                format_version: 3.0,
                classification: None,
                creation_date: epoch,
                originator: "TEST".to_string(),
                message_id: None,
                comments: Vec::new(),
            },
            segments: vec![seg],
        };

        let written = write_oem(&oem).unwrap();
        assert!(written.contains("COVARIANCE_START"));
        assert!(written.contains("COVARIANCE_STOP"));
        assert!(written.contains("EPOCH ="));
        assert!(written.contains("COV_REF_FRAME = RTN"));
        assert!(written.contains("COMMENT Test covariance comment"));
    }

    #[test]
    fn test_oem_write_data_block_comments() {
        use crate::ccsds::common::{CCSDSRefFrame, CCSDSTimeSystem, ODMHeader};
        use crate::ccsds::oem::{OEMMetadata, OEMSegment, OEMStateVector};
        use crate::time::Epoch;

        let epoch = Epoch::from_datetime(2024, 6, 1, 0, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);

        let metadata = OEMMetadata::new(
            "CMT_SAT".to_string(),
            "2024-003A".to_string(),
            "EARTH".to_string(),
            CCSDSRefFrame::J2000,
            CCSDSTimeSystem::UTC,
            epoch,
            epoch,
        );
        let mut seg = OEMSegment::new(metadata);
        seg.comments = vec![
            "Data block comment line 1".to_string(),
            "Data block comment line 2".to_string(),
        ];
        seg.push_state(OEMStateVector::new(
            epoch,
            [7000e3, 0.0, 0.0],
            [0.0, 7500.0, 0.0],
        ));

        let oem = OEM {
            header: ODMHeader {
                format_version: 3.0,
                classification: None,
                creation_date: epoch,
                originator: "TEST".to_string(),
                message_id: None,
                comments: Vec::new(),
            },
            segments: vec![seg],
        };

        let written = write_oem(&oem).unwrap();
        assert!(written.contains("COMMENT Data block comment line 1"));
        assert!(written.contains("COMMENT Data block comment line 2"));
    }

    #[test]
    fn test_oem_write_all_optional_fields_round_trip() {
        // Build OEM with all optional metadata fields set
        use crate::ccsds::common::{CCSDSRefFrame, CCSDSTimeSystem, ODMHeader};
        use crate::ccsds::oem::{OEMMetadata, OEMSegment, OEMStateVector};
        use crate::time::Epoch;

        let epoch = Epoch::from_datetime(2024, 6, 1, 0, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let ref_epoch =
            Epoch::from_datetime(2000, 1, 1, 12, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let mut metadata = OEMMetadata::new(
            "ALL_OPT".to_string(),
            "2024-001A".to_string(),
            "EARTH".to_string(),
            CCSDSRefFrame::TOD,
            CCSDSTimeSystem::UTC,
            epoch,
            epoch,
        );
        metadata.ref_frame_epoch = Some(ref_epoch);
        metadata.useable_start_time = Some(epoch);
        metadata.useable_stop_time = Some(epoch);
        metadata.interpolation = Some("HERMITE".to_string());
        metadata.interpolation_degree = Some(7);

        let mut seg = OEMSegment::new(metadata);
        seg.push_state(OEMStateVector::new(
            epoch,
            [7000e3, 0.0, 0.0],
            [0.0, 7500.0, 0.0],
        ));

        let oem = OEM {
            header: ODMHeader {
                format_version: 3.0,
                classification: None,
                creation_date: epoch,
                originator: "TEST".to_string(),
                message_id: None,
                comments: Vec::new(),
            },
            segments: vec![seg],
        };

        let written = write_oem(&oem).unwrap();
        let oem2 = parse_oem(&written).unwrap();

        let seg = &oem2.segments[0];
        assert!(seg.metadata.ref_frame_epoch.is_some());
        assert!(seg.metadata.useable_start_time.is_some());
        assert!(seg.metadata.useable_stop_time.is_some());
        assert_eq!(seg.metadata.interpolation.as_deref(), Some("HERMITE"));
        assert_eq!(seg.metadata.interpolation_degree, Some(7));
        assert_eq!(seg.metadata.ref_frame, CCSDSRefFrame::TOD);
    }

    // ------------------------------------------------------------------
    // CDM writer: round-trip tests with test assets
    // ------------------------------------------------------------------

    #[test]
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
