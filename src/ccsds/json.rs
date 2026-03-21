/*!
 * CCSDS JSON format support for OEM, OMM, OPM, and CDM messages.
 *
 * JSON parsing works by flattening JSON objects into KVN-style key=value lines
 * and delegating to the existing KVN parsers. This avoids duplicating validation
 * logic. JSON writing builds structured JSON objects using serde_json.
 */

use serde_json::{Map, Value, json};

use crate::ccsds::common::{
    CCSDSJsonKeyCase, covariance_to_lower_triangular, format_ccsds_datetime,
};
use crate::ccsds::error::ccsds_parse_error;
use crate::utils::errors::BraheError;

/// Convert a CCSDS keyword to the appropriate case for JSON output.
///
/// Container/structural keys should NOT use this function — they are always lowercase.
/// This function only applies to CCSDS data field keywords.
fn key(name: &str, case: CCSDSJsonKeyCase) -> String {
    match case {
        CCSDSJsonKeyCase::Lower => name.to_lowercase(),
        CCSDSJsonKeyCase::Upper => name.to_uppercase(),
    }
}

// =============================================================================
// OEM JSON
// =============================================================================

/// Parse an OEM message from JSON format.
///
/// Flattens the JSON structure into KVN-style lines with META_START/META_STOP
/// delimiters for each segment, then delegates to the KVN parser.
pub fn parse_oem_json(content: &str) -> Result<crate::ccsds::oem::OEM, BraheError> {
    let v: Value = serde_json::from_str(content)
        .map_err(|e| ccsds_parse_error("OEM", &format!("JSON parse error: {}", e)))?;

    let mut kvn_lines: Vec<String> = Vec::new();

    // Header
    if let Some(header) = v.get("header") {
        flatten_object(&mut kvn_lines, header);
    }

    // Segments
    if let Some(Value::Array(segments)) = v.get("segments") {
        for seg in segments {
            // Metadata block
            kvn_lines.push("META_START".to_string());
            if let Some(meta) = seg.get("metadata") {
                flatten_object(&mut kvn_lines, meta);
            }
            kvn_lines.push("META_STOP".to_string());

            // State vectors (data lines)
            if let Some(Value::Array(states)) = seg.get("states") {
                for state in states {
                    if let Some(obj) = state.as_object() {
                        let epoch = obj
                            .get("EPOCH")
                            .or_else(|| obj.get("epoch"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let x = get_json_f64(obj, "X").or_else(|| get_json_f64(obj, "x"));
                        let y = get_json_f64(obj, "Y").or_else(|| get_json_f64(obj, "y"));
                        let z = get_json_f64(obj, "Z").or_else(|| get_json_f64(obj, "z"));
                        let vx = get_json_f64(obj, "X_DOT").or_else(|| get_json_f64(obj, "x_dot"));
                        let vy = get_json_f64(obj, "Y_DOT").or_else(|| get_json_f64(obj, "y_dot"));
                        let vz = get_json_f64(obj, "Z_DOT").or_else(|| get_json_f64(obj, "z_dot"));

                        if let (Some(x), Some(y), Some(z), Some(vx), Some(vy), Some(vz)) =
                            (x, y, z, vx, vy, vz)
                        {
                            let ax =
                                get_json_f64(obj, "X_DDOT").or_else(|| get_json_f64(obj, "x_ddot"));
                            let ay =
                                get_json_f64(obj, "Y_DDOT").or_else(|| get_json_f64(obj, "y_ddot"));
                            let az =
                                get_json_f64(obj, "Z_DDOT").or_else(|| get_json_f64(obj, "z_ddot"));

                            if let (Some(ax), Some(ay), Some(az)) = (ax, ay, az) {
                                kvn_lines.push(format!(
                                    "{} {} {} {} {} {} {} {} {} {}",
                                    epoch, x, y, z, vx, vy, vz, ax, ay, az
                                ));
                            } else {
                                kvn_lines.push(format!(
                                    "{} {} {} {} {} {} {}",
                                    epoch, x, y, z, vx, vy, vz
                                ));
                            }
                        }
                    }
                }
            }

            // Covariance blocks
            if let Some(Value::Array(covariances)) = seg.get("covariances") {
                kvn_lines.push("COVARIANCE_START".to_string());
                for cov in covariances {
                    if let Some(obj) = cov.as_object() {
                        // Epoch
                        if let Some(epoch_val) = obj
                            .get("EPOCH")
                            .or_else(|| obj.get("epoch"))
                            .and_then(|v| v.as_str())
                        {
                            kvn_lines.push(format!("EPOCH = {}", epoch_val));
                        }
                        // COV_REF_FRAME
                        if let Some(frame) = obj
                            .get("COV_REF_FRAME")
                            .or_else(|| obj.get("cov_ref_frame"))
                            .and_then(|v| v.as_str())
                        {
                            kvn_lines.push(format!("COV_REF_FRAME = {}", frame));
                        }
                        // Lower-triangular values
                        if let Some(Value::Array(values)) =
                            obj.get("VALUES").or_else(|| obj.get("values"))
                        {
                            // values is array of rows: each row is an array of f64
                            for row in values {
                                if let Value::Array(row_vals) = row {
                                    let nums: Vec<String> = row_vals
                                        .iter()
                                        .filter_map(|v| v.as_f64().map(|f| format!("{:.10e}", f)))
                                        .collect();
                                    if !nums.is_empty() {
                                        kvn_lines.push(nums.join(" "));
                                    }
                                }
                            }
                        }
                    }
                }
                kvn_lines.push("COVARIANCE_STOP".to_string());
            }
        }
    }

    let kvn_content = kvn_lines.join("\n");
    crate::ccsds::kvn::parse_oem(&kvn_content)
}

/// Write an OEM message to JSON format.
pub fn write_oem_json(
    oem: &crate::ccsds::oem::OEM,
    key_case: CCSDSJsonKeyCase,
) -> Result<String, BraheError> {
    let mut root = Map::new();

    // Header
    let mut header = Map::new();
    header.insert(
        key("CCSDS_OEM_VERS", key_case),
        json!(oem.header.format_version),
    );
    if let Some(ref class) = oem.header.classification {
        header.insert(key("CLASSIFICATION", key_case), json!(class));
    }
    header.insert(
        key("CREATION_DATE", key_case),
        json!(format_ccsds_datetime(&oem.header.creation_date)),
    );
    header.insert(key("ORIGINATOR", key_case), json!(&oem.header.originator));
    if let Some(ref msg_id) = oem.header.message_id {
        header.insert(key("MESSAGE_ID", key_case), json!(msg_id));
    }
    root.insert("header".into(), Value::Object(header));

    // Segments
    let mut segments = Vec::new();
    for seg in &oem.segments {
        let mut seg_obj = Map::new();

        // Metadata
        let mut meta = Map::new();
        meta.insert(
            key("OBJECT_NAME", key_case),
            json!(&seg.metadata.object_name),
        );
        meta.insert(key("OBJECT_ID", key_case), json!(&seg.metadata.object_id));
        meta.insert(
            key("CENTER_NAME", key_case),
            json!(&seg.metadata.center_name),
        );
        meta.insert(
            key("REF_FRAME", key_case),
            json!(format!("{}", seg.metadata.ref_frame)),
        );
        if let Some(ref epoch) = seg.metadata.ref_frame_epoch {
            meta.insert(
                key("REF_FRAME_EPOCH", key_case),
                json!(format_ccsds_datetime(epoch)),
            );
        }
        meta.insert(
            key("TIME_SYSTEM", key_case),
            json!(format!("{}", seg.metadata.time_system)),
        );
        meta.insert(
            key("START_TIME", key_case),
            json!(format_ccsds_datetime(&seg.metadata.start_time)),
        );
        if let Some(ref t) = seg.metadata.useable_start_time {
            meta.insert(
                key("USEABLE_START_TIME", key_case),
                json!(format_ccsds_datetime(t)),
            );
        }
        if let Some(ref t) = seg.metadata.useable_stop_time {
            meta.insert(
                key("USEABLE_STOP_TIME", key_case),
                json!(format_ccsds_datetime(t)),
            );
        }
        meta.insert(
            key("STOP_TIME", key_case),
            json!(format_ccsds_datetime(&seg.metadata.stop_time)),
        );
        if let Some(ref interp) = seg.metadata.interpolation {
            meta.insert(key("INTERPOLATION", key_case), json!(interp));
        }
        if let Some(deg) = seg.metadata.interpolation_degree {
            meta.insert(key("INTERPOLATION_DEGREE", key_case), json!(deg));
        }
        seg_obj.insert("metadata".into(), Value::Object(meta));

        // States (convert m → km, m/s → km/s for CCSDS standard units)
        let mut states = Vec::new();
        for sv in &seg.states {
            let mut state_obj = Map::new();
            state_obj.insert(
                key("EPOCH", key_case),
                json!(format_ccsds_datetime(&sv.epoch)),
            );
            state_obj.insert(key("X", key_case), json!(sv.position[0] / 1000.0));
            state_obj.insert(key("Y", key_case), json!(sv.position[1] / 1000.0));
            state_obj.insert(key("Z", key_case), json!(sv.position[2] / 1000.0));
            state_obj.insert(key("X_DOT", key_case), json!(sv.velocity[0] / 1000.0));
            state_obj.insert(key("Y_DOT", key_case), json!(sv.velocity[1] / 1000.0));
            state_obj.insert(key("Z_DOT", key_case), json!(sv.velocity[2] / 1000.0));
            if let Some(ref acc) = sv.acceleration {
                state_obj.insert(key("X_DDOT", key_case), json!(acc[0] / 1000.0));
                state_obj.insert(key("Y_DDOT", key_case), json!(acc[1] / 1000.0));
                state_obj.insert(key("Z_DDOT", key_case), json!(acc[2] / 1000.0));
            }
            states.push(Value::Object(state_obj));
        }
        seg_obj.insert("states".into(), Value::Array(states));

        // Covariances
        if !seg.covariances.is_empty() {
            let mut covs = Vec::new();
            for cov in &seg.covariances {
                let mut cov_obj = Map::new();
                if let Some(ref epoch) = cov.epoch {
                    cov_obj.insert(key("EPOCH", key_case), json!(format_ccsds_datetime(epoch)));
                }
                if let Some(ref frame) = cov.cov_ref_frame {
                    cov_obj.insert(key("COV_REF_FRAME", key_case), json!(format!("{}", frame)));
                }
                // Convert m² → km² (factor 1e-6)
                let values = covariance_to_lower_triangular(&cov.matrix, 1e-6);
                let mut rows: Vec<Value> = Vec::new();
                let mut idx = 0;
                for row in 0..6 {
                    let row_vals: Vec<Value> = (0..=row)
                        .map(|_| {
                            let v = values[idx];
                            idx += 1;
                            json!(v)
                        })
                        .collect();
                    rows.push(Value::Array(row_vals));
                }
                cov_obj.insert("values".into(), Value::Array(rows));
                covs.push(Value::Object(cov_obj));
            }
            seg_obj.insert("covariances".into(), Value::Array(covs));
        }

        segments.push(Value::Object(seg_obj));
    }
    root.insert("segments".into(), Value::Array(segments));

    serde_json::to_string_pretty(&Value::Object(root))
        .map_err(|e| BraheError::Error(format!("OEM JSON serialization error: {}", e)))
}

// =============================================================================
// OMM JSON
// =============================================================================

/// Parse an OMM message from JSON format.
///
/// OMM is flat (no META_START/STOP). Flattens JSON to KVN lines and delegates
/// to the KVN parser.
pub fn parse_omm_json(content: &str) -> Result<crate::ccsds::omm::OMM, BraheError> {
    let v: Value = serde_json::from_str(content)
        .map_err(|e| ccsds_parse_error("OMM", &format!("JSON parse error: {}", e)))?;

    let mut kvn_lines: Vec<String> = Vec::new();

    // Process sections in KVN order (serde_json sorts alphabetically, which
    // would break KVN parser expectations)
    let ordered_sections = [
        "header",
        "metadata",
        "mean_elements",
        "tle_parameters",
        "spacecraft_parameters",
        "covariance",
        "user_defined",
    ];

    // Covariance keys must be in lower-triangular order for the KVN parser
    let cov_key_order = [
        "COV_REF_FRAME",
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

    for section in &ordered_sections {
        if let Some(obj) = v.get(*section).or_else(|| v.get(section.to_uppercase())) {
            if *section == "covariance" {
                flatten_object_ordered(&mut kvn_lines, obj, &cov_key_order);
            } else {
                flatten_object(&mut kvn_lines, obj);
            }
        }
    }

    let kvn_content = kvn_lines.join("\n");
    crate::ccsds::kvn::parse_omm(&kvn_content)
}

/// Write an OMM message to JSON format.
pub fn write_omm_json(
    omm: &crate::ccsds::omm::OMM,
    key_case: CCSDSJsonKeyCase,
) -> Result<String, BraheError> {
    let mut root = Map::new();

    // Header
    let mut header = Map::new();
    header.insert(
        key("CCSDS_OMM_VERS", key_case),
        json!(omm.header.format_version),
    );
    if let Some(ref class) = omm.header.classification {
        header.insert(key("CLASSIFICATION", key_case), json!(class));
    }
    header.insert(
        key("CREATION_DATE", key_case),
        json!(format_ccsds_datetime(&omm.header.creation_date)),
    );
    header.insert(key("ORIGINATOR", key_case), json!(&omm.header.originator));
    if let Some(ref msg_id) = omm.header.message_id {
        header.insert(key("MESSAGE_ID", key_case), json!(msg_id));
    }
    root.insert("header".into(), Value::Object(header));

    // Metadata
    let mut meta = Map::new();
    meta.insert(
        key("OBJECT_NAME", key_case),
        json!(&omm.metadata.object_name),
    );
    meta.insert(key("OBJECT_ID", key_case), json!(&omm.metadata.object_id));
    meta.insert(
        key("CENTER_NAME", key_case),
        json!(&omm.metadata.center_name),
    );
    meta.insert(
        key("REF_FRAME", key_case),
        json!(format!("{}", omm.metadata.ref_frame)),
    );
    if let Some(ref epoch) = omm.metadata.ref_frame_epoch {
        meta.insert(
            key("REF_FRAME_EPOCH", key_case),
            json!(format_ccsds_datetime(epoch)),
        );
    }
    meta.insert(
        key("TIME_SYSTEM", key_case),
        json!(format!("{}", omm.metadata.time_system)),
    );
    meta.insert(
        key("MEAN_ELEMENT_THEORY", key_case),
        json!(&omm.metadata.mean_element_theory),
    );
    root.insert("metadata".into(), Value::Object(meta));

    // Mean elements (units stored as file-native: rev/day, degrees, km)
    let mut me = Map::new();
    me.insert(
        key("EPOCH", key_case),
        json!(format_ccsds_datetime(&omm.mean_elements.epoch)),
    );
    if let Some(v) = omm.mean_elements.mean_motion {
        me.insert(key("MEAN_MOTION", key_case), json!(v));
    }
    if let Some(v) = omm.mean_elements.semi_major_axis {
        me.insert(key("SEMI_MAJOR_AXIS", key_case), json!(v));
    }
    me.insert(
        key("ECCENTRICITY", key_case),
        json!(omm.mean_elements.eccentricity),
    );
    me.insert(
        key("INCLINATION", key_case),
        json!(omm.mean_elements.inclination),
    );
    me.insert(
        key("RA_OF_ASC_NODE", key_case),
        json!(omm.mean_elements.ra_of_asc_node),
    );
    me.insert(
        key("ARG_OF_PERICENTER", key_case),
        json!(omm.mean_elements.arg_of_pericenter),
    );
    me.insert(
        key("MEAN_ANOMALY", key_case),
        json!(omm.mean_elements.mean_anomaly),
    );
    if let Some(v) = omm.mean_elements.gm {
        // GM stored internally as m³/s², write as km³/s²
        me.insert(key("GM", key_case), json!(v / 1e9));
    }
    root.insert("mean_elements".into(), Value::Object(me));

    // TLE parameters
    if let Some(ref tle) = omm.tle_parameters {
        let mut tp = Map::new();
        if let Some(v) = tle.ephemeris_type {
            tp.insert(key("EPHEMERIS_TYPE", key_case), json!(v));
        }
        if let Some(v) = tle.classification_type {
            tp.insert(key("CLASSIFICATION_TYPE", key_case), json!(v.to_string()));
        }
        if let Some(v) = tle.norad_cat_id {
            tp.insert(key("NORAD_CAT_ID", key_case), json!(v));
        }
        if let Some(v) = tle.element_set_no {
            tp.insert(key("ELEMENT_SET_NO", key_case), json!(v));
        }
        if let Some(v) = tle.rev_at_epoch {
            tp.insert(key("REV_AT_EPOCH", key_case), json!(v));
        }
        if let Some(v) = tle.bstar {
            tp.insert(key("BSTAR", key_case), json!(v));
        }
        if let Some(v) = tle.bterm {
            tp.insert(key("BTERM", key_case), json!(v));
        }
        if let Some(v) = tle.mean_motion_dot {
            tp.insert(key("MEAN_MOTION_DOT", key_case), json!(v));
        }
        if let Some(v) = tle.mean_motion_ddot {
            tp.insert(key("MEAN_MOTION_DDOT", key_case), json!(v));
        }
        if let Some(v) = tle.agom {
            tp.insert(key("AGOM", key_case), json!(v));
        }
        root.insert("tle_parameters".into(), Value::Object(tp));
    }

    // Spacecraft parameters
    if let Some(ref sc) = omm.spacecraft_parameters {
        let mut sp = Map::new();
        if let Some(v) = sc.mass {
            sp.insert(key("MASS", key_case), json!(v));
        }
        if let Some(v) = sc.solar_rad_area {
            sp.insert(key("SOLAR_RAD_AREA", key_case), json!(v));
        }
        if let Some(v) = sc.solar_rad_coeff {
            sp.insert(key("SOLAR_RAD_COEFF", key_case), json!(v));
        }
        if let Some(v) = sc.drag_area {
            sp.insert(key("DRAG_AREA", key_case), json!(v));
        }
        if let Some(v) = sc.drag_coeff {
            sp.insert(key("DRAG_COEFF", key_case), json!(v));
        }
        root.insert("spacecraft_parameters".into(), Value::Object(sp));
    }

    // Covariance
    if let Some(ref cov) = omm.covariance {
        let mut cv = Map::new();
        if let Some(ref frame) = cov.cov_ref_frame {
            cv.insert(key("COV_REF_FRAME", key_case), json!(format!("{}", frame)));
        }
        write_json_covariance_elements(&mut cv, &cov.matrix, key_case);
        root.insert("covariance".into(), Value::Object(cv));
    }

    // User-defined
    if let Some(ref ud) = omm.user_defined {
        let mut ud_obj = Map::new();
        for (k, v) in &ud.parameters {
            ud_obj.insert(format!("USER_DEFINED_{}", k), json!(v));
        }
        root.insert("user_defined".into(), Value::Object(ud_obj));
    }

    serde_json::to_string_pretty(&Value::Object(root))
        .map_err(|e| BraheError::Error(format!("OMM JSON serialization error: {}", e)))
}

// =============================================================================
// OPM JSON
// =============================================================================

/// Parse an OPM message from JSON format.
///
/// OPM is flat (like OMM). Flattens JSON to KVN lines and delegates to the
/// KVN parser. Maneuvers array flattens sequentially (KVN parser uses
/// MAN_EPOCH_IGNITION as delimiter).
pub fn parse_opm_json(content: &str) -> Result<crate::ccsds::opm::OPM, BraheError> {
    let v: Value = serde_json::from_str(content)
        .map_err(|e| ccsds_parse_error("OPM", &format!("JSON parse error: {}", e)))?;

    let mut kvn_lines: Vec<String> = Vec::new();

    // Process sections in KVN order (serde_json sorts alphabetically, which
    // would break KVN parser expectations)
    let ordered_sections = [
        "header",
        "metadata",
        "state_vector",
        "keplerian_elements",
        "spacecraft_parameters",
        "covariance",
        "user_defined",
    ];

    // KVN parser requires SEMI_MAJOR_AXIS before INCLINATION/RA_OF_ASC_NODE/ARG_OF_PERICENTER
    let kep_key_order = [
        "SEMI_MAJOR_AXIS",
        "ECCENTRICITY",
        "INCLINATION",
        "RA_OF_ASC_NODE",
        "ARG_OF_PERICENTER",
        "TRUE_ANOMALY",
        "MEAN_ANOMALY",
        "GM",
    ];

    // Covariance keys must be in lower-triangular order for the KVN parser
    let cov_key_order = [
        "COV_REF_FRAME",
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

    for section in &ordered_sections {
        if let Some(obj) = v.get(*section).or_else(|| v.get(section.to_uppercase())) {
            if *section == "keplerian_elements" {
                flatten_object_ordered(&mut kvn_lines, obj, &kep_key_order);
            } else if *section == "covariance" {
                flatten_object_ordered(&mut kvn_lines, obj, &cov_key_order);
            } else {
                flatten_object(&mut kvn_lines, obj);
            }
        }
    }

    // Maneuvers array — MAN_EPOCH_IGNITION must come first (KVN parser uses
    // it as a delimiter to flush the previous maneuver)
    let man_key_order = [
        "MAN_EPOCH_IGNITION",
        "MAN_DURATION",
        "MAN_DELTA_MASS",
        "MAN_REF_FRAME",
        "MAN_DV_1",
        "MAN_DV_2",
        "MAN_DV_3",
    ];

    if let Some(Value::Array(maneuvers)) = v.get("maneuvers").or_else(|| v.get("MANEUVERS")) {
        for man in maneuvers {
            flatten_object_ordered(&mut kvn_lines, man, &man_key_order);
        }
    }

    let kvn_content = kvn_lines.join("\n");
    crate::ccsds::kvn::parse_opm(&kvn_content)
}

/// Write an OPM message to JSON format.
pub fn write_opm_json(
    opm: &crate::ccsds::opm::OPM,
    key_case: CCSDSJsonKeyCase,
) -> Result<String, BraheError> {
    let mut root = Map::new();

    // Header
    let mut header = Map::new();
    header.insert(
        key("CCSDS_OPM_VERS", key_case),
        json!(opm.header.format_version),
    );
    if let Some(ref class) = opm.header.classification {
        header.insert(key("CLASSIFICATION", key_case), json!(class));
    }
    header.insert(
        key("CREATION_DATE", key_case),
        json!(format_ccsds_datetime(&opm.header.creation_date)),
    );
    header.insert(key("ORIGINATOR", key_case), json!(&opm.header.originator));
    if let Some(ref msg_id) = opm.header.message_id {
        header.insert(key("MESSAGE_ID", key_case), json!(msg_id));
    }
    root.insert("header".into(), Value::Object(header));

    // Metadata
    let mut meta = Map::new();
    meta.insert(
        key("OBJECT_NAME", key_case),
        json!(&opm.metadata.object_name),
    );
    meta.insert(key("OBJECT_ID", key_case), json!(&opm.metadata.object_id));
    meta.insert(
        key("CENTER_NAME", key_case),
        json!(&opm.metadata.center_name),
    );
    meta.insert(
        key("REF_FRAME", key_case),
        json!(format!("{}", opm.metadata.ref_frame)),
    );
    if let Some(ref epoch) = opm.metadata.ref_frame_epoch {
        meta.insert(
            key("REF_FRAME_EPOCH", key_case),
            json!(format_ccsds_datetime(epoch)),
        );
    }
    meta.insert(
        key("TIME_SYSTEM", key_case),
        json!(format!("{}", opm.metadata.time_system)),
    );
    root.insert("metadata".into(), Value::Object(meta));

    // State vector (convert m → km, m/s → km/s)
    let mut sv = Map::new();
    sv.insert(
        key("EPOCH", key_case),
        json!(format_ccsds_datetime(&opm.state_vector.epoch)),
    );
    sv.insert(
        key("X", key_case),
        json!(opm.state_vector.position[0] / 1000.0),
    );
    sv.insert(
        key("Y", key_case),
        json!(opm.state_vector.position[1] / 1000.0),
    );
    sv.insert(
        key("Z", key_case),
        json!(opm.state_vector.position[2] / 1000.0),
    );
    sv.insert(
        key("X_DOT", key_case),
        json!(opm.state_vector.velocity[0] / 1000.0),
    );
    sv.insert(
        key("Y_DOT", key_case),
        json!(opm.state_vector.velocity[1] / 1000.0),
    );
    sv.insert(
        key("Z_DOT", key_case),
        json!(opm.state_vector.velocity[2] / 1000.0),
    );
    root.insert("state_vector".into(), Value::Object(sv));

    // Keplerian elements
    if let Some(ref kep) = opm.keplerian_elements {
        let mut ke = Map::new();
        ke.insert(
            key("SEMI_MAJOR_AXIS", key_case),
            json!(kep.semi_major_axis / 1000.0), // m → km
        );
        ke.insert(key("ECCENTRICITY", key_case), json!(kep.eccentricity));
        ke.insert(key("INCLINATION", key_case), json!(kep.inclination));
        ke.insert(key("RA_OF_ASC_NODE", key_case), json!(kep.ra_of_asc_node));
        ke.insert(
            key("ARG_OF_PERICENTER", key_case),
            json!(kep.arg_of_pericenter),
        );
        if let Some(v) = kep.true_anomaly {
            ke.insert(key("TRUE_ANOMALY", key_case), json!(v));
        }
        if let Some(v) = kep.mean_anomaly {
            ke.insert(key("MEAN_ANOMALY", key_case), json!(v));
        }
        if let Some(v) = kep.gm {
            ke.insert(key("GM", key_case), json!(v / 1e9)); // m³/s² → km³/s²
        }
        root.insert("keplerian_elements".into(), Value::Object(ke));
    }

    // Spacecraft parameters
    if let Some(ref sc) = opm.spacecraft_parameters {
        let mut sp = Map::new();
        if let Some(v) = sc.mass {
            sp.insert(key("MASS", key_case), json!(v));
        }
        if let Some(v) = sc.solar_rad_area {
            sp.insert(key("SOLAR_RAD_AREA", key_case), json!(v));
        }
        if let Some(v) = sc.solar_rad_coeff {
            sp.insert(key("SOLAR_RAD_COEFF", key_case), json!(v));
        }
        if let Some(v) = sc.drag_area {
            sp.insert(key("DRAG_AREA", key_case), json!(v));
        }
        if let Some(v) = sc.drag_coeff {
            sp.insert(key("DRAG_COEFF", key_case), json!(v));
        }
        root.insert("spacecraft_parameters".into(), Value::Object(sp));
    }

    // Covariance
    if let Some(ref cov) = opm.covariance {
        let mut cv = Map::new();
        if let Some(ref frame) = cov.cov_ref_frame {
            cv.insert(key("COV_REF_FRAME", key_case), json!(format!("{}", frame)));
        }
        write_json_covariance_elements(&mut cv, &cov.matrix, key_case);
        root.insert("covariance".into(), Value::Object(cv));
    }

    // Maneuvers (convert m/s → km/s)
    if !opm.maneuvers.is_empty() {
        let mut mans = Vec::new();
        for man in &opm.maneuvers {
            let mut m = Map::new();
            m.insert(
                key("MAN_EPOCH_IGNITION", key_case),
                json!(format_ccsds_datetime(&man.epoch_ignition)),
            );
            m.insert(key("MAN_DURATION", key_case), json!(man.duration));
            if let Some(dm) = man.delta_mass {
                m.insert(key("MAN_DELTA_MASS", key_case), json!(dm));
            }
            m.insert(
                key("MAN_REF_FRAME", key_case),
                json!(format!("{}", man.ref_frame)),
            );
            m.insert(key("MAN_DV_1", key_case), json!(man.dv[0] / 1000.0));
            m.insert(key("MAN_DV_2", key_case), json!(man.dv[1] / 1000.0));
            m.insert(key("MAN_DV_3", key_case), json!(man.dv[2] / 1000.0));
            mans.push(Value::Object(m));
        }
        root.insert("maneuvers".into(), Value::Array(mans));
    }

    // User-defined
    if let Some(ref ud) = opm.user_defined {
        let mut ud_obj = Map::new();
        for (k, v) in &ud.parameters {
            ud_obj.insert(format!("USER_DEFINED_{}", k), json!(v));
        }
        root.insert("user_defined".into(), Value::Object(ud_obj));
    }

    serde_json::to_string_pretty(&Value::Object(root))
        .map_err(|e| BraheError::Error(format!("OPM JSON serialization error: {}", e)))
}

// =============================================================================
// CDM JSON
// =============================================================================

/// Parse a CDM message from JSON format.
///
/// Deserializes a JSON object into key-value pairs, reconstructs KVN-like
/// representation, and delegates to the KVN parser.
pub fn parse_cdm_json(content: &str) -> Result<crate::ccsds::cdm::CDM, BraheError> {
    let v: Value = serde_json::from_str(content)
        .map_err(|e| ccsds_parse_error("CDM", &format!("JSON parse error: {}", e)))?;

    let mut kvn_lines: Vec<String> = Vec::new();

    // Recursively flatten JSON objects into KVN key=value lines.
    fn flatten(lines: &mut Vec<String>, obj: &Value) {
        if let Value::Object(map) = obj {
            for (key, val) in map {
                let ukey = key.to_uppercase();

                let is_container = matches!(
                    ukey.as_str(),
                    "HEADER"
                        | "RELATIVE_METADATA"
                        | "OBJECT1"
                        | "OBJECT2"
                        | "METADATA"
                        | "STATE_VECTOR"
                        | "RTN_COVARIANCE"
                        | "OD_PARAMETERS"
                        | "ADDITIONAL_PARAMETERS"
                        | "XYZ_COVARIANCE"
                        | "ADDITIONAL_COVARIANCE_METADATA"
                        | "USER_DEFINED"
                );

                match val {
                    Value::Object(_) if is_container => {
                        flatten(lines, val);
                    }
                    Value::Object(_) => {
                        flatten(lines, val);
                    }
                    Value::Null => {}
                    Value::String(s) => {
                        lines.push(format!("{} = {}", ukey, s));
                    }
                    Value::Number(n) => {
                        lines.push(format!("{} = {}", ukey, n));
                    }
                    Value::Array(arr) => {
                        let parts: Vec<String> = arr.iter().map(|a| a.to_string()).collect();
                        lines.push(format!("{} = {}", ukey, parts.join(" ")));
                    }
                    Value::Bool(b) => {
                        lines.push(format!("{} = {}", ukey, if *b { "YES" } else { "NO" }));
                    }
                }
            }
        }
    }

    // Helper to flatten a CDM object section in correct order.
    //
    // Handles both lowercase and UPPERCASE keys (produced by different key_case
    // settings). OBJECT must come first since the KVN parser uses it as a
    // section delimiter, and we must not emit it twice.
    fn flatten_cdm_object(lines: &mut Vec<String>, obj: &Value) {
        if let Some(meta) = obj.get("metadata").or_else(|| obj.get("METADATA")) {
            // OBJECT must be emitted first (KVN delimiter)
            if let Some(s) = meta
                .get("OBJECT")
                .or_else(|| meta.get("object"))
                .and_then(|v| v.as_str())
            {
                lines.push(format!("OBJECT = {}", s));
            }
            // Emit remaining metadata keys, skipping OBJECT to avoid duplicate
            if let Value::Object(map) = meta {
                for (k, val) in map {
                    let ukey = k.to_uppercase();
                    if ukey == "OBJECT" {
                        continue;
                    }
                    match val {
                        Value::Null => {}
                        Value::String(s) => lines.push(format!("{} = {}", ukey, s)),
                        Value::Number(n) => lines.push(format!("{} = {}", ukey, n)),
                        Value::Bool(b) => {
                            lines.push(format!("{} = {}", ukey, if *b { "YES" } else { "NO" }))
                        }
                        _ => {}
                    }
                }
            }
        }
        if let Some(sv) = obj.get("state_vector").or_else(|| obj.get("STATE_VECTOR")) {
            flatten(lines, sv);
        }
        if let Some(Value::Array(arr)) = obj
            .get("rtn_covariance_ordered")
            .or_else(|| obj.get("RTN_COVARIANCE_ORDERED"))
        {
            for pair in arr {
                let Some(kv) = pair.as_array() else { continue };
                let Some(Value::String(key)) = kv.first() else {
                    continue;
                };
                let Some(val) = kv.get(1).and_then(|v| v.as_f64()) else {
                    continue;
                };
                lines.push(format!("{} = {}", key, val));
            }
        }
        if let Some(od) = obj
            .get("od_parameters")
            .or_else(|| obj.get("OD_PARAMETERS"))
        {
            flatten(lines, od);
        }
        if let Some(ap) = obj
            .get("additional_parameters")
            .or_else(|| obj.get("ADDITIONAL_PARAMETERS"))
        {
            flatten(lines, ap);
        }
    }

    if let Some(header) = v.get("header").or_else(|| v.get("HEADER")) {
        flatten(&mut kvn_lines, header);
    }
    if let Some(rel) = v
        .get("relative_metadata")
        .or_else(|| v.get("RELATIVE_METADATA"))
    {
        flatten(&mut kvn_lines, rel);
    }
    if let Some(obj1) = v.get("object1").or_else(|| v.get("OBJECT1")) {
        flatten_cdm_object(&mut kvn_lines, obj1);
    }
    if let Some(obj2) = v.get("object2").or_else(|| v.get("OBJECT2")) {
        flatten_cdm_object(&mut kvn_lines, obj2);
    }
    if let Some(ud) = v.get("user_defined").or_else(|| v.get("USER_DEFINED")) {
        flatten(&mut kvn_lines, ud);
    }

    let kvn_content = kvn_lines.join("\n");
    crate::ccsds::kvn::parse_cdm(&kvn_content)
}

/// Write a CDM message to JSON format.
///
/// Serializes the CDM to a structured JSON object. Accepts key_case parameter
/// to control whether CCSDS keywords are uppercase or lowercase.
pub fn write_cdm_json(
    cdm: &crate::ccsds::cdm::CDM,
    key_case: CCSDSJsonKeyCase,
) -> Result<String, BraheError> {
    use crate::ccsds::common::covariance9x9_to_lower_triangular;

    let mut root = Map::new();

    // Header
    let mut header = Map::new();
    header.insert(
        key("CCSDS_CDM_VERS", key_case),
        json!(cdm.header.format_version),
    );
    if let Some(ref class) = cdm.header.classification {
        header.insert(key("CLASSIFICATION", key_case), json!(class));
    }
    header.insert(
        key("CREATION_DATE", key_case),
        json!(format_ccsds_datetime(&cdm.header.creation_date)),
    );
    header.insert(key("ORIGINATOR", key_case), json!(&cdm.header.originator));
    if let Some(ref mf) = cdm.header.message_for {
        header.insert(key("MESSAGE_FOR", key_case), json!(mf));
    }
    header.insert(key("MESSAGE_ID", key_case), json!(&cdm.header.message_id));
    root.insert("header".into(), Value::Object(header));

    // Relative metadata
    let rm = &cdm.relative_metadata;
    let mut rel = Map::new();
    if let Some(ref v) = rm.conjunction_id {
        rel.insert(key("CONJUNCTION_ID", key_case), json!(v));
    }
    rel.insert(key("TCA", key_case), json!(format_ccsds_datetime(&rm.tca)));
    rel.insert(key("MISS_DISTANCE", key_case), json!(rm.miss_distance));
    if let Some(v) = rm.mahalanobis_distance {
        rel.insert(key("MAHALANOBIS_DISTANCE", key_case), json!(v));
    }
    if let Some(v) = rm.relative_speed {
        rel.insert(key("RELATIVE_SPEED", key_case), json!(v));
    }
    if let Some(v) = rm.relative_position_r {
        rel.insert(key("RELATIVE_POSITION_R", key_case), json!(v));
    }
    if let Some(v) = rm.relative_position_t {
        rel.insert(key("RELATIVE_POSITION_T", key_case), json!(v));
    }
    if let Some(v) = rm.relative_position_n {
        rel.insert(key("RELATIVE_POSITION_N", key_case), json!(v));
    }
    if let Some(v) = rm.relative_velocity_r {
        rel.insert(key("RELATIVE_VELOCITY_R", key_case), json!(v));
    }
    if let Some(v) = rm.relative_velocity_t {
        rel.insert(key("RELATIVE_VELOCITY_T", key_case), json!(v));
    }
    if let Some(v) = rm.relative_velocity_n {
        rel.insert(key("RELATIVE_VELOCITY_N", key_case), json!(v));
    }
    if let Some(v) = rm.approach_angle {
        rel.insert(key("APPROACH_ANGLE", key_case), json!(v));
    }
    if let Some(ref v) = rm.screen_type {
        rel.insert(key("SCREEN_TYPE", key_case), json!(v));
    }
    if let Some(ref v) = rm.screen_volume_frame {
        rel.insert(
            key("SCREEN_VOLUME_FRAME", key_case),
            json!(format!("{}", v)),
        );
    }
    if let Some(ref v) = rm.screen_volume_shape {
        rel.insert(key("SCREEN_VOLUME_SHAPE", key_case), json!(v));
    }
    if let Some(v) = rm.screen_volume_radius {
        rel.insert(key("SCREEN_VOLUME_RADIUS", key_case), json!(v));
    }
    if let Some(v) = rm.screen_volume_x {
        rel.insert(key("SCREEN_VOLUME_X", key_case), json!(v));
    }
    if let Some(v) = rm.screen_volume_y {
        rel.insert(key("SCREEN_VOLUME_Y", key_case), json!(v));
    }
    if let Some(v) = rm.screen_volume_z {
        rel.insert(key("SCREEN_VOLUME_Z", key_case), json!(v));
    }
    if let Some(ref v) = rm.start_screen_period {
        rel.insert(
            key("START_SCREEN_PERIOD", key_case),
            json!(format_ccsds_datetime(v)),
        );
    }
    if let Some(ref v) = rm.stop_screen_period {
        rel.insert(
            key("STOP_SCREEN_PERIOD", key_case),
            json!(format_ccsds_datetime(v)),
        );
    }
    if let Some(ref v) = rm.screen_entry_time {
        rel.insert(
            key("SCREEN_ENTRY_TIME", key_case),
            json!(format_ccsds_datetime(v)),
        );
    }
    if let Some(ref v) = rm.screen_exit_time {
        rel.insert(
            key("SCREEN_EXIT_TIME", key_case),
            json!(format_ccsds_datetime(v)),
        );
    }
    if let Some(v) = rm.screen_pc_threshold {
        rel.insert(key("SCREEN_PC_THRESHOLD", key_case), json!(v));
    }
    if let Some(v) = rm.collision_probability {
        rel.insert(key("COLLISION_PROBABILITY", key_case), json!(v));
    }
    if let Some(ref s) = rm.collision_probability_method {
        rel.insert(key("COLLISION_PROBABILITY_METHOD", key_case), json!(s));
    }
    if let Some(v) = rm.collision_max_probability {
        rel.insert(key("COLLISION_MAX_PROBABILITY", key_case), json!(v));
    }
    if let Some(ref s) = rm.collision_max_pc_method {
        rel.insert(key("COLLISION_MAX_PC_METHOD", key_case), json!(s));
    }
    if let Some(ref vals) = rm.collision_percentile {
        let parts: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
        rel.insert(
            key("COLLISION_PERCENTILE", key_case),
            json!(parts.join(" ")),
        );
    }
    if let Some(v) = rm.sefi_collision_probability {
        rel.insert(key("SEFI_COLLISION_PROBABILITY", key_case), json!(v));
    }
    if let Some(ref s) = rm.sefi_collision_probability_method {
        rel.insert(key("SEFI_COLLISION_PROBABILITY_METHOD", key_case), json!(s));
    }
    if let Some(ref s) = rm.sefi_fragmentation_model {
        rel.insert(key("SEFI_FRAGMENTATION_MODEL", key_case), json!(s));
    }
    if let Some(ref s) = rm.previous_message_id {
        rel.insert(key("PREVIOUS_MESSAGE_ID", key_case), json!(s));
    }
    if let Some(ref v) = rm.previous_message_epoch {
        rel.insert(
            key("PREVIOUS_MESSAGE_EPOCH", key_case),
            json!(format_ccsds_datetime(v)),
        );
    }
    if let Some(ref v) = rm.next_message_epoch {
        rel.insert(
            key("NEXT_MESSAGE_EPOCH", key_case),
            json!(format_ccsds_datetime(v)),
        );
    }
    root.insert("relative_metadata".into(), Value::Object(rel));

    // Helper to build object JSON
    let build_object = |obj: &crate::ccsds::cdm::CDMObject| -> Value {
        let m = &obj.metadata;
        let d = &obj.data;
        let mut o = Map::new();

        // Metadata
        let mut meta = Map::new();
        meta.insert(key("OBJECT", key_case), json!(&m.object));
        meta.insert(
            key("OBJECT_DESIGNATOR", key_case),
            json!(&m.object_designator),
        );
        meta.insert(key("CATALOG_NAME", key_case), json!(&m.catalog_name));
        meta.insert(key("OBJECT_NAME", key_case), json!(&m.object_name));
        meta.insert(
            key("INTERNATIONAL_DESIGNATOR", key_case),
            json!(&m.international_designator),
        );
        if let Some(ref v) = m.object_type {
            meta.insert(key("OBJECT_TYPE", key_case), json!(v));
        }
        meta.insert(key("EPHEMERIS_NAME", key_case), json!(&m.ephemeris_name));
        meta.insert(
            key("COVARIANCE_METHOD", key_case),
            json!(&m.covariance_method),
        );
        meta.insert(key("MANEUVERABLE", key_case), json!(&m.maneuverable));
        meta.insert(
            key("REF_FRAME", key_case),
            json!(format!("{}", m.ref_frame)),
        );
        o.insert("metadata".into(), Value::Object(meta));

        // State vector (in km/km/s)
        let mut sv = Map::new();
        sv.insert(key("X", key_case), json!(d.state_vector.position[0] / 1e3));
        sv.insert(key("Y", key_case), json!(d.state_vector.position[1] / 1e3));
        sv.insert(key("Z", key_case), json!(d.state_vector.position[2] / 1e3));
        sv.insert(
            key("X_DOT", key_case),
            json!(d.state_vector.velocity[0] / 1e3),
        );
        sv.insert(
            key("Y_DOT", key_case),
            json!(d.state_vector.velocity[1] / 1e3),
        );
        sv.insert(
            key("Z_DOT", key_case),
            json!(d.state_vector.velocity[2] / 1e3),
        );
        o.insert("state_vector".into(), Value::Object(sv));

        // RTN covariance
        let rtn_vals =
            covariance9x9_to_lower_triangular(&d.rtn_covariance.matrix, d.rtn_covariance.dimension);
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
        let cov_arr: Vec<Value> = rtn_vals
            .iter()
            .enumerate()
            .map(|(i, v)| json!([rtn_names[i], v]))
            .collect();
        o.insert("rtn_covariance_ordered".into(), Value::Array(cov_arr));

        Value::Object(o)
    };

    root.insert("object1".into(), build_object(&cdm.object1));
    root.insert("object2".into(), build_object(&cdm.object2));

    serde_json::to_string_pretty(&Value::Object(root))
        .map_err(|e| BraheError::Error(format!("CDM JSON serialization error: {}", e)))
}

// =============================================================================
// Helpers
// =============================================================================

/// Get an f64 value from a JSON object, handling both number and string types.
fn get_json_f64(obj: &Map<String, Value>, key: &str) -> Option<f64> {
    obj.get(key).and_then(|v| match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    })
}

/// Write covariance matrix as named CX_*/CY_*/CZ_* keys in a JSON object.
fn write_json_covariance_elements(
    cv: &mut Map<String, Value>,
    matrix: &nalgebra::SMatrix<f64, 6, 6>,
    key_case: CCSDSJsonKeyCase,
) {
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
        cv.insert(key(name, key_case), json!(values[i]));
    }
}

/// Emit a single JSON value as a KVN line.
fn emit_kvn(lines: &mut Vec<String>, ukey: &str, val: &Value) {
    match val {
        Value::Object(_) => flatten_object(lines, val),
        Value::Null => {}
        Value::String(s) => {
            lines.push(format!("{} = {}", ukey, s));
        }
        Value::Number(n) => {
            lines.push(format!("{} = {}", ukey, n));
        }
        Value::Array(arr) => {
            let parts: Vec<String> = arr.iter().map(|a| a.to_string()).collect();
            lines.push(format!("{} = {}", ukey, parts.join(" ")));
        }
        Value::Bool(b) => {
            lines.push(format!("{} = {}", ukey, if *b { "YES" } else { "NO" }));
        }
    }
}

/// Flatten a JSON object into KVN-style key=value lines.
///
/// Keys are uppercased for KVN compatibility. Nested objects are recursed into.
/// Null values are skipped.
fn flatten_object(lines: &mut Vec<String>, obj: &Value) {
    if let Value::Object(map) = obj {
        for (key, val) in map {
            emit_kvn(lines, &key.to_uppercase(), val);
        }
    }
}

/// Flatten a JSON object, emitting priority keys first.
///
/// Some KVN parsers expect certain keys to appear before others (e.g.,
/// SEMI_MAJOR_AXIS before INCLINATION). This helper emits the listed
/// priority keys first, then remaining keys in default order.
fn flatten_object_ordered(lines: &mut Vec<String>, obj: &Value, priority_keys: &[&str]) {
    if let Value::Object(map) = obj {
        // Emit priority keys first
        for &pk in priority_keys {
            let pk_lower = pk.to_lowercase();
            if let Some(val) = map.get(pk).or_else(|| map.get(&pk_lower)) {
                emit_kvn(lines, pk, val);
            }
        }
        // Emit remaining keys
        for (key, val) in map {
            let ukey = key.to_uppercase();
            if priority_keys.iter().any(|&pk| pk == ukey) {
                continue;
            }
            emit_kvn(lines, &ukey, val);
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::ccsds::common::CCSDSJsonKeyCase;
    use crate::ccsds::oem::OEM;
    use crate::ccsds::omm::OMM;
    use crate::ccsds::opm::OPM;

    // ---- OEM ----

    #[test]
    fn test_oem_json_round_trip() {
        let oem = OEM::from_file("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let json_str = write_oem_json(&oem, CCSDSJsonKeyCase::Lower).unwrap();
        let oem2 = parse_oem_json(&json_str).unwrap();

        assert_eq!(oem.header.originator, oem2.header.originator);
        assert_eq!(oem.segments.len(), oem2.segments.len());
        for (s1, s2) in oem.segments.iter().zip(oem2.segments.iter()) {
            assert_eq!(s1.metadata.object_name, s2.metadata.object_name);
            assert_eq!(s1.metadata.object_id, s2.metadata.object_id);
            assert_eq!(s1.states.len(), s2.states.len());
            for (sv1, sv2) in s1.states.iter().zip(s2.states.iter()) {
                for i in 0..3 {
                    assert!(
                        (sv1.position[i] - sv2.position[i]).abs() < 1.0,
                        "position[{}] mismatch: {} vs {}",
                        i,
                        sv1.position[i],
                        sv2.position[i]
                    );
                    assert!(
                        (sv1.velocity[i] - sv2.velocity[i]).abs() < 0.001,
                        "velocity[{}] mismatch: {} vs {}",
                        i,
                        sv1.velocity[i],
                        sv2.velocity[i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_oem_json_round_trip_with_covariance() {
        let oem = OEM::from_file("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        // Segment 2 has covariances
        assert!(!oem.segments[1].covariances.is_empty());

        let json_str = write_oem_json(&oem, CCSDSJsonKeyCase::Lower).unwrap();
        let oem2 = parse_oem_json(&json_str).unwrap();

        assert_eq!(
            oem.segments[1].covariances.len(),
            oem2.segments[1].covariances.len()
        );
        let cov1 = &oem.segments[1].covariances[0];
        let cov2 = &oem2.segments[1].covariances[0];
        assert!(cov1.epoch.is_some());
        assert!(cov2.epoch.is_some());
        // Compare a diagonal element
        assert!(
            (cov1.matrix[(0, 0)] - cov2.matrix[(0, 0)]).abs() / cov1.matrix[(0, 0)].abs() < 1e-4
        );
    }

    #[test]
    fn test_oem_json_uppercase_keys() {
        let oem = OEM::from_file("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let json_str = write_oem_json(&oem, CCSDSJsonKeyCase::Upper).unwrap();

        // Verify uppercase keys are present
        assert!(json_str.contains("\"OBJECT_NAME\""));
        assert!(json_str.contains("\"CREATION_DATE\""));
        assert!(json_str.contains("\"CCSDS_OEM_VERS\""));
        // Container keys should still be lowercase
        assert!(json_str.contains("\"header\""));
        assert!(json_str.contains("\"segments\""));
        assert!(json_str.contains("\"metadata\""));

        // Should still parse correctly
        let oem2 = parse_oem_json(&json_str).unwrap();
        assert_eq!(oem.header.originator, oem2.header.originator);
    }

    #[test]
    fn test_oem_json_parse_malformed() {
        let result = parse_oem_json("not valid json");
        assert!(result.is_err());
    }

    // ---- OMM ----

    #[test]
    fn test_omm_json_round_trip() {
        let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample1.txt").unwrap();
        let json_str = write_omm_json(&omm, CCSDSJsonKeyCase::Lower).unwrap();
        let omm2 = parse_omm_json(&json_str).unwrap();

        assert_eq!(omm.header.originator, omm2.header.originator);
        assert_eq!(omm.metadata.object_name, omm2.metadata.object_name);
        assert_eq!(omm.metadata.object_id, omm2.metadata.object_id);
        assert!((omm.mean_elements.eccentricity - omm2.mean_elements.eccentricity).abs() < 1e-10);
        assert!((omm.mean_elements.inclination - omm2.mean_elements.inclination).abs() < 1e-6);
        assert!(
            (omm.mean_elements.mean_motion.unwrap() - omm2.mean_elements.mean_motion.unwrap())
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn test_omm_json_uppercase_keys() {
        let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample1.txt").unwrap();
        let json_str = write_omm_json(&omm, CCSDSJsonKeyCase::Upper).unwrap();

        assert!(json_str.contains("\"OBJECT_NAME\""));
        assert!(json_str.contains("\"MEAN_MOTION\""));
        assert!(json_str.contains("\"header\""));
        assert!(json_str.contains("\"metadata\""));

        let omm2 = parse_omm_json(&json_str).unwrap();
        assert_eq!(omm.metadata.object_name, omm2.metadata.object_name);
    }

    #[test]
    fn test_omm_json_parse_malformed() {
        let result = parse_omm_json("not valid json");
        assert!(result.is_err());
    }

    // ---- OPM ----

    #[test]
    fn test_opm_json_round_trip() {
        let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample1.txt").unwrap();
        let json_str = write_opm_json(&opm, CCSDSJsonKeyCase::Lower).unwrap();
        let opm2 = parse_opm_json(&json_str).unwrap();

        assert_eq!(opm.header.originator, opm2.header.originator);
        assert_eq!(opm.metadata.object_name, opm2.metadata.object_name);
        for i in 0..3 {
            assert!(
                (opm.state_vector.position[i] - opm2.state_vector.position[i]).abs() < 1.0,
                "position[{}] mismatch",
                i
            );
            assert!(
                (opm.state_vector.velocity[i] - opm2.state_vector.velocity[i]).abs() < 0.001,
                "velocity[{}] mismatch",
                i
            );
        }
    }

    #[test]
    fn test_opm_json_round_trip_with_maneuvers() {
        let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample5.txt").unwrap();
        assert_eq!(opm.maneuvers.len(), 3);

        let json_str = write_opm_json(&opm, CCSDSJsonKeyCase::Lower).unwrap();
        let opm2 = parse_opm_json(&json_str).unwrap();

        assert_eq!(opm.maneuvers.len(), opm2.maneuvers.len());
        for (m1, m2) in opm.maneuvers.iter().zip(opm2.maneuvers.iter()) {
            assert!((m1.duration - m2.duration).abs() < 0.01);
            for i in 0..3 {
                assert!((m1.dv[i] - m2.dv[i]).abs() < 0.001);
            }
        }
    }

    #[test]
    fn test_opm_json_with_keplerian() {
        let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample5.txt").unwrap();
        assert!(opm.keplerian_elements.is_some());

        let json_str = write_opm_json(&opm, CCSDSJsonKeyCase::Lower).unwrap();
        let opm2 = parse_opm_json(&json_str).unwrap();

        let kep1 = opm.keplerian_elements.as_ref().unwrap();
        let kep2 = opm2.keplerian_elements.as_ref().unwrap();
        assert!((kep1.semi_major_axis - kep2.semi_major_axis).abs() < 1.0);
        assert!((kep1.eccentricity - kep2.eccentricity).abs() < 1e-6);
    }

    #[test]
    fn test_opm_json_uppercase_keys() {
        let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample1.txt").unwrap();
        let json_str = write_opm_json(&opm, CCSDSJsonKeyCase::Upper).unwrap();

        assert!(json_str.contains("\"OBJECT_NAME\""));
        assert!(json_str.contains("\"X\""));
        assert!(json_str.contains("\"header\""));
        assert!(json_str.contains("\"state_vector\""));

        let opm2 = parse_opm_json(&json_str).unwrap();
        assert_eq!(opm.metadata.object_name, opm2.metadata.object_name);
    }

    #[test]
    fn test_opm_json_parse_malformed() {
        let result = parse_opm_json("not valid json");
        assert!(result.is_err());
    }

    // ---- CDM ----

    #[test]
    fn test_cdm_json_round_trip_lowercase() {
        let cdm =
            crate::ccsds::cdm::CDM::from_file("test_assets/ccsds/cdm/CDMExample1.txt").unwrap();
        let json_str = write_cdm_json(&cdm, CCSDSJsonKeyCase::Lower).unwrap();
        let cdm2 = parse_cdm_json(&json_str).unwrap();

        assert_eq!(cdm.header.originator, cdm2.header.originator);
        assert_eq!(cdm.header.message_id, cdm2.header.message_id);
    }

    #[test]
    fn test_cdm_json_key_case() {
        let cdm =
            crate::ccsds::cdm::CDM::from_file("test_assets/ccsds/cdm/CDMExample1.txt").unwrap();
        let lower = write_cdm_json(&cdm, CCSDSJsonKeyCase::Lower).unwrap();
        let upper = write_cdm_json(&cdm, CCSDSJsonKeyCase::Upper).unwrap();

        assert!(lower.contains("\"originator\""));
        assert!(upper.contains("\"ORIGINATOR\""));
        // Both should parse correctly
        assert!(parse_cdm_json(&lower).is_ok());
        assert!(parse_cdm_json(&upper).is_ok());
    }

    // ---- key helper ----

    #[test]
    fn test_key_case_conversion() {
        assert_eq!(key("OBJECT_NAME", CCSDSJsonKeyCase::Lower), "object_name");
        assert_eq!(key("OBJECT_NAME", CCSDSJsonKeyCase::Upper), "OBJECT_NAME");
        assert_eq!(key("x", CCSDSJsonKeyCase::Upper), "X");
        assert_eq!(key("X", CCSDSJsonKeyCase::Lower), "x");
    }

    // =========================================================================
    // Parse edge cases
    // =========================================================================

    #[test]
    fn test_parse_oem_json_missing_header() {
        // JSON with segments but no header — KVN parser requires CCSDS_OEM_VERS,
        // so this should return an error.
        let json = r#"{
            "segments": [{
                "metadata": {
                    "OBJECT_NAME": "SAT",
                    "OBJECT_ID": "2024-001A",
                    "CENTER_NAME": "EARTH",
                    "REF_FRAME": "EME2000",
                    "TIME_SYSTEM": "UTC",
                    "START_TIME": "2024-01-01T00:00:00.000",
                    "STOP_TIME": "2024-01-01T01:00:00.000"
                },
                "states": [{
                    "EPOCH": "2024-01-01T00:00:00.000",
                    "X": 7000.0, "Y": 0.0, "Z": 0.0,
                    "X_DOT": 0.0, "Y_DOT": 7.5, "Z_DOT": 0.0
                }]
            }]
        }"#;
        let result = parse_oem_json(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_oem_json_missing_segments() {
        // JSON with header but no segments array — should produce an OEM
        // with zero segments (KVN parser treats this as empty data).
        let json = r#"{
            "header": {
                "CCSDS_OEM_VERS": "3.0",
                "CREATION_DATE": "2024-01-01T00:00:00.000",
                "ORIGINATOR": "TEST"
            }
        }"#;
        let result = parse_oem_json(json);
        // Should either succeed with 0 segments or fail cleanly
        if let Ok(oem) = result {
            assert_eq!(oem.segments.len(), 0);
        }
        // Err is also acceptable — KVN parser may require at least one segment
    }

    #[test]
    fn test_parse_oem_json_incomplete_state_vector() {
        // State missing Z_DOT — should be silently skipped (not included in KVN)
        let json = r#"{
            "header": {
                "CCSDS_OEM_VERS": "3.0",
                "CREATION_DATE": "2024-01-01T00:00:00.000",
                "ORIGINATOR": "TEST"
            },
            "segments": [{
                "metadata": {
                    "OBJECT_NAME": "SAT",
                    "OBJECT_ID": "2024-001A",
                    "CENTER_NAME": "EARTH",
                    "REF_FRAME": "EME2000",
                    "TIME_SYSTEM": "UTC",
                    "START_TIME": "2024-01-01T00:00:00.000",
                    "STOP_TIME": "2024-01-01T01:00:00.000"
                },
                "states": [
                    {
                        "EPOCH": "2024-01-01T00:00:00.000",
                        "X": 7000.0, "Y": 0.0, "Z": 0.0,
                        "X_DOT": 0.0, "Y_DOT": 7.5
                    },
                    {
                        "EPOCH": "2024-01-01T00:30:00.000",
                        "X": 6000.0, "Y": 1000.0, "Z": 0.0,
                        "X_DOT": -1.0, "Y_DOT": 6.0, "Z_DOT": 0.0
                    }
                ]
            }]
        }"#;
        let oem = parse_oem_json(json).unwrap();
        // The incomplete state should be skipped, only the complete one emitted
        assert_eq!(oem.segments[0].states.len(), 1);
    }

    #[test]
    fn test_parse_omm_json_uppercase_section_keys() {
        // OMM JSON with uppercase section names (HEADER, METADATA, etc.)
        let json = r#"{
            "HEADER": {
                "CCSDS_OMM_VERS": "3.0",
                "CREATION_DATE": "2024-01-01T00:00:00.000",
                "ORIGINATOR": "UPPER_TEST"
            },
            "METADATA": {
                "OBJECT_NAME": "TEST_SAT",
                "OBJECT_ID": "2024-001A",
                "CENTER_NAME": "EARTH",
                "REF_FRAME": "TEME",
                "TIME_SYSTEM": "UTC",
                "MEAN_ELEMENT_THEORY": "SGP4"
            },
            "MEAN_ELEMENTS": {
                "EPOCH": "2024-01-01T00:00:00.000",
                "MEAN_MOTION": 15.123456,
                "ECCENTRICITY": 0.001,
                "INCLINATION": 51.6,
                "RA_OF_ASC_NODE": 100.0,
                "ARG_OF_PERICENTER": 200.0,
                "MEAN_ANOMALY": 300.0
            }
        }"#;
        let omm = parse_omm_json(json).unwrap();
        assert_eq!(omm.metadata.object_name, "TEST_SAT");
        assert_eq!(omm.header.originator, "UPPER_TEST");
        assert!((omm.mean_elements.mean_motion.unwrap() - 15.123456).abs() < 1e-6);
    }

    #[test]
    fn test_parse_opm_json_uppercase_section_keys() {
        // OPM JSON with uppercase section names
        let json = r#"{
            "HEADER": {
                "CCSDS_OPM_VERS": "3.0",
                "CREATION_DATE": "2024-01-01T00:00:00.000",
                "ORIGINATOR": "UPPER_OPM"
            },
            "METADATA": {
                "OBJECT_NAME": "TEST_OPM",
                "OBJECT_ID": "2024-001A",
                "CENTER_NAME": "EARTH",
                "REF_FRAME": "EME2000",
                "TIME_SYSTEM": "UTC"
            },
            "STATE_VECTOR": {
                "EPOCH": "2024-01-01T00:00:00.000",
                "X": 7000.0, "Y": 0.0, "Z": 0.0,
                "X_DOT": 0.0, "Y_DOT": 7.5, "Z_DOT": 0.0
            }
        }"#;
        let opm = parse_opm_json(json).unwrap();
        assert_eq!(opm.metadata.object_name, "TEST_OPM");
        assert_eq!(opm.header.originator, "UPPER_OPM");
    }

    #[test]
    fn test_parse_cdm_json_uppercase_container_keys() {
        // CDM with OBJECT1/OBJECT2 uppercase container keys
        let cdm =
            crate::ccsds::cdm::CDM::from_file("test_assets/ccsds/cdm/CDMExample1.txt").unwrap();
        let json_str = write_cdm_json(&cdm, CCSDSJsonKeyCase::Upper).unwrap();

        // Manually replace lowercase container keys with uppercase to test parse
        let json_upper = json_str
            .replace("\"header\"", "\"HEADER\"")
            .replace("\"relative_metadata\"", "\"RELATIVE_METADATA\"")
            .replace("\"object1\"", "\"OBJECT1\"")
            .replace("\"object2\"", "\"OBJECT2\"")
            .replace("\"metadata\"", "\"METADATA\"")
            .replace("\"state_vector\"", "\"STATE_VECTOR\"")
            .replace("\"rtn_covariance_ordered\"", "\"RTN_COVARIANCE_ORDERED\"");

        let cdm2 = parse_cdm_json(&json_upper).unwrap();
        assert_eq!(cdm.header.originator, cdm2.header.originator);
        assert_eq!(cdm.header.message_id, cdm2.header.message_id);
    }

    #[test]
    fn test_get_json_f64_string_values() {
        // get_json_f64 should handle string-encoded numbers (e.g. SpaceTrack)
        let mut obj = Map::new();
        obj.insert("X".to_string(), json!("123.45"));
        obj.insert("Y".to_string(), json!(678.9));
        obj.insert("Z".to_string(), json!("not_a_number"));
        obj.insert("W".to_string(), json!(null));

        assert!((get_json_f64(&obj, "X").unwrap() - 123.45).abs() < 1e-10);
        assert!((get_json_f64(&obj, "Y").unwrap() - 678.9).abs() < 1e-10);
        assert!(get_json_f64(&obj, "Z").is_none());
        assert!(get_json_f64(&obj, "W").is_none());
        assert!(get_json_f64(&obj, "MISSING").is_none());
    }

    // =========================================================================
    // Write OEM — optional field branches
    // =========================================================================

    #[test]
    fn test_write_oem_json_optional_header_fields() {
        let mut oem = OEM::from_file("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        oem.header.classification = Some("PUBLIC".to_string());
        oem.header.message_id = Some("OEM-MSG-001".to_string());

        let json_str = write_oem_json(&oem, CCSDSJsonKeyCase::Lower).unwrap();
        assert!(json_str.contains("\"classification\""));
        assert!(json_str.contains("PUBLIC"));
        assert!(json_str.contains("\"message_id\""));
        assert!(json_str.contains("OEM-MSG-001"));

        // Round-trip should preserve the values in the re-generated JSON
        let oem2 = parse_oem_json(&json_str).unwrap();
        let json_str2 = write_oem_json(&oem2, CCSDSJsonKeyCase::Lower).unwrap();
        assert!(json_str2.contains("PUBLIC"));
        assert!(json_str2.contains("OEM-MSG-001"));
    }

    #[test]
    fn test_write_oem_json_optional_metadata_fields() {
        // OEMExample4.txt has useable_start/stop, interpolation, interpolation_degree
        let oem = OEM::from_file("test_assets/ccsds/oem/OEMExample4.txt").unwrap();
        let seg = &oem.segments[0];
        assert!(seg.metadata.useable_start_time.is_some());
        assert!(seg.metadata.useable_stop_time.is_some());
        assert!(seg.metadata.interpolation.is_some());
        assert!(seg.metadata.interpolation_degree.is_some());

        let json_str = write_oem_json(&oem, CCSDSJsonKeyCase::Lower).unwrap();
        assert!(json_str.contains("\"useable_start_time\""));
        assert!(json_str.contains("\"useable_stop_time\""));
        assert!(json_str.contains("\"interpolation\""));
        assert!(json_str.contains("\"interpolation_degree\""));

        // Round-trip
        let oem2 = parse_oem_json(&json_str).unwrap();
        assert!(oem2.segments[0].metadata.useable_start_time.is_some());
        assert!(oem2.segments[0].metadata.useable_stop_time.is_some());
        assert_eq!(
            oem2.segments[0].metadata.interpolation.as_deref(),
            Some("HERMITE")
        );
        assert_eq!(oem2.segments[0].metadata.interpolation_degree, Some(1));
    }

    #[test]
    fn test_write_oem_json_ref_frame_epoch() {
        let mut oem = OEM::from_file("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let ref_epoch = crate::time::Epoch::from_datetime(
            2000,
            1,
            1,
            12,
            0,
            0.0,
            0.0,
            crate::time::TimeSystem::UTC,
        );
        oem.segments[0].metadata.ref_frame_epoch = Some(ref_epoch);

        let json_str = write_oem_json(&oem, CCSDSJsonKeyCase::Lower).unwrap();
        assert!(json_str.contains("\"ref_frame_epoch\""));
        assert!(json_str.contains("2000-01-01T12:00:00"));
    }

    #[test]
    fn test_write_oem_json_state_with_acceleration() {
        let mut oem = OEM::from_file("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        // Add acceleration to the first state in the first segment
        oem.segments[0].states[0].acceleration = Some([0.001, 0.002, 0.003]);

        let json_str = write_oem_json(&oem, CCSDSJsonKeyCase::Lower).unwrap();
        assert!(json_str.contains("\"x_ddot\""));
        assert!(json_str.contains("\"y_ddot\""));
        assert!(json_str.contains("\"z_ddot\""));

        // Round-trip: verify acceleration survives
        let oem2 = parse_oem_json(&json_str).unwrap();
        let acc = oem2.segments[0].states[0].acceleration.unwrap();
        assert!((acc[0] - 0.001).abs() < 1e-6);
        assert!((acc[1] - 0.002).abs() < 1e-6);
        assert!((acc[2] - 0.003).abs() < 1e-6);
    }

    // =========================================================================
    // Write OMM — optional field branches
    // =========================================================================

    #[test]
    fn test_write_omm_json_with_semi_major_axis() {
        let metadata = crate::ccsds::omm::OMMMetadata::new(
            "SMA_SAT".to_string(),
            "2024-001A".to_string(),
            "EARTH".to_string(),
            crate::ccsds::common::CCSDSRefFrame::GCRF,
            crate::ccsds::common::CCSDSTimeSystem::UTC,
            "DSST".to_string(),
        );
        let mut elements = crate::ccsds::omm::OMMeanElements::new(
            crate::time::Epoch::from_datetime(
                2024,
                1,
                1,
                0,
                0,
                0.0,
                0.0,
                crate::time::TimeSystem::UTC,
            ),
            0.001, // eccentricity
            51.6,  // inclination
            100.0, // raan
            200.0, // argp
            300.0, // mean anomaly
        );
        elements.semi_major_axis = Some(6878.0); // km

        let omm = crate::ccsds::omm::OMM::new("SMA_TEST".to_string(), metadata, elements);

        let json_str = write_omm_json(&omm, CCSDSJsonKeyCase::Lower).unwrap();
        assert!(json_str.contains("\"semi_major_axis\""));
        assert!(json_str.contains("6878"));
    }

    #[test]
    fn test_write_omm_json_with_gm() {
        let metadata = crate::ccsds::omm::OMMMetadata::new(
            "GM_SAT".to_string(),
            "2024-001A".to_string(),
            "EARTH".to_string(),
            crate::ccsds::common::CCSDSRefFrame::GCRF,
            crate::ccsds::common::CCSDSTimeSystem::UTC,
            "DSST".to_string(),
        );
        // GM internally stored as m^3/s^2
        let elements = crate::ccsds::omm::OMMeanElements::new(
            crate::time::Epoch::from_datetime(
                2024,
                1,
                1,
                0,
                0,
                0.0,
                0.0,
                crate::time::TimeSystem::UTC,
            ),
            0.001,
            51.6,
            100.0,
            200.0,
            300.0,
        )
        .with_mean_motion(15.0)
        .with_gm(398600.4415e9); // m^3/s^2

        let omm = crate::ccsds::omm::OMM::new("GM_TEST".to_string(), metadata, elements);

        let json_str = write_omm_json(&omm, CCSDSJsonKeyCase::Lower).unwrap();
        assert!(json_str.contains("\"gm\""));
        // Should be written as km^3/s^2 (divided by 1e9)
        assert!(json_str.contains("398600.4415"));

        // Round-trip
        let omm2 = parse_omm_json(&json_str).unwrap();
        assert!((omm2.mean_elements.gm.unwrap() - 398600.4415e9).abs() / 398600.4415e9 < 1e-6);
    }

    #[test]
    fn test_write_omm_json_with_user_defined() {
        let metadata = crate::ccsds::omm::OMMMetadata::new(
            "UD_SAT".to_string(),
            "2024-001A".to_string(),
            "EARTH".to_string(),
            crate::ccsds::common::CCSDSRefFrame::TEME,
            crate::ccsds::common::CCSDSTimeSystem::UTC,
            "SGP4".to_string(),
        );
        let elements = crate::ccsds::omm::OMMeanElements::new(
            crate::time::Epoch::from_datetime(
                2024,
                1,
                1,
                0,
                0,
                0.0,
                0.0,
                crate::time::TimeSystem::UTC,
            ),
            0.001,
            51.6,
            100.0,
            200.0,
            300.0,
        )
        .with_mean_motion(15.0);

        let mut omm = crate::ccsds::omm::OMM::new("UD_TEST".to_string(), metadata, elements);
        let mut params = std::collections::HashMap::new();
        params.insert("CUSTOM_PARAM".to_string(), "hello".to_string());
        params.insert("ANOTHER".to_string(), "42".to_string());
        omm.user_defined = Some(crate::ccsds::common::CCSDSUserDefined { parameters: params });

        let json_str = write_omm_json(&omm, CCSDSJsonKeyCase::Lower).unwrap();
        assert!(json_str.contains("\"user_defined\""));
        assert!(json_str.contains("USER_DEFINED_CUSTOM_PARAM"));
        assert!(json_str.contains("hello"));
        assert!(json_str.contains("USER_DEFINED_ANOTHER"));
    }

    // =========================================================================
    // Write OPM — optional field branches
    // =========================================================================

    #[test]
    fn test_write_opm_json_true_anomaly_only() {
        let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample5.txt").unwrap();
        let kep = opm.keplerian_elements.as_ref().unwrap();
        // OPMExample5 has true_anomaly, no mean_anomaly
        assert!(kep.true_anomaly.is_some());

        let json_str = write_opm_json(&opm, CCSDSJsonKeyCase::Lower).unwrap();
        assert!(json_str.contains("\"true_anomaly\""));

        // Round-trip
        let opm2 = parse_opm_json(&json_str).unwrap();
        let kep2 = opm2.keplerian_elements.as_ref().unwrap();
        assert!(kep2.true_anomaly.is_some());
        assert!((kep.true_anomaly.unwrap() - kep2.true_anomaly.unwrap()).abs() < 1e-4);
    }

    #[test]
    fn test_write_opm_json_mean_anomaly_only() {
        // Build an OPM with keplerian elements that have mean_anomaly but not true_anomaly
        let metadata = crate::ccsds::opm::OPMMetadata::new(
            "MA_SAT".to_string(),
            "2024-001A".to_string(),
            "EARTH".to_string(),
            crate::ccsds::common::CCSDSRefFrame::GCRF,
            crate::ccsds::common::CCSDSTimeSystem::UTC,
        );
        let sv = crate::ccsds::opm::OPMStateVector::new(
            crate::time::Epoch::from_datetime(
                2024,
                1,
                1,
                0,
                0,
                0.0,
                0.0,
                crate::time::TimeSystem::UTC,
            ),
            [7000e3, 0.0, 0.0],
            [0.0, 7500.0, 0.0],
        );
        let mut opm = OPM::new("MA_TEST".to_string(), metadata, sv);
        opm.keplerian_elements = Some(crate::ccsds::opm::OPMKeplerianElements {
            semi_major_axis: 7000e3,
            eccentricity: 0.001,
            inclination: 51.6,
            ra_of_asc_node: 100.0,
            arg_of_pericenter: 200.0,
            true_anomaly: None,
            mean_anomaly: Some(45.0),
            gm: Some(398600.4415e9),
            comments: Vec::new(),
        });

        let json_str = write_opm_json(&opm, CCSDSJsonKeyCase::Lower).unwrap();
        assert!(json_str.contains("\"mean_anomaly\""));
        assert!(json_str.contains("\"gm\""));
        // GM should be written in km^3/s^2
        assert!(json_str.contains("398600.4415"));
        assert!(!json_str.contains("\"true_anomaly\""));

        // Round-trip
        let opm2 = parse_opm_json(&json_str).unwrap();
        let kep2 = opm2.keplerian_elements.as_ref().unwrap();
        assert!(kep2.mean_anomaly.is_some());
        assert!((kep2.mean_anomaly.unwrap() - 45.0).abs() < 1e-4);
        assert!(kep2.true_anomaly.is_none());
    }

    #[test]
    fn test_write_opm_json_maneuver_with_delta_mass() {
        let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample5.txt").unwrap();
        // OPMExample5 has maneuvers with delta_mass
        assert!(opm.maneuvers[0].delta_mass.is_some());

        let json_str = write_opm_json(&opm, CCSDSJsonKeyCase::Lower).unwrap();
        assert!(json_str.contains("\"man_delta_mass\""));
        assert!(json_str.contains("-18.418"));

        // Round-trip
        let opm2 = parse_opm_json(&json_str).unwrap();
        assert!(opm2.maneuvers[0].delta_mass.is_some());
        assert!((opm2.maneuvers[0].delta_mass.unwrap() - (-18.418)).abs() < 0.01);
    }

    #[test]
    fn test_write_opm_json_with_user_defined() {
        let metadata = crate::ccsds::opm::OPMMetadata::new(
            "UD_OPM".to_string(),
            "2024-001A".to_string(),
            "EARTH".to_string(),
            crate::ccsds::common::CCSDSRefFrame::GCRF,
            crate::ccsds::common::CCSDSTimeSystem::UTC,
        );
        let sv = crate::ccsds::opm::OPMStateVector::new(
            crate::time::Epoch::from_datetime(
                2024,
                1,
                1,
                0,
                0,
                0.0,
                0.0,
                crate::time::TimeSystem::UTC,
            ),
            [7000e3, 0.0, 0.0],
            [0.0, 7500.0, 0.0],
        );
        let mut opm = OPM::new("UD_OPM_TEST".to_string(), metadata, sv);
        let mut params = std::collections::HashMap::new();
        params.insert("MISSION_ID".to_string(), "ALPHA-1".to_string());
        opm.user_defined = Some(crate::ccsds::common::CCSDSUserDefined { parameters: params });

        let json_str = write_opm_json(&opm, CCSDSJsonKeyCase::Lower).unwrap();
        assert!(json_str.contains("\"user_defined\""));
        assert!(json_str.contains("USER_DEFINED_MISSION_ID"));
        assert!(json_str.contains("ALPHA-1"));
    }

    #[test]
    fn test_write_opm_json_with_spacecraft_parameters() {
        // OPMExample5 has spacecraft parameters
        let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample5.txt").unwrap();
        assert!(opm.spacecraft_parameters.is_some());

        let json_str = write_opm_json(&opm, CCSDSJsonKeyCase::Lower).unwrap();
        assert!(json_str.contains("\"spacecraft_parameters\""));
        assert!(json_str.contains("\"mass\""));
        assert!(json_str.contains("1913"));
        assert!(json_str.contains("\"solar_rad_area\""));
        assert!(json_str.contains("\"solar_rad_coeff\""));
        assert!(json_str.contains("\"drag_area\""));
        assert!(json_str.contains("\"drag_coeff\""));
    }

    // =========================================================================
    // Helper functions
    // =========================================================================

    #[test]
    fn test_emit_kvn_bool_values() {
        let mut lines = Vec::new();
        emit_kvn(&mut lines, "FLAG_TRUE", &json!(true));
        emit_kvn(&mut lines, "FLAG_FALSE", &json!(false));

        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "FLAG_TRUE = YES");
        assert_eq!(lines[1], "FLAG_FALSE = NO");
    }

    #[test]
    fn test_emit_kvn_null_skipped() {
        let mut lines = Vec::new();
        emit_kvn(&mut lines, "NULL_KEY", &json!(null));
        assert!(lines.is_empty());
    }

    #[test]
    fn test_emit_kvn_array_values() {
        let mut lines = Vec::new();
        emit_kvn(&mut lines, "ARR", &json!([1, 2, 3]));
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "ARR = 1 2 3");
    }

    #[test]
    fn test_flatten_object_ordered_deduplication() {
        // Priority keys should appear first and not be duplicated in the
        // remaining keys pass.
        let obj = json!({
            "SEMI_MAJOR_AXIS": 7000.0,
            "ECCENTRICITY": 0.001,
            "INCLINATION": 51.6,
            "EXTRA_KEY": "extra"
        });
        let priority = ["SEMI_MAJOR_AXIS", "ECCENTRICITY"];
        let mut lines = Vec::new();
        flatten_object_ordered(&mut lines, &obj, &priority);

        // SEMI_MAJOR_AXIS and ECCENTRICITY should appear first
        assert!(lines[0].starts_with("SEMI_MAJOR_AXIS = "));
        assert!(lines[0].contains("7000"));
        assert!(lines[1].starts_with("ECCENTRICITY = "));
        assert!(lines[1].contains("0.001"));
        // Remaining keys should appear after (order among remaining depends on serde_json)
        assert_eq!(lines.len(), 4);
        // Verify no duplicates
        let sma_count = lines
            .iter()
            .filter(|l| l.starts_with("SEMI_MAJOR_AXIS"))
            .count();
        let ecc_count = lines
            .iter()
            .filter(|l| l.starts_with("ECCENTRICITY"))
            .count();
        assert_eq!(sma_count, 1);
        assert_eq!(ecc_count, 1);
    }

    #[test]
    fn test_flatten_object_ordered_lowercase_keys() {
        // The ordered flattener should find keys case-insensitively
        let obj = json!({
            "semi_major_axis": 7000.0,
            "eccentricity": 0.001
        });
        let priority = ["SEMI_MAJOR_AXIS", "ECCENTRICITY"];
        let mut lines = Vec::new();
        flatten_object_ordered(&mut lines, &obj, &priority);

        assert!(lines.len() >= 2);
        assert!(lines[0].starts_with("SEMI_MAJOR_AXIS"));
        assert!(lines[1].starts_with("ECCENTRICITY"));
    }

    // =========================================================================
    // CDM — write with Bool in flatten
    // =========================================================================

    #[test]
    fn test_cdm_json_bool_in_flatten() {
        // The CDM flatten function handles Bool values (YES/NO).
        // Verify via a CDM round-trip that boolean-like fields survive.
        let cdm =
            crate::ccsds::cdm::CDM::from_file("test_assets/ccsds/cdm/CDMExample1.txt").unwrap();
        let json_str = write_cdm_json(&cdm, CCSDSJsonKeyCase::Lower).unwrap();

        // The JSON should be parseable
        let cdm2 = parse_cdm_json(&json_str).unwrap();
        assert_eq!(
            cdm.object1.metadata.object_name,
            cdm2.object1.metadata.object_name
        );
        assert_eq!(
            cdm.object2.metadata.object_name,
            cdm2.object2.metadata.object_name
        );
    }

    #[test]
    fn test_cdm_json_parse_malformed() {
        let result = parse_cdm_json("not valid json");
        assert!(result.is_err());
    }

    // =========================================================================
    // OEM with acceleration round-trip via test asset
    // =========================================================================

    #[test]
    fn test_oem_json_round_trip_example4_full() {
        // OEMExample4.txt has USEABLE_START_TIME, USEABLE_STOP_TIME,
        // INTERPOLATION, INTERPOLATION_DEGREE
        let oem = OEM::from_file("test_assets/ccsds/oem/OEMExample4.txt").unwrap();
        let json_str = write_oem_json(&oem, CCSDSJsonKeyCase::Upper).unwrap();

        // Verify uppercase data keys, lowercase container keys
        assert!(json_str.contains("\"USEABLE_START_TIME\""));
        assert!(json_str.contains("\"USEABLE_STOP_TIME\""));
        assert!(json_str.contains("\"INTERPOLATION\""));
        assert!(json_str.contains("\"INTERPOLATION_DEGREE\""));
        assert!(json_str.contains("\"header\""));
        assert!(json_str.contains("\"segments\""));

        // Round-trip
        let oem2 = parse_oem_json(&json_str).unwrap();
        assert_eq!(oem.segments.len(), oem2.segments.len());
        assert!(oem2.segments[0].metadata.interpolation.is_some());
    }

    // =========================================================================
    // OPM with keplerian GM conversion
    // =========================================================================

    #[test]
    fn test_opm_json_keplerian_gm_conversion() {
        // OPMExample5 has GM = 398600.4415 km^3/s^2
        let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample5.txt").unwrap();
        let kep = opm.keplerian_elements.as_ref().unwrap();
        // Internally stored as m^3/s^2
        assert!((kep.gm.unwrap() - 398600.4415e9).abs() / 398600.4415e9 < 1e-6);

        let json_str = write_opm_json(&opm, CCSDSJsonKeyCase::Lower).unwrap();
        // JSON should have km^3/s^2
        assert!(json_str.contains("398600.4415"));

        // Round-trip preserves the value
        let opm2 = parse_opm_json(&json_str).unwrap();
        let kep2 = opm2.keplerian_elements.as_ref().unwrap();
        assert!((kep2.gm.unwrap() - 398600.4415e9).abs() / 398600.4415e9 < 1e-6);
    }

    // =========================================================================
    // OEM JSON with lowercase state keys (x, y, z, x_dot, ...)
    // =========================================================================

    #[test]
    fn test_parse_oem_json_lowercase_state_keys() {
        // Verify that lowercase state vector keys (x, y, z, x_dot...) are handled
        let json = r#"{
            "header": {
                "ccsds_oem_vers": "3.0",
                "creation_date": "2024-01-01T00:00:00.000",
                "originator": "LOWER_TEST"
            },
            "segments": [{
                "metadata": {
                    "object_name": "SAT",
                    "object_id": "2024-001A",
                    "center_name": "EARTH",
                    "ref_frame": "EME2000",
                    "time_system": "UTC",
                    "start_time": "2024-01-01T00:00:00.000",
                    "stop_time": "2024-01-01T01:00:00.000"
                },
                "states": [{
                    "epoch": "2024-01-01T00:00:00.000",
                    "x": 7000.0, "y": 100.0, "z": 200.0,
                    "x_dot": 0.5, "y_dot": 7.5, "z_dot": 0.1
                }]
            }]
        }"#;
        let oem = parse_oem_json(json).unwrap();
        assert_eq!(oem.segments[0].states.len(), 1);
        // Values are in km in JSON, converted to m internally
        assert!((oem.segments[0].states[0].position[0] - 7000e3).abs() < 1.0);
    }

    #[test]
    fn test_parse_oem_json_with_acceleration_lowercase() {
        // Verify x_ddot/y_ddot/z_ddot lowercase keys
        let json = r#"{
            "header": {
                "ccsds_oem_vers": "3.0",
                "creation_date": "2024-01-01T00:00:00.000",
                "originator": "ACC_TEST"
            },
            "segments": [{
                "metadata": {
                    "object_name": "SAT",
                    "object_id": "2024-001A",
                    "center_name": "EARTH",
                    "ref_frame": "EME2000",
                    "time_system": "UTC",
                    "start_time": "2024-01-01T00:00:00.000",
                    "stop_time": "2024-01-01T01:00:00.000"
                },
                "states": [{
                    "epoch": "2024-01-01T00:00:00.000",
                    "x": 7000.0, "y": 0.0, "z": 0.0,
                    "x_dot": 0.0, "y_dot": 7.5, "z_dot": 0.0,
                    "x_ddot": 0.001, "y_ddot": 0.002, "z_ddot": 0.003
                }]
            }]
        }"#;
        let oem = parse_oem_json(json).unwrap();
        let acc = oem.segments[0].states[0].acceleration.unwrap();
        // Acceleration values are in km/s^2 in JSON, converted to m/s^2
        assert!((acc[0] - 0.001e3).abs() < 1e-6);
        assert!((acc[1] - 0.002e3).abs() < 1e-6);
        assert!((acc[2] - 0.003e3).abs() < 1e-6);
    }
}
