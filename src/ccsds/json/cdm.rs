/*!
 * JSON reader and writer for the Conjunction Data Message (CDM).
 *
 * Reference: CCSDS 508.0-B-1 (Conjunction Data Message), section 3
 */

use serde_json::{Map, Value, json};

use crate::ccsds::common::{CCSDSJsonKeyCase, CCSDSTimeSystem, format_ccsds_datetime_in};
use crate::ccsds::error::ccsds_parse_error;
use crate::ccsds::json::common::{COMMENTS_KEY, emit_json_comments, key};
use crate::utils::errors::BraheError;

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
                if ukey == COMMENTS_KEY {
                    continue;
                }

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
            emit_json_comments(lines, meta);
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
            emit_json_comments(lines, sv);
            flatten(lines, sv);
        }
        if let Some(cov) = obj
            .get("rtn_covariance")
            .or_else(|| obj.get("RTN_COVARIANCE"))
        {
            emit_json_comments(lines, cov);
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
            emit_json_comments(lines, od);
            flatten(lines, od);
        }
        if let Some(ap) = obj
            .get("additional_parameters")
            .or_else(|| obj.get("ADDITIONAL_PARAMETERS"))
        {
            emit_json_comments(lines, ap);
            flatten(lines, ap);
        }
    }

    if let Some(header) = v.get("header").or_else(|| v.get("HEADER")) {
        emit_json_comments(&mut kvn_lines, header);
        flatten(&mut kvn_lines, header);
    }
    if let Some(rel) = v
        .get("relative_metadata")
        .or_else(|| v.get("RELATIVE_METADATA"))
    {
        emit_json_comments(&mut kvn_lines, rel);
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
    let mut cdm = crate::ccsds::kvn::parse_cdm(&kvn_content)?;

    // The object Data block and its sub-blocks each carry a COMMENT row, but
    // KVN places nothing between them, so the parser attributes a run spanning
    // both to the innermost block. JSON keeps them apart, so the data-level
    // comments are read from the document rather than from the flattened text.
    for (key, object) in [("object1", &mut cdm.object1), ("object2", &mut cdm.object2)] {
        if let Some(obj) = v.get(key).or_else(|| v.get(key.to_uppercase()))
            && let Some(Value::Array(comments)) =
                obj.get("comments").or_else(|| obj.get("COMMENTS"))
        {
            object.data.comments = comments
                .iter()
                .filter_map(|c| c.as_str().map(|s| s.to_string()))
                .collect();
        }
    }

    Ok(cdm)
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
        json!(format_ccsds_datetime_in(
            &cdm.header.creation_date,
            &CCSDSTimeSystem::UTC
        )),
    );
    header.insert(key("ORIGINATOR", key_case), json!(&cdm.header.originator));
    if let Some(ref mf) = cdm.header.message_for {
        header.insert(key("MESSAGE_FOR", key_case), json!(mf));
    }
    header.insert(key("MESSAGE_ID", key_case), json!(&cdm.header.message_id));
    if !cdm.header.comments.is_empty() {
        header.insert("comments".into(), json!(cdm.header.comments));
    }
    root.insert("header".into(), Value::Object(header));

    // Relative metadata
    let rm = &cdm.relative_metadata;
    let mut rel = Map::new();
    if let Some(ref v) = rm.conjunction_id {
        rel.insert(key("CONJUNCTION_ID", key_case), json!(v));
    }
    rel.insert(
        key("TCA", key_case),
        json!(format_ccsds_datetime_in(&rm.tca, &CCSDSTimeSystem::UTC)),
    );
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
            json!(format_ccsds_datetime_in(v, &CCSDSTimeSystem::UTC)),
        );
    }
    if let Some(ref v) = rm.stop_screen_period {
        rel.insert(
            key("STOP_SCREEN_PERIOD", key_case),
            json!(format_ccsds_datetime_in(v, &CCSDSTimeSystem::UTC)),
        );
    }
    if let Some(ref v) = rm.screen_entry_time {
        rel.insert(
            key("SCREEN_ENTRY_TIME", key_case),
            json!(format_ccsds_datetime_in(v, &CCSDSTimeSystem::UTC)),
        );
    }
    if let Some(ref v) = rm.screen_exit_time {
        rel.insert(
            key("SCREEN_EXIT_TIME", key_case),
            json!(format_ccsds_datetime_in(v, &CCSDSTimeSystem::UTC)),
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
            json!(format_ccsds_datetime_in(v, &CCSDSTimeSystem::UTC)),
        );
    }
    if let Some(ref v) = rm.next_message_epoch {
        rel.insert(
            key("NEXT_MESSAGE_EPOCH", key_case),
            json!(format_ccsds_datetime_in(v, &CCSDSTimeSystem::UTC)),
        );
    }
    if !cdm.relative_metadata.comments.is_empty() {
        rel.insert("comments".into(), json!(cdm.relative_metadata.comments));
    }
    root.insert("relative_metadata".into(), Value::Object(rel));

    // Helper to build object JSON
    // CCSDS 508.0-B-1 subsection 6.2.3.4: all CDM time tags are UTC.
    let utc = |e: &crate::time::Epoch| format_ccsds_datetime_in(e, &CCSDSTimeSystem::UTC);

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
        if !m.comments.is_empty() {
            meta.insert("comments".into(), json!(m.comments));
        }
        o.insert("metadata".into(), Value::Object(meta));

        // OD parameters. Units and scaling mirror the KVN writer, since the
        // JSON reader flattens these back into KVN lines.
        if let Some(ref od) = d.od_parameters {
            let mut p = Map::new();
            if let Some(ref e) = od.time_lastob_start {
                p.insert(key("TIME_LASTOB_START", key_case), json!(utc(e)));
            }
            if let Some(ref e) = od.time_lastob_end {
                p.insert(key("TIME_LASTOB_END", key_case), json!(utc(e)));
            }
            if let Some(v) = od.recommended_od_span {
                p.insert(key("RECOMMENDED_OD_SPAN", key_case), json!(v));
            }
            if let Some(v) = od.actual_od_span {
                p.insert(key("ACTUAL_OD_SPAN", key_case), json!(v));
            }
            if let Some(v) = od.obs_available {
                p.insert(key("OBS_AVAILABLE", key_case), json!(v));
            }
            if let Some(v) = od.obs_used {
                p.insert(key("OBS_USED", key_case), json!(v));
            }
            if let Some(v) = od.tracks_available {
                p.insert(key("TRACKS_AVAILABLE", key_case), json!(v));
            }
            if let Some(v) = od.tracks_used {
                p.insert(key("TRACKS_USED", key_case), json!(v));
            }
            if let Some(v) = od.residuals_accepted {
                p.insert(key("RESIDUALS_ACCEPTED", key_case), json!(v));
            }
            if let Some(v) = od.weighted_rms {
                p.insert(key("WEIGHTED_RMS", key_case), json!(v));
            }
            if let Some(ref e) = od.od_epoch {
                p.insert(key("OD_EPOCH", key_case), json!(utc(e)));
            }
            if !od.comments.is_empty() {
                p.insert("comments".into(), json!(od.comments));
            }
            o.insert("od_parameters".into(), Value::Object(p));
        }

        // Additional parameters.
        if let Some(ref ap) = d.additional_parameters {
            let mut p = Map::new();
            {
                let mut put = |k: &str, v: Option<f64>| {
                    if let Some(v) = v {
                        p.insert(key(k, key_case), json!(v));
                    }
                };
                put("AREA_PC", ap.area_pc);
                put("AREA_PC_MIN", ap.area_pc_min);
                put("AREA_PC_MAX", ap.area_pc_max);
                put("AREA_DRG", ap.area_drg);
                put("AREA_SRP", ap.area_srp);
                put("OEB_Q1", ap.oeb_q1);
                put("OEB_Q2", ap.oeb_q2);
                put("OEB_Q3", ap.oeb_q3);
                put("OEB_QC", ap.oeb_qc);
                put("OEB_MAX", ap.oeb_max);
                put("OEB_INT", ap.oeb_int);
                put("OEB_MIN", ap.oeb_min);
                put("AREA_ALONG_OEB_MAX", ap.area_along_oeb_max);
                put("AREA_ALONG_OEB_INT", ap.area_along_oeb_int);
                put("AREA_ALONG_OEB_MIN", ap.area_along_oeb_min);
                put("RCS", ap.rcs);
                put("RCS_MIN", ap.rcs_min);
                put("RCS_MAX", ap.rcs_max);
                put("VM_ABSOLUTE", ap.vm_absolute);
                put("VM_APPARENT_MIN", ap.vm_apparent_min);
                put("VM_APPARENT", ap.vm_apparent);
                put("VM_APPARENT_MAX", ap.vm_apparent_max);
                put("REFLECTANCE", ap.reflectance);
                put("MASS", ap.mass);
                put("HBR", ap.hbr);
                put("CD_AREA_OVER_MASS", ap.cd_area_over_mass);
                put("CR_AREA_OVER_MASS", ap.cr_area_over_mass);
                put("THRUST_ACCELERATION", ap.thrust_acceleration);
                put("SEDR", ap.sedr);
                put("LEAD_TIME_REQD_BEFORE_TCA", ap.lead_time_reqd_before_tca);
                put("INCLINATION", ap.inclination);
                put("COV_CONFIDENCE", ap.cov_confidence);
                // Altitudes are stored in metres and written in kilometres.
                put("APOAPSIS_ALTITUDE", ap.apoapsis_altitude.map(|v| v / 1e3));
                put("PERIAPSIS_ALTITUDE", ap.periapsis_altitude.map(|v| v / 1e3));
            }
            if let Some(ref v) = ap.oeb_parent_frame {
                p.insert(key("OEB_PARENT_FRAME", key_case), json!(v));
            }
            if let Some(ref e) = ap.oeb_parent_frame_epoch {
                p.insert(key("OEB_PARENT_FRAME_EPOCH", key_case), json!(utc(e)));
            }
            if let Some(ref v) = ap.min_dv {
                p.insert(key("MIN_DV", key_case), json!(v.to_vec()));
            }
            if let Some(ref v) = ap.max_dv {
                p.insert(key("MAX_DV", key_case), json!(v.to_vec()));
            }
            if let Some(ref v) = ap.cov_confidence_method {
                p.insert(key("COV_CONFIDENCE_METHOD", key_case), json!(v));
            }
            if !ap.comments.is_empty() {
                p.insert("comments".into(), json!(ap.comments));
            }
            o.insert("additional_parameters".into(), Value::Object(p));
        }

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
        if !d.comments.is_empty() {
            o.insert("comments".into(), json!(d.comments));
        }
        if !d.state_vector.comments.is_empty() {
            sv.insert("comments".into(), json!(d.state_vector.comments));
        }
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

        // The ordered array carries only values, so the block's comments ride
        // alongside it in an object of their own.
        if !d.rtn_covariance.comments.is_empty() {
            let mut cov_obj = Map::new();
            cov_obj.insert("comments".into(), json!(d.rtn_covariance.comments));
            o.insert("rtn_covariance".into(), Value::Object(cov_obj));
        }

        Value::Object(o)
    };

    root.insert("object1".into(), build_object(&cdm.object1));
    root.insert("object2".into(), build_object(&cdm.object2));

    serde_json::to_string_pretty(&Value::Object(root))
        .map_err(|e| BraheError::Error(format!("CDM JSON serialization error: {}", e)))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {

    use crate::ccsds::common::{CCSDSFormat, CCSDSJsonKeyCase};
    use crate::ccsds::json::{parse_cdm_json, write_cdm_json};

    use serial_test::parallel;
    // ---- CDM ----

    #[test]
    #[parallel]
    fn test_cdm_json_round_trip_lowercase() {
        let cdm =
            crate::ccsds::cdm::CDM::from_file("test_assets/ccsds/cdm/CDMExample1.txt").unwrap();
        let json_str = write_cdm_json(&cdm, CCSDSJsonKeyCase::Lower).unwrap();
        let cdm2 = parse_cdm_json(&json_str).unwrap();

        assert_eq!(cdm.header.originator, cdm2.header.originator);
        assert_eq!(cdm.header.message_id, cdm2.header.message_id);
    }

    #[test]
    #[parallel]
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

    #[test]
    #[parallel]
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

    // =========================================================================
    // CDM — write with Bool in flatten
    // =========================================================================

    #[test]
    #[parallel]
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
    #[parallel]
    fn test_cdm_json_parse_malformed() {
        let result = parse_cdm_json("not valid json");
        assert!(result.is_err());
    }

    #[test]
    #[serial_test::parallel]
    fn test_cdm_json_round_trip_preserves_data_section_comments() {
        use crate::ccsds::cdm::CDM;

        // KVN puts nothing between the Data comment and the first sub-block
        // comment, so the parser attributes the run to the innermost block.
        // JSON keeps them apart, so this level survives there.
        let source = std::fs::read_to_string("test_assets/ccsds/cdm/CDMExample2.txt").unwrap();
        let mut cdm = CDM::from_str(&source).unwrap();
        cdm.object1.data.comments = vec!["Object1 Data".to_string()];
        cdm.object2.data.comments = vec!["Object2 Data".to_string()];

        let reparsed = CDM::from_str(&cdm.to_string(CCSDSFormat::JSON).unwrap()).unwrap();
        assert_eq!(reparsed.object1.data.comments, vec!["Object1 Data"]);
        assert_eq!(reparsed.object2.data.comments, vec!["Object2 Data"]);
    }

    #[test]
    #[serial_test::parallel]
    fn test_cdm_data_blocks_survive_every_encoding() {
        use crate::ccsds::cdm::CDM;

        // CDMExample2 carries OD parameters and additional parameters for both
        // objects; MIN_DV and MAX_DV are added because no fixture uses them.
        let source = std::fs::read_to_string("test_assets/ccsds/cdm/CDMExample2.txt").unwrap();
        let mut cdm = CDM::from_str(&source).unwrap();
        let ap = cdm.object1.data.additional_parameters.as_mut().unwrap();
        ap.min_dv = Some([0.1, 0.2, 0.3]);
        ap.max_dv = Some([1.1, 1.2, 1.3]);

        let od = cdm.object1.data.od_parameters.clone().unwrap();
        let ap = cdm.object1.data.additional_parameters.clone().unwrap();

        for format in [CCSDSFormat::KVN, CCSDSFormat::XML, CCSDSFormat::JSON] {
            let reparsed = CDM::from_str(&cdm.to_string(format).unwrap()).unwrap();

            let rod = reparsed
                .object1
                .data
                .od_parameters
                .as_ref()
                .unwrap_or_else(|| panic!("{:?} dropped the OD parameters block", format));
            assert_eq!(
                rod.obs_available, od.obs_available,
                "{:?} OBS_AVAILABLE",
                format
            );
            assert_eq!(rod.obs_used, od.obs_used, "{:?} OBS_USED", format);
            assert_eq!(rod.tracks_used, od.tracks_used, "{:?} TRACKS_USED", format);
            assert_eq!(
                rod.recommended_od_span, od.recommended_od_span,
                "{:?} RECOMMENDED_OD_SPAN",
                format
            );

            let rap = reparsed
                .object1
                .data
                .additional_parameters
                .as_ref()
                .unwrap_or_else(|| panic!("{:?} dropped the additional parameters block", format));
            assert_eq!(rap.area_pc, ap.area_pc, "{:?} AREA_PC", format);
            assert_eq!(rap.mass, ap.mass, "{:?} MASS", format);
            assert_eq!(
                rap.cd_area_over_mass, ap.cd_area_over_mass,
                "{:?} CD_AREA_OVER_MASS",
                format
            );
            assert_eq!(rap.sedr, ap.sedr, "{:?} SEDR", format);
            assert_eq!(rap.min_dv, ap.min_dv, "{:?} MIN_DV", format);
            assert_eq!(rap.max_dv, ap.max_dv, "{:?} MAX_DV", format);
        }
    }

    #[test]
    #[serial_test::parallel]
    fn test_cdm_json_round_trip_preserves_comments() {
        use crate::ccsds::cdm::CDM;

        let source = std::fs::read_to_string("test_assets/ccsds/cdm/CDMExample2.txt").unwrap();
        let cdm = CDM::from_str(&source).unwrap();
        let reparsed = CDM::from_str(&cdm.to_string(CCSDSFormat::JSON).unwrap()).unwrap();

        assert_eq!(
            reparsed.relative_metadata.comments,
            cdm.relative_metadata.comments
        );
        assert_eq!(
            reparsed.object1.metadata.comments,
            cdm.object1.metadata.comments
        );
        assert_eq!(
            reparsed.object1.data.state_vector.comments,
            cdm.object1.data.state_vector.comments
        );
        assert_eq!(
            reparsed.object1.data.rtn_covariance.comments,
            cdm.object1.data.rtn_covariance.comments
        );
        assert_eq!(
            reparsed.object2.metadata.comments,
            cdm.object2.metadata.comments
        );
    }
}
