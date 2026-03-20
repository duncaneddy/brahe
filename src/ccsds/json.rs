/*!
 * CCSDS JSON format support.
 *
 * Stub — implemented in Stage 2 (OEM), Stage 5 (OMM), Stage 6 (OPM).
 */

use crate::ccsds::error::ccsds_parse_error;
use crate::utils::errors::BraheError;

/// Parse an OEM message from JSON format.
pub fn parse_oem_json(_content: &str) -> Result<crate::ccsds::oem::OEM, BraheError> {
    Err(ccsds_parse_error("OEM", "JSON parsing not yet implemented"))
}

/// Parse an OMM message from JSON format.
pub fn parse_omm_json(_content: &str) -> Result<crate::ccsds::omm::OMM, BraheError> {
    Err(ccsds_parse_error("OMM", "JSON parsing not yet implemented"))
}

/// Parse an OPM message from JSON format.
pub fn parse_opm_json(_content: &str) -> Result<crate::ccsds::opm::OPM, BraheError> {
    Err(ccsds_parse_error("OPM", "JSON parsing not yet implemented"))
}

/// Write an OEM message to JSON format.
pub fn write_oem_json(_oem: &crate::ccsds::oem::OEM) -> Result<String, BraheError> {
    Err(BraheError::Error(
        "OEM JSON writer not yet implemented".to_string(),
    ))
}

/// Write an OMM message to JSON format.
pub fn write_omm_json(_omm: &crate::ccsds::omm::OMM) -> Result<String, BraheError> {
    Err(BraheError::Error(
        "OMM JSON writer not yet implemented".to_string(),
    ))
}

/// Write an OPM message to JSON format.
pub fn write_opm_json(_opm: &crate::ccsds::opm::OPM) -> Result<String, BraheError> {
    Err(BraheError::Error(
        "OPM JSON writer not yet implemented".to_string(),
    ))
}

/// Parse a CDM message from JSON format.
///
/// Deserializes a JSON object into key-value pairs, reconstructs KVN-like
/// representation, and delegates to the KVN parser.
pub fn parse_cdm_json(content: &str) -> Result<crate::ccsds::cdm::CDM, BraheError> {
    let v: serde_json::Value = serde_json::from_str(content)
        .map_err(|e| ccsds_parse_error("CDM", &format!("JSON parse error: {}", e)))?;

    let mut kvn_lines: Vec<String> = Vec::new();

    // Recursively flatten JSON objects into KVN key=value lines.
    // Keys that are already uppercase CCSDS keywords get emitted directly.
    // Nested object keys (like "header", "metadata", "state_vector") are
    // traversed but not emitted as keywords themselves.
    fn flatten(lines: &mut Vec<String>, obj: &serde_json::Value) {
        if let serde_json::Value::Object(map) = obj {
            for (key, val) in map {
                // Determine the KVN keyword: use uppercase version of the key
                let ukey = key.to_uppercase();

                // Skip structural container keys that don't map to KVN keywords
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
                    serde_json::Value::Object(_) if is_container => {
                        flatten(lines, val);
                    }
                    serde_json::Value::Object(_) => {
                        // Non-container nested object — recurse
                        flatten(lines, val);
                    }
                    serde_json::Value::Null => {}
                    serde_json::Value::String(s) => {
                        lines.push(format!("{} = {}", ukey, s));
                    }
                    serde_json::Value::Number(n) => {
                        lines.push(format!("{} = {}", ukey, n));
                    }
                    serde_json::Value::Array(arr) => {
                        let parts: Vec<String> = arr.iter().map(|a| a.to_string()).collect();
                        lines.push(format!("{} = {}", ukey, parts.join(" ")));
                    }
                    serde_json::Value::Bool(b) => {
                        lines.push(format!("{} = {}", ukey, if *b { "YES" } else { "NO" }));
                    }
                }
            }
        }
    }

    // Helper to flatten a CDM object section in correct order
    fn flatten_cdm_object(lines: &mut Vec<String>, obj: &serde_json::Value) {
        // OBJECT keyword MUST come first to set context for the KVN parser
        if let Some(meta) = obj.get("metadata") {
            if let Some(s) = meta.get("OBJECT").and_then(|v| v.as_str()) {
                lines.push(format!("OBJECT = {}", s));
            }
            // Then emit remaining metadata fields
            flatten(lines, meta);
        }
        // State vector
        if let Some(sv) = obj.get("state_vector") {
            flatten(lines, sv);
        }
        // RTN covariance (ordered array format)
        if let Some(serde_json::Value::Array(arr)) = obj.get("rtn_covariance_ordered") {
            for pair in arr {
                // Each pair is [key_string, number_value]
                let Some(kv) = pair.as_array() else { continue };
                let Some(serde_json::Value::String(key)) = kv.first() else {
                    continue;
                };
                let Some(val) = kv.get(1).and_then(|v| v.as_f64()) else {
                    continue;
                };
                lines.push(format!("{} = {}", key, val));
            }
        }
        // OD parameters, additional parameters, etc.
        if let Some(od) = obj.get("od_parameters") {
            flatten(lines, od);
        }
        if let Some(ap) = obj.get("additional_parameters") {
            flatten(lines, ap);
        }
    }

    // Flatten each section in correct KVN order: header → relative_metadata → object1 → object2
    if let Some(header) = v.get("header") {
        flatten(&mut kvn_lines, header);
    }
    if let Some(rel) = v.get("relative_metadata") {
        flatten(&mut kvn_lines, rel);
    }
    if let Some(obj1) = v.get("object1") {
        flatten_cdm_object(&mut kvn_lines, obj1);
    }
    if let Some(obj2) = v.get("object2") {
        flatten_cdm_object(&mut kvn_lines, obj2);
    }
    if let Some(ud) = v.get("user_defined") {
        flatten(&mut kvn_lines, ud);
    }

    let kvn_content = kvn_lines.join("\n");
    crate::ccsds::kvn::parse_cdm(&kvn_content)
}

/// Write a CDM message to JSON format.
///
/// Serializes the CDM to a structured JSON object with field names matching
/// the CCSDS keyword names. All values in SI units. Epochs formatted as
/// ISO 8601 UTC strings.
pub fn write_cdm_json(cdm: &crate::ccsds::cdm::CDM) -> Result<String, BraheError> {
    use crate::ccsds::common::{covariance9x9_to_lower_triangular, format_ccsds_datetime};
    use serde_json::{Map, Value, json};

    let mut root = Map::new();

    // Header
    let mut header = Map::new();
    header.insert("CCSDS_CDM_VERS".into(), json!(cdm.header.format_version));
    header.insert(
        "CREATION_DATE".into(),
        json!(format_ccsds_datetime(&cdm.header.creation_date)),
    );
    header.insert("ORIGINATOR".into(), json!(&cdm.header.originator));
    if let Some(ref mf) = cdm.header.message_for {
        header.insert("MESSAGE_FOR".into(), json!(mf));
    }
    header.insert("MESSAGE_ID".into(), json!(&cdm.header.message_id));
    root.insert("header".into(), Value::Object(header));

    // Relative metadata
    let rm = &cdm.relative_metadata;
    let mut rel = Map::new();
    rel.insert("TCA".into(), json!(format_ccsds_datetime(&rm.tca)));
    rel.insert("MISS_DISTANCE".into(), json!(rm.miss_distance));
    if let Some(v) = rm.relative_speed {
        rel.insert("RELATIVE_SPEED".into(), json!(v));
    }
    if let Some(v) = rm.relative_position_r {
        rel.insert("RELATIVE_POSITION_R".into(), json!(v));
    }
    if let Some(v) = rm.relative_position_t {
        rel.insert("RELATIVE_POSITION_T".into(), json!(v));
    }
    if let Some(v) = rm.relative_position_n {
        rel.insert("RELATIVE_POSITION_N".into(), json!(v));
    }
    if let Some(v) = rm.relative_velocity_r {
        rel.insert("RELATIVE_VELOCITY_R".into(), json!(v));
    }
    if let Some(v) = rm.relative_velocity_t {
        rel.insert("RELATIVE_VELOCITY_T".into(), json!(v));
    }
    if let Some(v) = rm.relative_velocity_n {
        rel.insert("RELATIVE_VELOCITY_N".into(), json!(v));
    }
    if let Some(v) = rm.collision_probability {
        rel.insert("COLLISION_PROBABILITY".into(), json!(v));
    }
    if let Some(ref s) = rm.collision_probability_method {
        rel.insert("COLLISION_PROBABILITY_METHOD".into(), json!(s));
    }
    root.insert("relative_metadata".into(), Value::Object(rel));

    // Helper to build object JSON
    let build_object = |obj: &crate::ccsds::cdm::CDMObject| -> Value {
        let m = &obj.metadata;
        let d = &obj.data;
        let mut o = Map::new();

        // Metadata
        let mut meta = Map::new();
        meta.insert("OBJECT".into(), json!(&m.object));
        meta.insert("OBJECT_DESIGNATOR".into(), json!(&m.object_designator));
        meta.insert("CATALOG_NAME".into(), json!(&m.catalog_name));
        meta.insert("OBJECT_NAME".into(), json!(&m.object_name));
        meta.insert(
            "INTERNATIONAL_DESIGNATOR".into(),
            json!(&m.international_designator),
        );
        if let Some(ref v) = m.object_type {
            meta.insert("OBJECT_TYPE".into(), json!(v));
        }
        meta.insert("EPHEMERIS_NAME".into(), json!(&m.ephemeris_name));
        meta.insert("COVARIANCE_METHOD".into(), json!(&m.covariance_method));
        meta.insert("MANEUVERABLE".into(), json!(&m.maneuverable));
        meta.insert("REF_FRAME".into(), json!(format!("{}", m.ref_frame)));
        o.insert("metadata".into(), Value::Object(meta));

        // State vector (in km/km/s for file-format consistency)
        let mut sv = Map::new();
        sv.insert("X".into(), json!(d.state_vector.position[0] / 1e3));
        sv.insert("Y".into(), json!(d.state_vector.position[1] / 1e3));
        sv.insert("Z".into(), json!(d.state_vector.position[2] / 1e3));
        sv.insert("X_DOT".into(), json!(d.state_vector.velocity[0] / 1e3));
        sv.insert("Y_DOT".into(), json!(d.state_vector.velocity[1] / 1e3));
        sv.insert("Z_DOT".into(), json!(d.state_vector.velocity[2] / 1e3));
        o.insert("state_vector".into(), Value::Object(sv));

        // RTN covariance — emit each element as a named key=value in order.
        // We write them as a flat map alongside the state vector to ensure
        // the KVN parser picks them up in the right order when flattened.
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
        // Store as ordered array of [name, value] pairs to preserve ordering
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
