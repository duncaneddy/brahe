/*!
 * Shared JSON key casing and KVN flattening helpers.
 *
 * The JSON readers flatten a document into KVN-style lines and hand them to
 * the KVN parser, so the field dispatch is written once. The flatteners, the
 * comment emitters, and the key-casing helper they all use live here.
 */

use serde_json::{Map, Value, json};

use crate::ccsds::common::{CCSDSJsonKeyCase, covariance_to_lower_triangular, round_ccsds_value};

/// Metadata keys that must be flattened before any other.
///
/// CCSDS 502.0-B-3 subsection 7.5.11 reads every epoch in the message's
/// `TIME_SYSTEM`, and the KVN parsers these readers delegate to apply the
/// scale declared so far, so the declaration has to precede the epochs it
/// governs. JSON object keys arrive in an order that would otherwise put
/// `START_TIME` and `REF_FRAME_EPOCH` ahead of it.
pub(super) const TIME_SYSTEM_FIRST: [&str; 1] = ["TIME_SYSTEM"];

/// Convert a CCSDS keyword to the appropriate case for JSON output.
///
/// Container/structural keys should NOT use this function — they are always lowercase.
/// This function only applies to CCSDS data field keywords.
pub(super) fn key(name: &str, case: CCSDSJsonKeyCase) -> String {
    match case {
        CCSDSJsonKeyCase::Lower => name.to_lowercase(),
        CCSDSJsonKeyCase::Upper => name.to_uppercase(),
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Get an f64 value from a JSON object, handling both number and string types.
pub(super) fn get_json_f64(obj: &Map<String, Value>, key: &str) -> Option<f64> {
    obj.get(key).and_then(|v| match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    })
}

/// Write covariance matrix as named CX_*/CY_*/CZ_* keys in a JSON object.
pub(super) fn write_json_covariance_elements(
    cv: &mut Map<String, Value>,
    matrix: &nalgebra::SMatrix<f64, 6, 6>,
    key_case: CCSDSJsonKeyCase,
) {
    let values = covariance_to_lower_triangular(matrix, 1e-6).map(round_ccsds_value);
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
pub(super) fn emit_kvn(lines: &mut Vec<String>, ukey: &str, val: &Value) {
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
pub(super) fn flatten_object(lines: &mut Vec<String>, obj: &Value) {
    if let Value::Object(map) = obj {
        for (key, val) in map {
            let ukey = key.to_uppercase();
            if ukey == COMMENTS_KEY {
                continue;
            }
            emit_kvn(lines, &ukey, val);
        }
    }
}

/// Flatten a JSON object into KVN-style key=value lines, skipping the given
/// (already-uppercased) keys.
pub(super) fn flatten_object_skip(lines: &mut Vec<String>, obj: &Value, skip: &[&str]) {
    if let Value::Object(map) = obj {
        for (key, val) in map {
            let ukey = key.to_uppercase();
            if skip.contains(&ukey.as_str()) {
                continue;
            }
            emit_kvn(lines, &ukey, val);
        }
    }
}

/// JSON key holding a block's comment array, in either case.
pub(super) const COMMENTS_KEY: &str = "COMMENTS";

/// Emit a block's `comments` array as CCSDS `COMMENT` lines.
///
/// The KVN parsers attribute a comment to the block whose keywords follow it,
/// so these are emitted immediately ahead of the block being flattened. Where a
/// keyword delimits one block from the next — `EPOCH` inside an OEM covariance
/// section — the comments follow that keyword instead, so they are not flushed
/// into the preceding block.
///
/// # Arguments
///
/// * `lines` - The KVN line buffer being built.
/// * `obj` - The JSON object that may carry a `comments` array. Objects
///   without one, and values that are not objects, are ignored.
///
/// # Returns
///
/// Nothing; `lines` is extended in place.
///
/// # Examples
///
/// ```ignore
/// let mut lines = Vec::new();
/// let block = serde_json::json!({"comments": ["first", "second"]});
/// emit_json_comments(&mut lines, &block);
/// assert_eq!(lines, ["COMMENT first", "COMMENT second"]);
/// ```
pub(super) fn emit_json_comments(lines: &mut Vec<String>, obj: &Value) {
    if let Value::Object(map) = obj
        && let Some(Value::Array(comments)) = map.get("comments").or_else(|| map.get("COMMENTS"))
    {
        for comment in comments {
            if let Value::String(text) = comment {
                lines.push(format!("COMMENT {}", text));
            }
        }
    }
}

/// Emit a block's comments and then its keywords.
///
/// # Arguments
///
/// * `lines` - The KVN line buffer being built.
/// * `obj` - The JSON object holding the block's comments and keywords.
///
/// # Returns
///
/// Nothing; `lines` is extended in place.
///
/// # Examples
///
/// ```ignore
/// let mut lines = Vec::new();
/// let block = serde_json::json!({"comments": ["note"], "ORIGINATOR": "NASA/JPL"});
/// flatten_block(&mut lines, &block);
/// assert_eq!(lines, ["COMMENT note", "ORIGINATOR = NASA/JPL"]);
/// ```
pub(super) fn flatten_block(lines: &mut Vec<String>, obj: &Value) {
    emit_json_comments(lines, obj);
    flatten_object(lines, obj);
}

/// Flatten a JSON object, emitting priority keys first.
///
/// Some KVN parsers expect certain keys to appear before others (e.g.,
/// SEMI_MAJOR_AXIS before INCLINATION). This helper emits the listed
/// priority keys first, then remaining keys in default order.
pub(super) fn flatten_object_ordered(lines: &mut Vec<String>, obj: &Value, priority_keys: &[&str]) {
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
            if ukey == COMMENTS_KEY || priority_keys.iter().any(|&pk| pk == ukey) {
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

    use serial_test::parallel;
    // ---- key helper ----

    #[test]
    #[parallel]
    fn test_key_case_conversion() {
        assert_eq!(key("OBJECT_NAME", CCSDSJsonKeyCase::Lower), "object_name");
        assert_eq!(key("OBJECT_NAME", CCSDSJsonKeyCase::Upper), "OBJECT_NAME");
        assert_eq!(key("x", CCSDSJsonKeyCase::Upper), "X");
        assert_eq!(key("X", CCSDSJsonKeyCase::Lower), "x");
    }

    #[test]
    #[parallel]
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
    // Helper functions
    // =========================================================================

    #[test]
    #[parallel]
    fn test_emit_kvn_bool_values() {
        let mut lines = Vec::new();
        emit_kvn(&mut lines, "FLAG_TRUE", &json!(true));
        emit_kvn(&mut lines, "FLAG_FALSE", &json!(false));

        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "FLAG_TRUE = YES");
        assert_eq!(lines[1], "FLAG_FALSE = NO");
    }

    #[test]
    #[parallel]
    fn test_emit_kvn_null_skipped() {
        let mut lines = Vec::new();
        emit_kvn(&mut lines, "NULL_KEY", &json!(null));
        assert!(lines.is_empty());
    }

    #[test]
    #[parallel]
    fn test_emit_kvn_array_values() {
        let mut lines = Vec::new();
        emit_kvn(&mut lines, "ARR", &json!([1, 2, 3]));
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "ARR = 1 2 3");
    }

    #[test]
    #[parallel]
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
    #[parallel]
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
}
