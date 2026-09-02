/*!
 * JSON reader and writer for the Orbit Ephemeris Message (OEM).
 *
 * Reference: CCSDS 502.0-B-3 (Orbit Data Messages), section 5
 */

use serde_json::{Map, Value, json};

use crate::ccsds::common::{
    CCSDSJsonKeyCase, CCSDSTimeSystem, covariance_to_lower_triangular, format_ccsds_datetime_in,
    round_ccsds_value,
};
use crate::ccsds::error::ccsds_parse_error;
use crate::ccsds::json::common::{
    TIME_SYSTEM_FIRST, emit_json_comments, flatten_block, flatten_object_ordered, get_json_f64, key,
};
use crate::utils::errors::BraheError;

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
        flatten_block(&mut kvn_lines, header);
    }

    // Segments
    if let Some(Value::Array(segments)) = v.get("segments") {
        for seg in segments {
            // Metadata block
            kvn_lines.push("META_START".to_string());
            if let Some(meta) = seg.get("metadata") {
                emit_json_comments(&mut kvn_lines, meta);
                flatten_object_ordered(&mut kvn_lines, meta, &TIME_SYSTEM_FIRST);
            }
            kvn_lines.push("META_STOP".to_string());

            // Data-section comments sit between META_STOP and the first
            // ephemeris line.
            emit_json_comments(&mut kvn_lines, seg);

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
                        // EPOCH delimits one covariance from the next, so it
                        // leads: comments emitted ahead of it would be flushed
                        // into the preceding block. This is the order the KVN
                        // writer uses.
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
                        emit_json_comments(&mut kvn_lines, cov);
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
        json!(format_ccsds_datetime_in(
            &oem.header.creation_date,
            &CCSDSTimeSystem::UTC
        )),
    );
    header.insert(key("ORIGINATOR", key_case), json!(&oem.header.originator));
    if let Some(ref msg_id) = oem.header.message_id {
        header.insert(key("MESSAGE_ID", key_case), json!(msg_id));
    }
    if !oem.header.comments.is_empty() {
        header.insert("comments".into(), json!(oem.header.comments));
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
                json!(format_ccsds_datetime_in(epoch, &seg.metadata.time_system)),
            );
        }
        meta.insert(
            key("TIME_SYSTEM", key_case),
            json!(format!("{}", seg.metadata.time_system)),
        );
        meta.insert(
            key("START_TIME", key_case),
            json!(format_ccsds_datetime_in(
                &seg.metadata.start_time,
                &seg.metadata.time_system
            )),
        );
        if let Some(ref t) = seg.metadata.useable_start_time {
            meta.insert(
                key("USEABLE_START_TIME", key_case),
                json!(format_ccsds_datetime_in(t, &seg.metadata.time_system)),
            );
        }
        if let Some(ref t) = seg.metadata.useable_stop_time {
            meta.insert(
                key("USEABLE_STOP_TIME", key_case),
                json!(format_ccsds_datetime_in(t, &seg.metadata.time_system)),
            );
        }
        meta.insert(
            key("STOP_TIME", key_case),
            json!(format_ccsds_datetime_in(
                &seg.metadata.stop_time,
                &seg.metadata.time_system
            )),
        );
        if let Some(ref interp) = seg.metadata.interpolation {
            meta.insert(key("INTERPOLATION", key_case), json!(interp));
        }
        if let Some(deg) = seg.metadata.interpolation_degree {
            meta.insert(key("INTERPOLATION_DEGREE", key_case), json!(deg));
        }
        if !seg.metadata.comments.is_empty() {
            meta.insert("comments".into(), json!(seg.metadata.comments));
        }
        seg_obj.insert("metadata".into(), Value::Object(meta));

        // States (convert m → km, m/s → km/s for CCSDS standard units)
        let mut states = Vec::new();
        for sv in &seg.states {
            let mut state_obj = Map::new();
            state_obj.insert(
                key("EPOCH", key_case),
                json!(format_ccsds_datetime_in(
                    &sv.epoch,
                    &seg.metadata.time_system
                )),
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
        if !seg.comments.is_empty() {
            seg_obj.insert("comments".into(), json!(seg.comments));
        }
        seg_obj.insert("states".into(), Value::Array(states));

        // Covariances
        if !seg.covariances.is_empty() {
            let mut covs = Vec::new();
            for cov in &seg.covariances {
                let mut cov_obj = Map::new();
                if let Some(ref epoch) = cov.epoch {
                    cov_obj.insert(
                        key("EPOCH", key_case),
                        json!(format_ccsds_datetime_in(epoch, &seg.metadata.time_system)),
                    );
                }
                if let Some(ref frame) = cov.cov_ref_frame {
                    cov_obj.insert(key("COV_REF_FRAME", key_case), json!(format!("{}", frame)));
                }
                if !cov.comments.is_empty() {
                    cov_obj.insert("comments".into(), json!(cov.comments));
                }
                // Convert m² → km² (factor 1e-6)
                let values =
                    covariance_to_lower_triangular(&cov.matrix, 1e-6).map(round_ccsds_value);
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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {

    use crate::ccsds::common::{CCSDSFormat, CCSDSJsonKeyCase};
    use crate::ccsds::json::{parse_oem_json, write_oem_json};
    use crate::ccsds::oem::OEM;

    use serial_test::parallel;
    // ---- OEM ----

    #[test]
    #[parallel]
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
    #[parallel]
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
    #[parallel]
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
    #[parallel]
    fn test_oem_json_parse_malformed() {
        let result = parse_oem_json("not valid json");
        assert!(result.is_err());
    }

    // =========================================================================
    // Parse edge cases
    // =========================================================================

    #[test]
    #[parallel]
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
    #[parallel]
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
    #[parallel]
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

    // =========================================================================
    // Write OEM — optional field branches
    // =========================================================================

    #[test]
    #[parallel]
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
    #[parallel]
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
    #[parallel]
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
    #[parallel]
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
    // OEM with acceleration round-trip via test asset
    // =========================================================================

    #[test]
    #[parallel]
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
    // OEM JSON with lowercase state keys (x, y, z, x_dot, ...)
    // =========================================================================

    #[test]
    #[parallel]
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
    #[parallel]
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

    #[test]
    #[serial_test::parallel]
    fn test_oem_json_round_trip_preserves_comments() {
        let source =
            std::fs::read_to_string("test_assets/ccsds/oem/OEMExampleWithHeaderComment.txt")
                .unwrap();
        let oem = OEM::from_str(&source).unwrap();
        assert!(!oem.header.comments.is_empty());
        assert!(!oem.segments[0].comments.is_empty());

        let reparsed = OEM::from_str(&oem.to_string(CCSDSFormat::JSON).unwrap()).unwrap();
        assert_eq!(reparsed.header.comments, oem.header.comments);
        assert_eq!(reparsed.segments[0].comments, oem.segments[0].comments);
        assert_eq!(
            reparsed.segments[0].metadata.comments,
            oem.segments[0].metadata.comments
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_oem_json_keeps_each_covariance_comment_with_its_own_block() {
        // EPOCH delimits one covariance from the next in the KVN form the JSON
        // reader delegates to, so comments emitted ahead of it were flushed
        // into the preceding block and the second covariance received none.
        let source = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let mut oem = OEM::from_str(&source).unwrap();
        let segment = oem
            .segments
            .iter_mut()
            .find(|s| s.covariances.len() >= 2)
            .expect("fixture has a segment with two covariances");
        segment.covariances[0].comments = vec!["first covariance".to_string()];
        segment.covariances[1].comments = vec!["second covariance".to_string()];

        let reparsed = OEM::from_str(&oem.to_string(CCSDSFormat::JSON).unwrap()).unwrap();
        let segment = reparsed
            .segments
            .iter()
            .find(|s| s.covariances.len() >= 2)
            .unwrap();
        assert_eq!(segment.covariances[0].comments, vec!["first covariance"]);
        assert_eq!(segment.covariances[1].comments, vec!["second covariance"]);
    }
}
