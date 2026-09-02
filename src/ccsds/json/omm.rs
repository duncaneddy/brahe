/*!
 * JSON reader and writer for the Orbit Mean-elements Message (OMM).
 *
 * Reference: CCSDS 502.0-B-3 (Orbit Data Messages), section 4
 */

use serde_json::{Map, Value, json};

use crate::ccsds::common::{CCSDSJsonKeyCase, CCSDSTimeSystem, format_ccsds_datetime_in};
use crate::ccsds::error::ccsds_parse_error;
use crate::ccsds::json::common::{
    TIME_SYSTEM_FIRST, emit_json_comments, flatten_object, flatten_object_ordered, key,
    write_json_covariance_elements,
};
use crate::utils::errors::BraheError;

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
            emit_json_comments(&mut kvn_lines, obj);
            if *section == "metadata" {
                flatten_object_ordered(&mut kvn_lines, obj, &TIME_SYSTEM_FIRST);
            } else if *section == "covariance" {
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
        json!(format_ccsds_datetime_in(
            &omm.header.creation_date,
            &CCSDSTimeSystem::UTC
        )),
    );
    header.insert(key("ORIGINATOR", key_case), json!(&omm.header.originator));
    if let Some(ref msg_id) = omm.header.message_id {
        header.insert(key("MESSAGE_ID", key_case), json!(msg_id));
    }
    if !omm.header.comments.is_empty() {
        header.insert("comments".into(), json!(omm.header.comments));
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
            json!(format_ccsds_datetime_in(epoch, &omm.metadata.time_system)),
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
    if !omm.metadata.comments.is_empty() {
        meta.insert("comments".into(), json!(omm.metadata.comments));
    }
    root.insert("metadata".into(), Value::Object(meta));

    // Mean elements (units stored as file-native: rev/day, degrees, km)
    let mut me = Map::new();
    me.insert(
        key("EPOCH", key_case),
        json!(format_ccsds_datetime_in(
            &omm.mean_elements.epoch,
            &omm.metadata.time_system
        )),
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
    if !omm.mean_elements.comments.is_empty() {
        me.insert("comments".into(), json!(omm.mean_elements.comments));
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
        if !tle.comments.is_empty() {
            tp.insert("comments".into(), json!(tle.comments));
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
        if !sc.comments.is_empty() {
            sp.insert("comments".into(), json!(sc.comments));
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
        if !cov.comments.is_empty() {
            cv.insert("comments".into(), json!(cov.comments));
        }
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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {

    use crate::ccsds::common::{CCSDSFormat, CCSDSJsonKeyCase};
    use crate::ccsds::json::{parse_omm_json, write_omm_json};

    use crate::ccsds::omm::OMM;

    use serial_test::parallel;
    // ---- OMM ----

    #[test]
    #[parallel]
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
    #[parallel]
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
    #[parallel]
    fn test_omm_json_parse_malformed() {
        let result = parse_omm_json("not valid json");
        assert!(result.is_err());
    }

    #[test]
    #[parallel]
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

    // =========================================================================
    // Write OMM — optional field branches
    // =========================================================================

    #[test]
    #[parallel]
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
    #[parallel]
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
    #[parallel]
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

    #[test]
    #[serial_test::parallel]
    fn test_omm_json_round_trip_preserves_comments() {
        let omm = OMM::from_str(
            &std::fs::read_to_string("test_assets/ccsds/omm/OMM-section-comments.txt").unwrap(),
        )
        .unwrap();
        let reparsed = OMM::from_str(&omm.to_string(CCSDSFormat::JSON).unwrap()).unwrap();

        assert_eq!(reparsed.header.comments, vec!["header comment"]);
        assert_eq!(reparsed.metadata.comments, vec!["metadata comment"]);
        assert_eq!(
            reparsed.mean_elements.comments,
            vec!["mean element comment"]
        );
        assert_eq!(
            reparsed.tle_parameters.as_ref().unwrap().comments,
            vec!["tle comment"]
        );
        assert_eq!(
            reparsed.spacecraft_parameters.as_ref().unwrap().comments,
            vec!["spacecraft comment"]
        );
    }
}
