/*!
 * JSON reader and writer for the Orbit Parameter Message (OPM).
 *
 * Reference: CCSDS 502.0-B-3 (Orbit Data Messages), section 3
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
            emit_json_comments(&mut kvn_lines, obj);
            if *section == "metadata" {
                flatten_object_ordered(&mut kvn_lines, obj, &TIME_SYSTEM_FIRST);
            } else if *section == "keplerian_elements" {
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
            emit_json_comments(&mut kvn_lines, man);
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
        json!(format_ccsds_datetime_in(
            &opm.header.creation_date,
            &CCSDSTimeSystem::UTC
        )),
    );
    header.insert(key("ORIGINATOR", key_case), json!(&opm.header.originator));
    if let Some(ref msg_id) = opm.header.message_id {
        header.insert(key("MESSAGE_ID", key_case), json!(msg_id));
    }
    if !opm.header.comments.is_empty() {
        header.insert("comments".into(), json!(opm.header.comments));
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
            json!(format_ccsds_datetime_in(epoch, &opm.metadata.time_system)),
        );
    }
    meta.insert(
        key("TIME_SYSTEM", key_case),
        json!(format!("{}", opm.metadata.time_system)),
    );
    if !opm.metadata.comments.is_empty() {
        meta.insert("comments".into(), json!(opm.metadata.comments));
    }
    root.insert("metadata".into(), Value::Object(meta));

    // State vector (convert m → km, m/s → km/s)
    let mut sv = Map::new();
    sv.insert(
        key("EPOCH", key_case),
        json!(format_ccsds_datetime_in(
            &opm.state_vector.epoch,
            &opm.metadata.time_system
        )),
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
    if !opm.state_vector.comments.is_empty() {
        sv.insert("comments".into(), json!(opm.state_vector.comments));
    }
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
        if !kep.comments.is_empty() {
            ke.insert("comments".into(), json!(kep.comments));
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
        if !sc.comments.is_empty() {
            sp.insert("comments".into(), json!(sc.comments));
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
        if !cov.comments.is_empty() {
            cv.insert("comments".into(), json!(cov.comments));
        }
        root.insert("covariance".into(), Value::Object(cv));
    }

    // Maneuvers (convert m/s → km/s)
    if !opm.maneuvers.is_empty() {
        let mut mans = Vec::new();
        for man in &opm.maneuvers {
            let mut m = Map::new();
            m.insert(
                key("MAN_EPOCH_IGNITION", key_case),
                json!(format_ccsds_datetime_in(
                    &man.epoch_ignition,
                    &opm.metadata.time_system
                )),
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
            if !man.comments.is_empty() {
                m.insert("comments".into(), json!(man.comments));
            }
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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {

    use crate::ccsds::common::{CCSDSFormat, CCSDSJsonKeyCase};
    use crate::ccsds::json::{parse_opm_json, write_opm_json};

    use crate::ccsds::opm::OPM;

    use serial_test::parallel;
    // ---- OPM ----

    #[test]
    #[parallel]
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
    #[parallel]
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
    #[parallel]
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
    #[parallel]
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
    #[parallel]
    fn test_opm_json_parse_malformed() {
        let result = parse_opm_json("not valid json");
        assert!(result.is_err());
    }

    #[test]
    #[parallel]
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

    // =========================================================================
    // Write OPM — optional field branches
    // =========================================================================

    #[test]
    #[parallel]
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
    #[parallel]
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
    #[parallel]
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
    #[parallel]
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
    #[parallel]
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
    // OPM with keplerian GM conversion
    // =========================================================================

    #[test]
    #[parallel]
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

    #[test]
    #[serial_test::parallel]
    fn test_opm_json_round_trip_preserves_comments() {
        let opm = OPM::from_str(
            &std::fs::read_to_string("test_assets/ccsds/opm/OPM-section-comments.txt").unwrap(),
        )
        .unwrap();
        let reparsed = OPM::from_str(&opm.to_string(CCSDSFormat::JSON).unwrap()).unwrap();

        assert_eq!(reparsed.header.comments, vec!["header comment"]);
        assert_eq!(reparsed.metadata.comments, vec!["metadata comment"]);
        assert_eq!(reparsed.state_vector.comments, vec!["state vector comment"]);
        assert_eq!(reparsed.maneuvers.len(), 2);
        assert_eq!(
            reparsed.maneuvers[0].comments,
            vec!["first maneuver comment"]
        );
        assert_eq!(
            reparsed.maneuvers[1].comments,
            vec!["second maneuver comment"]
        );
    }
}
