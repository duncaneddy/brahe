/*!
 * JSON reader and writer for the Attitude Ephemeris Message (AEM).
 *
 * Reference: CCSDS 504.0-B-2 (Attitude Data Messages), section 4
 */

use serde_json::{Map, Value, json};

use crate::ccsds::aem::{AEM, AEMAttitudeData, AEMAttitudeType};
use crate::ccsds::common::{
    CCSDSJsonKeyCase, CCSDSTimeSystem, format_ccsds_datetime, format_ccsds_datetime_in,
    format_euler_rot_seq,
};
use crate::ccsds::error::{ccsds_missing_field, ccsds_parse_error};
use crate::ccsds::json::common::{
    emit_json_comments, emit_kvn, flatten_object_skip, get_json_f64, key,
};
use crate::ccsds::kvn::parse_aem;
use crate::constants::RAD2DEG;
use crate::time::Epoch;
use crate::utils::errors::BraheError;

// =============================================================================
// AEM JSON
// =============================================================================

/// Ordered data-line column keywords for a given AEM `ATTITUDE_TYPE`
/// (504.0-B-2 table 4-4), excluding `EPOCH` which is handled separately.
fn aem_data_column_keys(
    attitude_type: AEMAttitudeType,
) -> &'static [&'static str] {
    use AEMAttitudeType::*;
    match attitude_type {
        Quaternion => &["Q1", "Q2", "Q3", "QC"],
        QuaternionDerivative => &[
            "Q1", "Q2", "Q3", "QC", "Q1_DOT", "Q2_DOT", "Q3_DOT", "QC_DOT",
        ],
        QuaternionAngVel => &["Q1", "Q2", "Q3", "QC", "ANGVEL_X", "ANGVEL_Y", "ANGVEL_Z"],
        EulerAngle => &["ANGLE_1", "ANGLE_2", "ANGLE_3"],
        EulerAngleDerivative => &[
            "ANGLE_1",
            "ANGLE_2",
            "ANGLE_3",
            "ANGLE_1_DOT",
            "ANGLE_2_DOT",
            "ANGLE_3_DOT",
        ],
        EulerAngleAngVel => &[
            "ANGLE_1", "ANGLE_2", "ANGLE_3", "ANGVEL_X", "ANGVEL_Y", "ANGVEL_Z",
        ],
        Spin => &["SPIN_ALPHA", "SPIN_DELTA", "SPIN_ANGLE", "SPIN_ANGLE_VEL"],
        SpinNutation => &[
            "SPIN_ALPHA",
            "SPIN_DELTA",
            "SPIN_ANGLE",
            "SPIN_ANGLE_VEL",
            "NUTATION",
            "NUTATION_PER",
            "NUTATION_PHASE",
        ],
        SpinNutationMom => &[
            "SPIN_ALPHA",
            "SPIN_DELTA",
            "SPIN_ANGLE",
            "SPIN_ANGLE_VEL",
            "MOMENTUM_ALPHA",
            "MOMENTUM_DELTA",
            "NUTATION_VEL",
        ],
    }
}

/// Builds the fixed-order numeric columns (as KVN-ready strings) for one AEM
/// data line from a JSON state object, matching table 4-4's column layout
/// for `attitude_type`.
fn aem_json_state_columns(
    obj: &Map<String, Value>,
    attitude_type: AEMAttitudeType,
) -> Result<Vec<String>, BraheError> {
    aem_data_column_keys(attitude_type)
        .iter()
        .map(|k| {
            get_json_f64(obj, k)
                .or_else(|| get_json_f64(obj, &k.to_lowercase()))
                .map(|v| v.to_string())
                .ok_or_else(|| ccsds_missing_field("AEM", k))
        })
        .collect()
}

/// Parse an AEM message from JSON format.
///
/// Flattens the JSON structure into KVN-style lines with META_START/META_STOP
/// and DATA_START/DATA_STOP delimiters for each segment, then delegates to
/// the KVN parser.
pub fn parse_aem_json(content: &str) -> Result<AEM, BraheError> {

    let v: Value = serde_json::from_str(content)
        .map_err(|e| ccsds_parse_error("AEM", &format!("JSON parse error: {}", e)))?;

    let mut kvn_lines: Vec<String> = Vec::new();

    // Header comments are only attributed to the header by the KVN parser
    // while it has not yet seen CLASSIFICATION/CREATION_DATE/ORIGINATOR/
    // MESSAGE_ID, so they must be emitted ahead of those keywords.
    if let Some(obj) = v.get("header").or_else(|| v.get("HEADER")) {
        emit_json_comments(&mut kvn_lines, obj);
        flatten_object_skip(&mut kvn_lines, obj, &["COMMENTS"]);
    }

    if let Some(Value::Array(segments)) = v.get("segments").or_else(|| v.get("SEGMENTS")) {
        for seg in segments {
            let meta = seg.get("metadata").or_else(|| seg.get("METADATA"));

            kvn_lines.push("META_START".to_string());
            let attitude_type = if let Some(meta_obj) = meta {
                emit_json_comments(&mut kvn_lines, meta_obj);
                // serde_json's Map sorts keys alphabetically, which would
                // place START_TIME/STOP_TIME/USEABLE_* ahead of TIME_SYSTEM;
                // the KVN parser resolves those epochs using whichever
                // TIME_SYSTEM it has already seen (defaulting to UTC), so
                // TIME_SYSTEM must be emitted first. OBJECT_NAME is ordered
                // first purely for readability. COMMENTS is skipped since
                // it was already emitted above via emit_json_comments.
                if let Value::Object(map) = meta_obj {
                    if let Some(val) = map.get("TIME_SYSTEM").or_else(|| map.get("time_system")) {
                        emit_kvn(&mut kvn_lines, "TIME_SYSTEM", val);
                    }
                    if let Some(val) = map.get("OBJECT_NAME").or_else(|| map.get("object_name")) {
                        emit_kvn(&mut kvn_lines, "OBJECT_NAME", val);
                    }
                    for (k, val) in map {
                        let ukey = k.to_uppercase();
                        if ukey == "TIME_SYSTEM" || ukey == "OBJECT_NAME" || ukey == "COMMENTS" {
                            continue;
                        }
                        emit_kvn(&mut kvn_lines, &ukey, val);
                    }
                }

                meta_obj
                    .get("ATTITUDE_TYPE")
                    .or_else(|| meta_obj.get("attitude_type"))
                    .and_then(|v| v.as_str())
                    .map(AEMAttitudeType::parse)
                    .transpose()?
            } else {
                None
            };
            let attitude_type =
                attitude_type.ok_or_else(|| ccsds_missing_field("AEM", "ATTITUDE_TYPE"))?;
            kvn_lines.push("META_STOP".to_string());

            kvn_lines.push("DATA_START".to_string());
            if let Some(Value::Array(comments)) =
                seg.get("comments").or_else(|| seg.get("COMMENTS"))
            {
                for c in comments {
                    if let Some(s) = c.as_str() {
                        kvn_lines.push(format!("COMMENT {}", s));
                    }
                }
            }

            if let Some(Value::Array(states)) = seg.get("states").or_else(|| seg.get("STATES")) {
                for state in states {
                    if let Some(obj) = state.as_object() {
                        let epoch = obj
                            .get("EPOCH")
                            .or_else(|| obj.get("epoch"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let cols = aem_json_state_columns(obj, attitude_type)?;
                        kvn_lines.push(format!("{} {}", epoch, cols.join(" ")));
                    }
                }
            }
            kvn_lines.push("DATA_STOP".to_string());
        }
    }

    let kvn_content = kvn_lines.join("\n");
    parse_aem(&kvn_content)
}

/// Write an AEM message to JSON format.
///
/// Values are written in wire units (degrees for angles/rates, scalar-last
/// quaternion component keys `Q1..QC`/`Q1_DOT..QC_DOT`), matching the KVN
/// representation so that [`parse_aem_json`]'s flatten-to-KVN-lines path
/// reproduces them unchanged.
pub fn write_aem_json(
    aem: &AEM,
    key_case: CCSDSJsonKeyCase,
) -> Result<String, BraheError> {

    let mut root = Map::new();

    // Header
    let mut header = Map::new();
    header.insert(
        key("CCSDS_AEM_VERS", key_case),
        json!(aem.header.format_version),
    );
    if let Some(ref class) = aem.header.classification {
        header.insert(key("CLASSIFICATION", key_case), json!(class));
    }
    header.insert(
        key("CREATION_DATE", key_case),
        json!(format_ccsds_datetime_in(
            &aem.header.creation_date,
            &CCSDSTimeSystem::UTC
        )),
    );
    header.insert(key("ORIGINATOR", key_case), json!(&aem.header.originator));
    if let Some(ref msg_id) = aem.header.message_id {
        header.insert(key("MESSAGE_ID", key_case), json!(msg_id));
    }
    if !aem.header.comments.is_empty() {
        header.insert("comments".into(), json!(aem.header.comments));
    }
    root.insert("header".into(), Value::Object(header));

    // Segments
    let mut segments = Vec::new();
    for seg in &aem.segments {
        let write_ts = seg.metadata.time_system.to_time_system();
        let epoch_for_write = |e: &Epoch| -> Epoch {
            match write_ts {
                Some(ts) => e.to_time_system(ts),
                None => *e,
            }
        };

        let mut seg_obj = Map::new();

        // Metadata
        let mut meta = Map::new();
        meta.insert(
            key("OBJECT_NAME", key_case),
            json!(&seg.metadata.object_name),
        );
        meta.insert(key("OBJECT_ID", key_case), json!(&seg.metadata.object_id));
        if let Some(ref center) = seg.metadata.center_name {
            meta.insert(key("CENTER_NAME", key_case), json!(center));
        }
        meta.insert(
            key("REF_FRAME_A", key_case),
            json!(format!("{}", seg.metadata.ref_frame_a)),
        );
        meta.insert(
            key("REF_FRAME_B", key_case),
            json!(format!("{}", seg.metadata.ref_frame_b)),
        );
        meta.insert(
            key("TIME_SYSTEM", key_case),
            json!(format!("{}", seg.metadata.time_system)),
        );
        meta.insert(
            key("START_TIME", key_case),
            json!(format_ccsds_datetime(&epoch_for_write(
                &seg.metadata.start_time
            ))),
        );
        if let Some(ref t) = seg.metadata.useable_start_time {
            meta.insert(
                key("USEABLE_START_TIME", key_case),
                json!(format_ccsds_datetime(&epoch_for_write(t))),
            );
        }
        if let Some(ref t) = seg.metadata.useable_stop_time {
            meta.insert(
                key("USEABLE_STOP_TIME", key_case),
                json!(format_ccsds_datetime(&epoch_for_write(t))),
            );
        }
        meta.insert(
            key("STOP_TIME", key_case),
            json!(format_ccsds_datetime(&epoch_for_write(
                &seg.metadata.stop_time
            ))),
        );
        meta.insert(
            key("ATTITUDE_TYPE", key_case),
            json!(format!("{}", seg.metadata.attitude_type)),
        );
        if let Some(seq) = seg.metadata.euler_rot_seq {
            meta.insert(
                key("EULER_ROT_SEQ", key_case),
                json!(format_euler_rot_seq(seq)),
            );
        }
        if let Some(ref frame) = seg.metadata.angvel_frame {
            meta.insert(key("ANGVEL_FRAME", key_case), json!(format!("{}", frame)));
        }
        if let Some(method) = seg.metadata.interpolation_method {
            meta.insert(
                key("INTERPOLATION_METHOD", key_case),
                json!(format!("{}", method)),
            );
        }
        if let Some(degree) = seg.metadata.interpolation_degree {
            meta.insert(key("INTERPOLATION_DEGREE", key_case), json!(degree));
        }
        if !seg.metadata.comments.is_empty() {
            meta.insert("comments".into(), json!(seg.metadata.comments));
        }
        seg_obj.insert("metadata".into(), Value::Object(meta));

        if !seg.comments.is_empty() {
            seg_obj.insert("comments".into(), json!(seg.comments));
        }

        // States
        let mut states = Vec::new();
        for state in &seg.states {
            let mut obj = Map::new();
            obj.insert(
                key("EPOCH", key_case),
                json!(format_ccsds_datetime(&epoch_for_write(&state.epoch))),
            );
            match &state.data {
                AEMAttitudeData::Quaternion { quaternion } => {
                    let v = quaternion.to_vector(false);
                    obj.insert(key("Q1", key_case), json!(v[0]));
                    obj.insert(key("Q2", key_case), json!(v[1]));
                    obj.insert(key("Q3", key_case), json!(v[2]));
                    obj.insert(key("QC", key_case), json!(v[3]));
                }
                AEMAttitudeData::QuaternionDerivative {
                    quaternion,
                    derivative,
                } => {
                    let v = quaternion.to_vector(false);
                    obj.insert(key("Q1", key_case), json!(v[0]));
                    obj.insert(key("Q2", key_case), json!(v[1]));
                    obj.insert(key("Q3", key_case), json!(v[2]));
                    obj.insert(key("QC", key_case), json!(v[3]));
                    // derivative is stored scalar-first; wire order is scalar-last.
                    obj.insert(key("Q1_DOT", key_case), json!(derivative[1]));
                    obj.insert(key("Q2_DOT", key_case), json!(derivative[2]));
                    obj.insert(key("Q3_DOT", key_case), json!(derivative[3]));
                    obj.insert(key("QC_DOT", key_case), json!(derivative[0]));
                }
                AEMAttitudeData::QuaternionAngVel {
                    quaternion,
                    angular_velocity,
                } => {
                    let v = quaternion.to_vector(false);
                    obj.insert(key("Q1", key_case), json!(v[0]));
                    obj.insert(key("Q2", key_case), json!(v[1]));
                    obj.insert(key("Q3", key_case), json!(v[2]));
                    obj.insert(key("QC", key_case), json!(v[3]));
                    obj.insert(
                        key("ANGVEL_X", key_case),
                        json!(angular_velocity[0] * RAD2DEG),
                    );
                    obj.insert(
                        key("ANGVEL_Y", key_case),
                        json!(angular_velocity[1] * RAD2DEG),
                    );
                    obj.insert(
                        key("ANGVEL_Z", key_case),
                        json!(angular_velocity[2] * RAD2DEG),
                    );
                }
                AEMAttitudeData::EulerAngle { angles } => {
                    obj.insert(key("ANGLE_1", key_case), json!(angles.phi * RAD2DEG));
                    obj.insert(key("ANGLE_2", key_case), json!(angles.theta * RAD2DEG));
                    obj.insert(key("ANGLE_3", key_case), json!(angles.psi * RAD2DEG));
                }
                AEMAttitudeData::EulerAngleDerivative { angles, rates } => {
                    obj.insert(key("ANGLE_1", key_case), json!(angles.phi * RAD2DEG));
                    obj.insert(key("ANGLE_2", key_case), json!(angles.theta * RAD2DEG));
                    obj.insert(key("ANGLE_3", key_case), json!(angles.psi * RAD2DEG));
                    obj.insert(key("ANGLE_1_DOT", key_case), json!(rates[0] * RAD2DEG));
                    obj.insert(key("ANGLE_2_DOT", key_case), json!(rates[1] * RAD2DEG));
                    obj.insert(key("ANGLE_3_DOT", key_case), json!(rates[2] * RAD2DEG));
                }
                AEMAttitudeData::EulerAngleAngVel {
                    angles,
                    angular_velocity,
                } => {
                    obj.insert(key("ANGLE_1", key_case), json!(angles.phi * RAD2DEG));
                    obj.insert(key("ANGLE_2", key_case), json!(angles.theta * RAD2DEG));
                    obj.insert(key("ANGLE_3", key_case), json!(angles.psi * RAD2DEG));
                    obj.insert(
                        key("ANGVEL_X", key_case),
                        json!(angular_velocity[0] * RAD2DEG),
                    );
                    obj.insert(
                        key("ANGVEL_Y", key_case),
                        json!(angular_velocity[1] * RAD2DEG),
                    );
                    obj.insert(
                        key("ANGVEL_Z", key_case),
                        json!(angular_velocity[2] * RAD2DEG),
                    );
                }
                AEMAttitudeData::Spin {
                    spin_alpha,
                    spin_delta,
                    spin_angle,
                    spin_angle_vel,
                } => {
                    obj.insert(key("SPIN_ALPHA", key_case), json!(spin_alpha * RAD2DEG));
                    obj.insert(key("SPIN_DELTA", key_case), json!(spin_delta * RAD2DEG));
                    obj.insert(key("SPIN_ANGLE", key_case), json!(spin_angle * RAD2DEG));
                    obj.insert(
                        key("SPIN_ANGLE_VEL", key_case),
                        json!(spin_angle_vel * RAD2DEG),
                    );
                }
                AEMAttitudeData::SpinNutation {
                    spin_alpha,
                    spin_delta,
                    spin_angle,
                    spin_angle_vel,
                    nutation,
                    nutation_period,
                    nutation_phase,
                } => {
                    obj.insert(key("SPIN_ALPHA", key_case), json!(spin_alpha * RAD2DEG));
                    obj.insert(key("SPIN_DELTA", key_case), json!(spin_delta * RAD2DEG));
                    obj.insert(key("SPIN_ANGLE", key_case), json!(spin_angle * RAD2DEG));
                    obj.insert(
                        key("SPIN_ANGLE_VEL", key_case),
                        json!(spin_angle_vel * RAD2DEG),
                    );
                    obj.insert(key("NUTATION", key_case), json!(nutation * RAD2DEG));
                    obj.insert(key("NUTATION_PER", key_case), json!(nutation_period));
                    obj.insert(
                        key("NUTATION_PHASE", key_case),
                        json!(nutation_phase * RAD2DEG),
                    );
                }
                AEMAttitudeData::SpinNutationMom {
                    spin_alpha,
                    spin_delta,
                    spin_angle,
                    spin_angle_vel,
                    momentum_alpha,
                    momentum_delta,
                    nutation_vel,
                } => {
                    obj.insert(key("SPIN_ALPHA", key_case), json!(spin_alpha * RAD2DEG));
                    obj.insert(key("SPIN_DELTA", key_case), json!(spin_delta * RAD2DEG));
                    obj.insert(key("SPIN_ANGLE", key_case), json!(spin_angle * RAD2DEG));
                    obj.insert(
                        key("SPIN_ANGLE_VEL", key_case),
                        json!(spin_angle_vel * RAD2DEG),
                    );
                    obj.insert(
                        key("MOMENTUM_ALPHA", key_case),
                        json!(momentum_alpha * RAD2DEG),
                    );
                    obj.insert(
                        key("MOMENTUM_DELTA", key_case),
                        json!(momentum_delta * RAD2DEG),
                    );
                    obj.insert(key("NUTATION_VEL", key_case), json!(nutation_vel * RAD2DEG));
                }
            }
            states.push(Value::Object(obj));
        }
        seg_obj.insert("states".into(), Value::Array(states));

        segments.push(Value::Object(seg_obj));
    }
    root.insert("segments".into(), Value::Array(segments));

    serde_json::to_string_pretty(&Value::Object(root))
        .map_err(|e| BraheError::Error(format!("AEM JSON serialization error: {}", e)))
}

