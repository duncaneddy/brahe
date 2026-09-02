/*!
 * JSON reader and writer for the Attitude Parameter Message (APM).
 *
 * Reference: CCSDS 504.0-B-2 (Attitude Data Messages), section 3
 */

use serde_json::{Map, Value, json};

use crate::ccsds::common::{CCSDSJsonKeyCase, CCSDSTimeSystem, format_ccsds_datetime_in};
use crate::ccsds::error::{ccsds_missing_field, ccsds_parse_error};
use crate::ccsds::json::common::{emit_json_comments, emit_kvn, flatten_object_skip, key};
use crate::utils::errors::BraheError;

// =============================================================================
// APM JSON
// =============================================================================

/// Parse an APM message from JSON format.
///
/// Flattens the JSON structure into KVN-style lines, including
/// `X_START`/`X_STOP` delimiters for each logical block (quaternion, Euler
/// angle, angular velocity, spin, inertia, maneuver), then delegates to the
/// KVN parser. Values are expected in wire units (degrees for angles/rates),
/// matching the KVN representation, since parsing performs no unit
/// conversion beyond what the KVN parser already does.
pub fn parse_apm_json(content: &str) -> Result<crate::ccsds::apm::APM, BraheError> {
    let v: Value = serde_json::from_str(content)
        .map_err(|e| ccsds_parse_error("APM", &format!("JSON parse error: {}", e)))?;

    let mut kvn_lines: Vec<String> = Vec::new();

    // Header comments are only attributed to the header by the KVN parser
    // while it has not yet seen CLASSIFICATION/CREATION_DATE/ORIGINATOR/
    // MESSAGE_ID, so they must be emitted ahead of those keywords (their
    // position relative to CCSDS_APM_VERS itself does not matter).
    if let Some(obj) = v.get("header").or_else(|| v.get("HEADER")) {
        emit_json_comments(&mut kvn_lines, obj);
        flatten_object_skip(&mut kvn_lines, obj, &["COMMENTS"]);
    }

    // serde_json's Map sorts keys alphabetically, which would place
    // CENTER_NAME ahead of OBJECT_NAME; the KVN parser requires OBJECT_NAME
    // first to transition out of the header state. Metadata comments must be
    // emitted before OBJECT_NAME too: once the parser has transitioned into
    // the metadata state, a comment is instead attributed to the data
    // section (matching `parse_apm`'s handling of a KVN file where the
    // metadata's COMMENT lines always precede OBJECT_NAME).
    if let Some(obj) = v.get("metadata").or_else(|| v.get("METADATA")) {
        emit_json_comments(&mut kvn_lines, obj);
        if let Value::Object(map) = obj {
            if let Some(val) = map.get("OBJECT_NAME").or_else(|| map.get("object_name")) {
                emit_kvn(&mut kvn_lines, "OBJECT_NAME", val);
            }
            for (k, val) in map {
                let ukey = k.to_uppercase();
                if ukey == "OBJECT_NAME" || ukey == "COMMENTS" {
                    continue;
                }
                emit_kvn(&mut kvn_lines, &ukey, val);
            }
        }
    }

    // Top-level data-section comments (before the first logical block) must
    // be emitted while still in the KVN parser's metadata state, i.e. before
    // EPOCH, so they are routed to APM::comments rather than treated as
    // header comments.
    if let Some(Value::Array(comments)) = v.get("comments").or_else(|| v.get("COMMENTS")) {
        for c in comments {
            if let Some(s) = c.as_str() {
                kvn_lines.push(format!("COMMENT {}", s));
            }
        }
    }

    if let Some(epoch) = v
        .get("epoch")
        .or_else(|| v.get("EPOCH"))
        .and_then(|e| e.as_str())
    {
        kvn_lines.push(format!("EPOCH = {}", epoch));
    }

    // Logical blocks: each array item becomes an X_START/.../X_STOP group.
    // Field order within a block does not matter to the KVN parser (each
    // key independently sets local state; only the STOP keyword triggers
    // assembly), so comments can be emitted right after the START keyword
    // and the remaining fields flattened in any order.
    let block_sections: [(&str, &str, &str); 6] = [
        ("quaternion_states", "QUAT_START", "QUAT_STOP"),
        ("euler_states", "EULER_START", "EULER_STOP"),
        ("angular_velocities", "ANGVEL_START", "ANGVEL_STOP"),
        ("spins", "SPIN_START", "SPIN_STOP"),
        ("inertias", "INERTIA_START", "INERTIA_STOP"),
        ("maneuvers", "MAN_START", "MAN_STOP"),
    ];

    for (key_name, start, stop) in block_sections {
        if let Some(Value::Array(items)) =
            v.get(key_name).or_else(|| v.get(key_name.to_uppercase()))
        {
            for item in items {
                kvn_lines.push(start.to_string());
                emit_json_comments(&mut kvn_lines, item);
                flatten_object_skip(&mut kvn_lines, item, &["COMMENTS"]);
                kvn_lines.push(stop.to_string());
            }
        }
    }

    let kvn_content = kvn_lines.join("\n");
    crate::ccsds::kvn::parse_apm(&kvn_content)
}

/// Write an APM message to JSON format.
///
/// Values are written in wire units (degrees for angles/rates, scalar-last
/// quaternion component keys `Q1..QC`/`Q1_DOT..QC_DOT`), matching the KVN
/// representation so that [`parse_apm_json`]'s flatten-to-KVN-lines path
/// reproduces them unchanged.
pub fn write_apm_json(
    apm: &crate::ccsds::apm::APM,
    key_case: CCSDSJsonKeyCase,
) -> Result<String, BraheError> {
    use crate::ccsds::apm::APMNutation;
    use crate::ccsds::common::format_euler_rot_seq;
    use crate::constants::RAD2DEG;

    if !apm.has_blocks() {
        return Err(ccsds_missing_field("APM", "at least one logical block"));
    }

    let mut root = Map::new();

    // Header
    let mut header = Map::new();
    header.insert(
        key("CCSDS_APM_VERS", key_case),
        json!(apm.header.format_version),
    );
    if let Some(ref class) = apm.header.classification {
        header.insert(key("CLASSIFICATION", key_case), json!(class));
    }
    header.insert(
        key("CREATION_DATE", key_case),
        json!(format_ccsds_datetime_in(
            &apm.header.creation_date,
            &CCSDSTimeSystem::UTC
        )),
    );
    header.insert(key("ORIGINATOR", key_case), json!(&apm.header.originator));
    if let Some(ref msg_id) = apm.header.message_id {
        header.insert(key("MESSAGE_ID", key_case), json!(msg_id));
    }
    if !apm.header.comments.is_empty() {
        header.insert("comments".into(), json!(apm.header.comments));
    }
    root.insert("header".into(), Value::Object(header));

    // Metadata
    let mut meta = Map::new();
    meta.insert(
        key("OBJECT_NAME", key_case),
        json!(&apm.metadata.object_name),
    );
    meta.insert(key("OBJECT_ID", key_case), json!(&apm.metadata.object_id));
    if let Some(ref center) = apm.metadata.center_name {
        meta.insert(key("CENTER_NAME", key_case), json!(center));
    }
    meta.insert(
        key("TIME_SYSTEM", key_case),
        json!(format!("{}", apm.metadata.time_system)),
    );
    if !apm.metadata.comments.is_empty() {
        meta.insert("comments".into(), json!(apm.metadata.comments));
    }
    root.insert("metadata".into(), Value::Object(meta));

    root.insert(
        "epoch".into(),
        json!(format_ccsds_datetime_in(
            &apm.epoch,
            &apm.metadata.time_system
        )),
    );

    if !apm.comments.is_empty() {
        root.insert("comments".into(), json!(apm.comments));
    }

    // Quaternion states
    if !apm.quaternion_states.is_empty() {
        let mut arr = Vec::new();
        for q in &apm.quaternion_states {
            let mut obj = Map::new();
            obj.insert(
                key("REF_FRAME_A", key_case),
                json!(format!("{}", q.ref_frame_a)),
            );
            obj.insert(
                key("REF_FRAME_B", key_case),
                json!(format!("{}", q.ref_frame_b)),
            );
            let qv = q.quaternion.to_vector(false);
            obj.insert(key("Q1", key_case), json!(qv[0]));
            obj.insert(key("Q2", key_case), json!(qv[1]));
            obj.insert(key("Q3", key_case), json!(qv[2]));
            obj.insert(key("QC", key_case), json!(qv[3]));
            if let Some(d) = q.quaternion_derivative {
                // d is stored scalar-first; wire order is scalar-last.
                obj.insert(key("Q1_DOT", key_case), json!(d[1]));
                obj.insert(key("Q2_DOT", key_case), json!(d[2]));
                obj.insert(key("Q3_DOT", key_case), json!(d[3]));
                obj.insert(key("QC_DOT", key_case), json!(d[0]));
            }
            if !q.comments.is_empty() {
                obj.insert("comments".into(), json!(q.comments));
            }
            arr.push(Value::Object(obj));
        }
        root.insert("quaternion_states".into(), Value::Array(arr));
    }

    // Euler angle states
    if !apm.euler_states.is_empty() {
        let mut arr = Vec::new();
        for e in &apm.euler_states {
            let mut obj = Map::new();
            obj.insert(
                key("REF_FRAME_A", key_case),
                json!(format!("{}", e.ref_frame_a)),
            );
            obj.insert(
                key("REF_FRAME_B", key_case),
                json!(format!("{}", e.ref_frame_b)),
            );
            obj.insert(
                key("EULER_ROT_SEQ", key_case),
                json!(format_euler_rot_seq(e.angles.order)),
            );
            obj.insert(key("ANGLE_1", key_case), json!(e.angles.phi * RAD2DEG));
            obj.insert(key("ANGLE_2", key_case), json!(e.angles.theta * RAD2DEG));
            obj.insert(key("ANGLE_3", key_case), json!(e.angles.psi * RAD2DEG));
            if let Some(r) = e.rates {
                obj.insert(key("ANGLE_1_DOT", key_case), json!(r[0] * RAD2DEG));
                obj.insert(key("ANGLE_2_DOT", key_case), json!(r[1] * RAD2DEG));
                obj.insert(key("ANGLE_3_DOT", key_case), json!(r[2] * RAD2DEG));
            }
            if !e.comments.is_empty() {
                obj.insert("comments".into(), json!(e.comments));
            }
            arr.push(Value::Object(obj));
        }
        root.insert("euler_states".into(), Value::Array(arr));
    }

    // Angular velocities
    if !apm.angular_velocities.is_empty() {
        let mut arr = Vec::new();
        for av in &apm.angular_velocities {
            let mut obj = Map::new();
            obj.insert(
                key("REF_FRAME_A", key_case),
                json!(format!("{}", av.ref_frame_a)),
            );
            obj.insert(
                key("REF_FRAME_B", key_case),
                json!(format!("{}", av.ref_frame_b)),
            );
            obj.insert(
                key("ANGVEL_FRAME", key_case),
                json!(format!("{}", av.angvel_frame)),
            );
            obj.insert(
                key("ANGVEL_X", key_case),
                json!(av.angular_velocity[0] * RAD2DEG),
            );
            obj.insert(
                key("ANGVEL_Y", key_case),
                json!(av.angular_velocity[1] * RAD2DEG),
            );
            obj.insert(
                key("ANGVEL_Z", key_case),
                json!(av.angular_velocity[2] * RAD2DEG),
            );
            if !av.comments.is_empty() {
                obj.insert("comments".into(), json!(av.comments));
            }
            arr.push(Value::Object(obj));
        }
        root.insert("angular_velocities".into(), Value::Array(arr));
    }

    // Spins
    if !apm.spins.is_empty() {
        let mut arr = Vec::new();
        for s in &apm.spins {
            let mut obj = Map::new();
            obj.insert(
                key("REF_FRAME_A", key_case),
                json!(format!("{}", s.ref_frame_a)),
            );
            obj.insert(
                key("REF_FRAME_B", key_case),
                json!(format!("{}", s.ref_frame_b)),
            );
            obj.insert(key("SPIN_ALPHA", key_case), json!(s.spin_alpha * RAD2DEG));
            obj.insert(key("SPIN_DELTA", key_case), json!(s.spin_delta * RAD2DEG));
            obj.insert(key("SPIN_ANGLE", key_case), json!(s.spin_angle * RAD2DEG));
            obj.insert(
                key("SPIN_ANGLE_VEL", key_case),
                json!(s.spin_angle_vel * RAD2DEG),
            );
            match &s.nutation {
                APMNutation::None => {}
                APMNutation::Angle {
                    nutation,
                    nutation_period,
                    nutation_phase,
                } => {
                    obj.insert(key("NUTATION", key_case), json!(nutation * RAD2DEG));
                    obj.insert(key("NUTATION_PER", key_case), json!(nutation_period));
                    obj.insert(
                        key("NUTATION_PHASE", key_case),
                        json!(nutation_phase * RAD2DEG),
                    );
                }
                APMNutation::Momentum {
                    momentum_alpha,
                    momentum_delta,
                    nutation_vel,
                } => {
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
            if !s.comments.is_empty() {
                obj.insert("comments".into(), json!(s.comments));
            }
            arr.push(Value::Object(obj));
        }
        root.insert("spins".into(), Value::Array(arr));
    }

    // Inertias
    if !apm.inertias.is_empty() {
        let mut arr = Vec::new();
        for i in &apm.inertias {
            let mut obj = Map::new();
            obj.insert(
                key("INERTIA_REF_FRAME", key_case),
                json!(format!("{}", i.inertia_ref_frame)),
            );
            obj.insert(key("IXX", key_case), json!(i.ixx));
            obj.insert(key("IYY", key_case), json!(i.iyy));
            obj.insert(key("IZZ", key_case), json!(i.izz));
            obj.insert(key("IXY", key_case), json!(i.ixy));
            obj.insert(key("IXZ", key_case), json!(i.ixz));
            obj.insert(key("IYZ", key_case), json!(i.iyz));
            if !i.comments.is_empty() {
                obj.insert("comments".into(), json!(i.comments));
            }
            arr.push(Value::Object(obj));
        }
        root.insert("inertias".into(), Value::Array(arr));
    }

    // Maneuvers
    if !apm.maneuvers.is_empty() {
        let mut arr = Vec::new();
        for m in &apm.maneuvers {
            let mut obj = Map::new();
            obj.insert(
                key("MAN_EPOCH_START", key_case),
                json!(format_ccsds_datetime_in(
                    &m.epoch_start,
                    &apm.metadata.time_system
                )),
            );
            obj.insert(key("MAN_DURATION", key_case), json!(m.duration));
            obj.insert(
                key("MAN_REF_FRAME", key_case),
                json!(format!("{}", m.ref_frame)),
            );
            obj.insert(key("MAN_TOR_X", key_case), json!(m.torque[0]));
            obj.insert(key("MAN_TOR_Y", key_case), json!(m.torque[1]));
            obj.insert(key("MAN_TOR_Z", key_case), json!(m.torque[2]));
            if let Some(dm) = m.delta_mass {
                obj.insert(key("MAN_DELTA_MASS", key_case), json!(dm));
            }
            if !m.comments.is_empty() {
                obj.insert("comments".into(), json!(m.comments));
            }
            arr.push(Value::Object(obj));
        }
        root.insert("maneuvers".into(), Value::Array(arr));
    }

    serde_json::to_string_pretty(&Value::Object(root))
        .map_err(|e| BraheError::Error(format!("APM JSON serialization error: {}", e)))
}
