/*!
 * KVN reader and writer for the Attitude Parameter Message (APM).
 *
 * Reference: CCSDS 504.0-B-2 (Attitude Data Messages), section 3
 */

use nalgebra::{Vector3, Vector4};

use crate::attitude::attitude_types::{EulerAngle, EulerAngleOrder, Quaternion};
use crate::ccsds::apm::{
    APM, APMAngularVelocity, APMEulerState, APMHeader, APMInertia, APMManeuver, APMMetadata,
    APMNutation, APMQuaternionState, APMSpin,
};
use crate::ccsds::common::{
    CCSDSTimeSystem, format_ccsds_datetime_in, format_euler_rot_seq, parse_ccsds_datetime,
    parse_euler_rot_seq, strip_units,
};
use crate::ccsds::error::{ccsds_missing_field, ccsds_parse_error};
use crate::ccsds::frames::ADMReferenceFrame;
use crate::ccsds::kvn::common::{KVNToken, tokenize_line};
use crate::constants::{AngleFormat, DEG2RAD, RAD2DEG};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

/// Parse an APM numeric field, producing a field-specific error message on
/// failure.
fn parse_apm_f64(val: &str, field: &str) -> Result<f64, BraheError> {
    val.parse::<f64>()
        .map_err(|_| ccsds_parse_error("APM", &format!("invalid {} value '{}'", field, val)))
}

/// Parser state for APM KVN parsing.
#[derive(Debug, PartialEq)]
enum APMState {
    Header,
    Metadata,
    /// Top-level data section: between/before logical blocks.
    Data,
    Quat,
    Euler,
    AngVel,
    Spin,
    Inertia,
    Man,
}

/// Parse an APM message from KVN format.
///
/// APM has no explicit metadata delimiters; the transition from header to
/// metadata is triggered by the `OBJECT_NAME` keyword, and from metadata to
/// the data section by `EPOCH` (504.0-B-2 §3.2.1). Six repeatable logical
/// blocks (quaternion, Euler angle, angular velocity, spin, inertia,
/// maneuver) are delimited by `*_START`/`*_STOP` markers; a flat accumulator
/// is flushed and validated at each `*_STOP`.
pub fn parse_apm(content: &str) -> Result<APM, BraheError> {
    let mut state = APMState::Header;

    // Header
    let mut format_version: Option<f64> = None;
    let mut classification: Option<String> = None;
    let mut creation_date: Option<Epoch> = None;
    let mut originator: Option<String> = None;
    let mut message_id: Option<String> = None;
    let mut header_comments: Vec<String> = Vec::new();
    // CCSDS_APM_VERS is the only keyword that may precede a header comment
    // (table 3-1); once any other header field is seen, subsequent comments
    // (still before OBJECT_NAME) belong to the metadata section instead.
    let mut header_fields_started = false;

    // Metadata
    let mut object_name: Option<String> = None;
    let mut object_id: Option<String> = None;
    let mut center_name: Option<String> = None;
    let mut time_system: Option<CCSDSTimeSystem> = None;
    let mut metadata_comments: Vec<String> = Vec::new();

    // Data (top level)
    let mut epoch: Option<Epoch> = None;
    let mut data_comments: Vec<String> = Vec::new();

    // Shared block accumulators (reused across quaternion/Euler/angular
    // velocity/spin, which are never open concurrently).
    let mut blk_ref_frame_a: Option<ADMReferenceFrame> = None;
    let mut blk_ref_frame_b: Option<ADMReferenceFrame> = None;
    let mut blk_comments: Vec<String> = Vec::new();

    // Quaternion block
    let mut q1: Option<f64> = None;
    let mut q2: Option<f64> = None;
    let mut q3: Option<f64> = None;
    let mut qc: Option<f64> = None;
    let mut q1_dot: Option<f64> = None;
    let mut q2_dot: Option<f64> = None;
    let mut q3_dot: Option<f64> = None;
    let mut qc_dot: Option<f64> = None;

    // Euler angle block
    let mut euler_rot_seq: Option<EulerAngleOrder> = None;
    let mut angle1: Option<f64> = None;
    let mut angle2: Option<f64> = None;
    let mut angle3: Option<f64> = None;
    let mut angle1_dot: Option<f64> = None;
    let mut angle2_dot: Option<f64> = None;
    let mut angle3_dot: Option<f64> = None;

    // Angular velocity block
    let mut angvel_frame: Option<ADMReferenceFrame> = None;
    let mut angvel_x: Option<f64> = None;
    let mut angvel_y: Option<f64> = None;
    let mut angvel_z: Option<f64> = None;

    // Spin block
    let mut spin_alpha: Option<f64> = None;
    let mut spin_delta: Option<f64> = None;
    let mut spin_angle: Option<f64> = None;
    let mut spin_angle_vel: Option<f64> = None;
    let mut nutation: Option<f64> = None;
    let mut nutation_per: Option<f64> = None;
    let mut nutation_phase: Option<f64> = None;
    let mut momentum_alpha: Option<f64> = None;
    let mut momentum_delta: Option<f64> = None;
    let mut nutation_vel: Option<f64> = None;

    // Inertia block
    let mut inertia_ref_frame: Option<ADMReferenceFrame> = None;
    let mut ixx: Option<f64> = None;
    let mut iyy: Option<f64> = None;
    let mut izz: Option<f64> = None;
    let mut ixy: Option<f64> = None;
    let mut ixz: Option<f64> = None;
    let mut iyz: Option<f64> = None;

    // Maneuver block
    let mut man_epoch_start: Option<Epoch> = None;
    let mut man_duration: Option<f64> = None;
    let mut man_ref_frame: Option<ADMReferenceFrame> = None;
    let mut man_tor_x: Option<f64> = None;
    let mut man_tor_y: Option<f64> = None;
    let mut man_tor_z: Option<f64> = None;
    let mut man_delta_mass: Option<f64> = None;

    // Completed blocks
    let mut quaternion_states: Vec<APMQuaternionState> = Vec::new();
    let mut euler_states: Vec<APMEulerState> = Vec::new();
    let mut angular_velocities: Vec<APMAngularVelocity> = Vec::new();
    let mut spins: Vec<APMSpin> = Vec::new();
    let mut inertias: Vec<APMInertia> = Vec::new();
    let mut maneuvers: Vec<APMManeuver> = Vec::new();

    let active_ts = |ts: &Option<CCSDSTimeSystem>| ts.clone().unwrap_or(CCSDSTimeSystem::UTC);

    for line in content.lines() {
        let token = tokenize_line(line);
        match (&state, token) {
            // ===== HEADER =====
            (APMState::Header, KVNToken::KeyValue { key, value }) => {
                let val = strip_units(&value);
                match key.as_str() {
                    "CCSDS_APM_VERS" => {
                        let v: f64 = val
                            .parse()
                            .map_err(|_| ccsds_parse_error("APM", "invalid version"))?;
                        if (v - 1.0).abs() < 1e-9 {
                            return Err(ccsds_parse_error(
                                "APM",
                                "version 1.0 (504.0-B-1) files are not supported; only version 2.0",
                            ));
                        }
                        format_version = Some(v);
                    }
                    "CLASSIFICATION" => {
                        classification = Some(val.to_string());
                        header_fields_started = true;
                    }
                    "CREATION_DATE" => {
                        creation_date = Some(parse_ccsds_datetime(val, &CCSDSTimeSystem::UTC)?);
                        header_fields_started = true;
                    }
                    "ORIGINATOR" => {
                        originator = Some(val.to_string());
                        header_fields_started = true;
                    }
                    "MESSAGE_ID" => {
                        message_id = Some(val.to_string());
                        header_fields_started = true;
                    }
                    "OBJECT_NAME" => {
                        object_name = Some(val.to_string());
                        state = APMState::Metadata;
                    }
                    _ => {
                        return Err(ccsds_parse_error(
                            "APM",
                            &format!("unexpected header keyword '{}'", key),
                        ));
                    }
                }
            }
            (APMState::Header, KVNToken::Comment(text)) => {
                if header_fields_started {
                    metadata_comments.push(text);
                } else {
                    header_comments.push(text);
                }
            }
            (APMState::Header, KVNToken::Empty) => {}

            // ===== METADATA =====
            (APMState::Metadata, KVNToken::KeyValue { key, value }) => {
                let val = strip_units(&value);
                match key.as_str() {
                    "OBJECT_ID" => object_id = Some(val.to_string()),
                    "CENTER_NAME" => center_name = Some(val.to_string()),
                    "TIME_SYSTEM" => time_system = Some(CCSDSTimeSystem::parse(val)?),
                    "EPOCH" => {
                        epoch = Some(parse_ccsds_datetime(val, &active_ts(&time_system))?);
                        state = APMState::Data;
                    }
                    _ => {
                        return Err(ccsds_parse_error(
                            "APM",
                            &format!("unexpected metadata keyword '{}'", key),
                        ));
                    }
                }
            }
            // Per 504.0-B-2 §6.10.2 a metadata comment is only valid before
            // OBJECT_NAME (handled above, in the Header state); a comment
            // seen after OBJECT_NAME belongs to the data section's top-level
            // comment block (before the first logical block).
            (APMState::Metadata, KVNToken::Comment(text)) => {
                data_comments.push(text);
            }
            (APMState::Metadata, KVNToken::Empty) => {}

            // ===== DATA (top level, between/before blocks) =====
            (APMState::Data, KVNToken::KeyValue { key, value: _ }) => match key.as_str() {
                "QUAT_START" => {
                    blk_ref_frame_a = None;
                    blk_ref_frame_b = None;
                    blk_comments.clear();
                    q1 = None;
                    q2 = None;
                    q3 = None;
                    qc = None;
                    q1_dot = None;
                    q2_dot = None;
                    q3_dot = None;
                    qc_dot = None;
                    state = APMState::Quat;
                }
                "EULER_START" => {
                    blk_ref_frame_a = None;
                    blk_ref_frame_b = None;
                    blk_comments.clear();
                    euler_rot_seq = None;
                    angle1 = None;
                    angle2 = None;
                    angle3 = None;
                    angle1_dot = None;
                    angle2_dot = None;
                    angle3_dot = None;
                    state = APMState::Euler;
                }
                "ANGVEL_START" => {
                    blk_ref_frame_a = None;
                    blk_ref_frame_b = None;
                    blk_comments.clear();
                    angvel_frame = None;
                    angvel_x = None;
                    angvel_y = None;
                    angvel_z = None;
                    state = APMState::AngVel;
                }
                "SPIN_START" => {
                    blk_ref_frame_a = None;
                    blk_ref_frame_b = None;
                    blk_comments.clear();
                    spin_alpha = None;
                    spin_delta = None;
                    spin_angle = None;
                    spin_angle_vel = None;
                    nutation = None;
                    nutation_per = None;
                    nutation_phase = None;
                    momentum_alpha = None;
                    momentum_delta = None;
                    nutation_vel = None;
                    state = APMState::Spin;
                }
                "INERTIA_START" => {
                    inertia_ref_frame = None;
                    blk_comments.clear();
                    ixx = None;
                    iyy = None;
                    izz = None;
                    ixy = None;
                    ixz = None;
                    iyz = None;
                    state = APMState::Inertia;
                }
                "MAN_START" => {
                    man_epoch_start = None;
                    man_duration = None;
                    man_ref_frame = None;
                    man_tor_x = None;
                    man_tor_y = None;
                    man_tor_z = None;
                    man_delta_mass = None;
                    blk_comments.clear();
                    state = APMState::Man;
                }
                k if k.starts_with("USER_DEFINED_") => {
                    return Err(ccsds_parse_error(
                        "APM",
                        &format!(
                            "unexpected keyword '{}' in data section: USER_DEFINED_* is not part of APM per 504.0-B-2 §3.2.4.2",
                            k
                        ),
                    ));
                }
                _ => {
                    return Err(ccsds_parse_error(
                        "APM",
                        &format!("unexpected keyword '{}' in data section", key),
                    ));
                }
            },
            (APMState::Data, KVNToken::Comment(text)) => {
                data_comments.push(text);
            }
            (APMState::Data, KVNToken::Empty) => {}

            // ===== QUATERNION BLOCK =====
            (APMState::Quat, KVNToken::KeyValue { key, value }) => {
                let val = strip_units(&value);
                match key.as_str() {
                    "REF_FRAME_A" => blk_ref_frame_a = Some(ADMReferenceFrame::parse(val)),
                    "REF_FRAME_B" => blk_ref_frame_b = Some(ADMReferenceFrame::parse(val)),
                    "Q1" => q1 = Some(parse_apm_f64(val, "Q1")?),
                    "Q2" => q2 = Some(parse_apm_f64(val, "Q2")?),
                    "Q3" => q3 = Some(parse_apm_f64(val, "Q3")?),
                    "QC" => qc = Some(parse_apm_f64(val, "QC")?),
                    "Q1_DOT" => q1_dot = Some(parse_apm_f64(val, "Q1_DOT")?),
                    "Q2_DOT" => q2_dot = Some(parse_apm_f64(val, "Q2_DOT")?),
                    "Q3_DOT" => q3_dot = Some(parse_apm_f64(val, "Q3_DOT")?),
                    "QC_DOT" => qc_dot = Some(parse_apm_f64(val, "QC_DOT")?),
                    "QUAT_STOP" => {
                        let ref_frame_a = blk_ref_frame_a
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "REF_FRAME_A"))?;
                        let ref_frame_b = blk_ref_frame_b
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "REF_FRAME_B"))?;
                        let q1v = q1.take().ok_or_else(|| ccsds_missing_field("APM", "Q1"))?;
                        let q2v = q2.take().ok_or_else(|| ccsds_missing_field("APM", "Q2"))?;
                        let q3v = q3.take().ok_or_else(|| ccsds_missing_field("APM", "Q3"))?;
                        let qcv = qc.take().ok_or_else(|| ccsds_missing_field("APM", "QC"))?;
                        let quaternion =
                            Quaternion::from_vector(Vector4::new(q1v, q2v, q3v, qcv), false);

                        let derivative = match (
                            q1_dot.take(),
                            q2_dot.take(),
                            q3_dot.take(),
                            qc_dot.take(),
                        ) {
                            (Some(a), Some(b), Some(c), Some(d)) => Some(Vector4::new(d, a, b, c)),
                            (None, None, None, None) => None,
                            _ => {
                                return Err(ccsds_parse_error(
                                    "APM",
                                    "incomplete quaternion derivative: Q1_DOT/Q2_DOT/Q3_DOT/QC_DOT must all be present or all absent",
                                ));
                            }
                        };

                        let mut quat =
                            APMQuaternionState::new(ref_frame_a, ref_frame_b, quaternion);
                        if let Some(d) = derivative {
                            quat = quat.with_derivative(d);
                        }
                        quat.comments = std::mem::take(&mut blk_comments);
                        quaternion_states.push(quat);
                        state = APMState::Data;
                    }
                    _ => {
                        return Err(ccsds_parse_error(
                            "APM",
                            &format!("unexpected keyword '{}' in quaternion block", key),
                        ));
                    }
                }
            }
            (APMState::Quat, KVNToken::Comment(text)) => blk_comments.push(text),
            (APMState::Quat, KVNToken::Empty) => {}

            // ===== EULER ANGLE BLOCK =====
            (APMState::Euler, KVNToken::KeyValue { key, value }) => {
                let val = strip_units(&value);
                match key.as_str() {
                    "REF_FRAME_A" => blk_ref_frame_a = Some(ADMReferenceFrame::parse(val)),
                    "REF_FRAME_B" => blk_ref_frame_b = Some(ADMReferenceFrame::parse(val)),
                    "EULER_ROT_SEQ" => euler_rot_seq = Some(parse_euler_rot_seq(val)?),
                    "ANGLE_1" => angle1 = Some(parse_apm_f64(val, "ANGLE_1")?),
                    "ANGLE_2" => angle2 = Some(parse_apm_f64(val, "ANGLE_2")?),
                    "ANGLE_3" => angle3 = Some(parse_apm_f64(val, "ANGLE_3")?),
                    "ANGLE_1_DOT" => angle1_dot = Some(parse_apm_f64(val, "ANGLE_1_DOT")?),
                    "ANGLE_2_DOT" => angle2_dot = Some(parse_apm_f64(val, "ANGLE_2_DOT")?),
                    "ANGLE_3_DOT" => angle3_dot = Some(parse_apm_f64(val, "ANGLE_3_DOT")?),
                    "EULER_STOP" => {
                        let ref_frame_a = blk_ref_frame_a
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "REF_FRAME_A"))?;
                        let ref_frame_b = blk_ref_frame_b
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "REF_FRAME_B"))?;
                        let seq = euler_rot_seq
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "EULER_ROT_SEQ"))?;
                        let a1 = angle1
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "ANGLE_1"))?;
                        let a2 = angle2
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "ANGLE_2"))?;
                        let a3 = angle3
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "ANGLE_3"))?;
                        let angles = EulerAngle::new(seq, a1, a2, a3, AngleFormat::Degrees);

                        let rates = match (angle1_dot.take(), angle2_dot.take(), angle3_dot.take())
                        {
                            (Some(a), Some(b), Some(c)) => {
                                Some(Vector3::new(a * DEG2RAD, b * DEG2RAD, c * DEG2RAD))
                            }
                            (None, None, None) => None,
                            _ => {
                                return Err(ccsds_parse_error(
                                    "APM",
                                    "incomplete Euler angle rates: ANGLE_1_DOT/ANGLE_2_DOT/ANGLE_3_DOT must all be present or all absent",
                                ));
                            }
                        };

                        let mut euler = APMEulerState::new(ref_frame_a, ref_frame_b, angles);
                        if let Some(r) = rates {
                            euler = euler.with_rates(r);
                        }
                        euler.comments = std::mem::take(&mut blk_comments);
                        euler_states.push(euler);
                        state = APMState::Data;
                    }
                    _ => {
                        return Err(ccsds_parse_error(
                            "APM",
                            &format!("unexpected keyword '{}' in Euler angle block", key),
                        ));
                    }
                }
            }
            (APMState::Euler, KVNToken::Comment(text)) => blk_comments.push(text),
            (APMState::Euler, KVNToken::Empty) => {}

            // ===== ANGULAR VELOCITY BLOCK =====
            (APMState::AngVel, KVNToken::KeyValue { key, value }) => {
                let val = strip_units(&value);
                match key.as_str() {
                    "REF_FRAME_A" => blk_ref_frame_a = Some(ADMReferenceFrame::parse(val)),
                    "REF_FRAME_B" => blk_ref_frame_b = Some(ADMReferenceFrame::parse(val)),
                    "ANGVEL_FRAME" => angvel_frame = Some(ADMReferenceFrame::parse(val)),
                    "ANGVEL_X" => angvel_x = Some(parse_apm_f64(val, "ANGVEL_X")?),
                    "ANGVEL_Y" => angvel_y = Some(parse_apm_f64(val, "ANGVEL_Y")?),
                    "ANGVEL_Z" => angvel_z = Some(parse_apm_f64(val, "ANGVEL_Z")?),
                    "ANGVEL_STOP" => {
                        let ref_frame_a = blk_ref_frame_a
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "REF_FRAME_A"))?;
                        let ref_frame_b = blk_ref_frame_b
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "REF_FRAME_B"))?;
                        let frame = angvel_frame
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "ANGVEL_FRAME"))?;
                        let x = angvel_x
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "ANGVEL_X"))?;
                        let y = angvel_y
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "ANGVEL_Y"))?;
                        let z = angvel_z
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "ANGVEL_Z"))?;
                        let vel = Vector3::new(x * DEG2RAD, y * DEG2RAD, z * DEG2RAD);
                        let mut av = APMAngularVelocity::new(ref_frame_a, ref_frame_b, frame, vel);
                        av.comments = std::mem::take(&mut blk_comments);
                        angular_velocities.push(av);
                        state = APMState::Data;
                    }
                    _ => {
                        return Err(ccsds_parse_error(
                            "APM",
                            &format!("unexpected keyword '{}' in angular velocity block", key),
                        ));
                    }
                }
            }
            (APMState::AngVel, KVNToken::Comment(text)) => blk_comments.push(text),
            (APMState::AngVel, KVNToken::Empty) => {}

            // ===== SPIN BLOCK =====
            (APMState::Spin, KVNToken::KeyValue { key, value }) => {
                let val = strip_units(&value);
                match key.as_str() {
                    "REF_FRAME_A" => blk_ref_frame_a = Some(ADMReferenceFrame::parse(val)),
                    "REF_FRAME_B" => blk_ref_frame_b = Some(ADMReferenceFrame::parse(val)),
                    "SPIN_ALPHA" => spin_alpha = Some(parse_apm_f64(val, "SPIN_ALPHA")?),
                    "SPIN_DELTA" => spin_delta = Some(parse_apm_f64(val, "SPIN_DELTA")?),
                    "SPIN_ANGLE" => spin_angle = Some(parse_apm_f64(val, "SPIN_ANGLE")?),
                    "SPIN_ANGLE_VEL" => {
                        spin_angle_vel = Some(parse_apm_f64(val, "SPIN_ANGLE_VEL")?)
                    }
                    "NUTATION" => nutation = Some(parse_apm_f64(val, "NUTATION")?),
                    "NUTATION_PER" => nutation_per = Some(parse_apm_f64(val, "NUTATION_PER")?),
                    "NUTATION_PHASE" => {
                        nutation_phase = Some(parse_apm_f64(val, "NUTATION_PHASE")?)
                    }
                    "MOMENTUM_ALPHA" => {
                        momentum_alpha = Some(parse_apm_f64(val, "MOMENTUM_ALPHA")?)
                    }
                    "MOMENTUM_DELTA" => {
                        momentum_delta = Some(parse_apm_f64(val, "MOMENTUM_DELTA")?)
                    }
                    "NUTATION_VEL" => nutation_vel = Some(parse_apm_f64(val, "NUTATION_VEL")?),
                    "SPIN_STOP" => {
                        let ref_frame_a = blk_ref_frame_a
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "REF_FRAME_A"))?;
                        let ref_frame_b = blk_ref_frame_b
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "REF_FRAME_B"))?;
                        let alpha = spin_alpha
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "SPIN_ALPHA"))?;
                        let delta = spin_delta
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "SPIN_DELTA"))?;
                        let angle = spin_angle
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "SPIN_ANGLE"))?;
                        let angle_vel = spin_angle_vel
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "SPIN_ANGLE_VEL"))?;
                        let mut spin = APMSpin::new(
                            ref_frame_a,
                            ref_frame_b,
                            alpha,
                            delta,
                            angle,
                            angle_vel,
                            AngleFormat::Degrees,
                        );

                        let nut_triple =
                            (nutation.take(), nutation_per.take(), nutation_phase.take());
                        let mom_triple = (
                            momentum_alpha.take(),
                            momentum_delta.take(),
                            nutation_vel.take(),
                        );
                        let nut_complete = matches!(nut_triple, (Some(_), Some(_), Some(_)));
                        let nut_partial = !nut_complete
                            && (nut_triple.0.is_some()
                                || nut_triple.1.is_some()
                                || nut_triple.2.is_some());
                        let mom_complete = matches!(mom_triple, (Some(_), Some(_), Some(_)));
                        let mom_partial = !mom_complete
                            && (mom_triple.0.is_some()
                                || mom_triple.1.is_some()
                                || mom_triple.2.is_some());

                        if nut_partial {
                            return Err(ccsds_parse_error(
                                "APM",
                                "incomplete spin nutation triple: NUTATION/NUTATION_PER/NUTATION_PHASE must all be present or all absent",
                            ));
                        }
                        if mom_partial {
                            return Err(ccsds_parse_error(
                                "APM",
                                "incomplete spin momentum triple: MOMENTUM_ALPHA/MOMENTUM_DELTA/NUTATION_VEL must all be present or all absent",
                            ));
                        }
                        if nut_complete && mom_complete {
                            return Err(ccsds_parse_error(
                                "APM",
                                "spin block cannot contain both the NUTATION triple and the MOMENTUM triple (504.0-B-2 §3.2.4.6)",
                            ));
                        }
                        if nut_complete {
                            spin = spin.with_nutation_angle(
                                nut_triple.0.unwrap(),
                                nut_triple.1.unwrap(),
                                nut_triple.2.unwrap(),
                                AngleFormat::Degrees,
                            );
                        } else if mom_complete {
                            spin = spin.with_nutation_momentum(
                                mom_triple.0.unwrap(),
                                mom_triple.1.unwrap(),
                                mom_triple.2.unwrap(),
                                AngleFormat::Degrees,
                            );
                        }
                        spin.comments = std::mem::take(&mut blk_comments);
                        spins.push(spin);
                        state = APMState::Data;
                    }
                    _ => {
                        return Err(ccsds_parse_error(
                            "APM",
                            &format!("unexpected keyword '{}' in spin block", key),
                        ));
                    }
                }
            }
            (APMState::Spin, KVNToken::Comment(text)) => blk_comments.push(text),
            (APMState::Spin, KVNToken::Empty) => {}

            // ===== INERTIA BLOCK =====
            (APMState::Inertia, KVNToken::KeyValue { key, value }) => {
                let val = strip_units(&value);
                match key.as_str() {
                    "INERTIA_REF_FRAME" => inertia_ref_frame = Some(ADMReferenceFrame::parse(val)),
                    "IXX" => ixx = Some(parse_apm_f64(val, "IXX")?),
                    "IYY" => iyy = Some(parse_apm_f64(val, "IYY")?),
                    "IZZ" => izz = Some(parse_apm_f64(val, "IZZ")?),
                    "IXY" => ixy = Some(parse_apm_f64(val, "IXY")?),
                    "IXZ" => ixz = Some(parse_apm_f64(val, "IXZ")?),
                    "IYZ" => iyz = Some(parse_apm_f64(val, "IYZ")?),
                    "INERTIA_STOP" => {
                        let frame = inertia_ref_frame
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "INERTIA_REF_FRAME"))?;
                        let xx = ixx
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "IXX"))?;
                        let yy = iyy
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "IYY"))?;
                        let zz = izz
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "IZZ"))?;
                        let xy = ixy
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "IXY"))?;
                        let xz = ixz
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "IXZ"))?;
                        let yz = iyz
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "IYZ"))?;
                        let mut inertia = APMInertia::new(frame, xx, yy, zz, xy, xz, yz);
                        inertia.comments = std::mem::take(&mut blk_comments);
                        inertias.push(inertia);
                        state = APMState::Data;
                    }
                    _ => {
                        return Err(ccsds_parse_error(
                            "APM",
                            &format!("unexpected keyword '{}' in inertia block", key),
                        ));
                    }
                }
            }
            (APMState::Inertia, KVNToken::Comment(text)) => blk_comments.push(text),
            (APMState::Inertia, KVNToken::Empty) => {}

            // ===== MANEUVER BLOCK =====
            (APMState::Man, KVNToken::KeyValue { key, value }) => {
                let val = strip_units(&value);
                match key.as_str() {
                    "MAN_EPOCH_START" => {
                        man_epoch_start =
                            Some(parse_ccsds_datetime(val, &active_ts(&time_system))?);
                    }
                    "MAN_DURATION" => {
                        man_duration = Some(parse_apm_f64(val, "MAN_DURATION")?);
                    }
                    "MAN_REF_FRAME" => man_ref_frame = Some(ADMReferenceFrame::parse(val)),
                    "MAN_TOR_X" => man_tor_x = Some(parse_apm_f64(val, "MAN_TOR_X")?),
                    "MAN_TOR_Y" => man_tor_y = Some(parse_apm_f64(val, "MAN_TOR_Y")?),
                    "MAN_TOR_Z" => man_tor_z = Some(parse_apm_f64(val, "MAN_TOR_Z")?),
                    "MAN_DELTA_MASS" => {
                        man_delta_mass = Some(parse_apm_f64(val, "MAN_DELTA_MASS")?);
                    }
                    "MAN_STOP" => {
                        let epoch_start = man_epoch_start
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "MAN_EPOCH_START"))?;
                        let duration = man_duration
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "MAN_DURATION"))?;
                        let frame = man_ref_frame
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "MAN_REF_FRAME"))?;
                        let tx = man_tor_x
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "MAN_TOR_X"))?;
                        let ty = man_tor_y
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "MAN_TOR_Y"))?;
                        let tz = man_tor_z
                            .take()
                            .ok_or_else(|| ccsds_missing_field("APM", "MAN_TOR_Z"))?;
                        let mut man = APMManeuver::new(
                            epoch_start,
                            duration,
                            frame,
                            Vector3::new(tx, ty, tz),
                        );
                        if let Some(dm) = man_delta_mass.take() {
                            man = man.with_delta_mass(dm);
                        }
                        man.comments = std::mem::take(&mut blk_comments);
                        maneuvers.push(man);
                        state = APMState::Data;
                    }
                    _ => {
                        return Err(ccsds_parse_error(
                            "APM",
                            &format!("unexpected keyword '{}' in maneuver block", key),
                        ));
                    }
                }
            }
            (APMState::Man, KVNToken::Comment(text)) => blk_comments.push(text),
            (APMState::Man, KVNToken::Empty) => {}

            // Catch unexpected tokens
            (st, token) => {
                return Err(ccsds_parse_error(
                    "APM",
                    &format!("unexpected token {:?} in state {:?}", token, st),
                ));
            }
        }
    }

    // A file that ends while the state machine is still inside a logical
    // block (a `*_START` without a matching `*_STOP`) is malformed per
    // 504.0-B-2 §3.2.4.3.
    let unterminated_block = match state {
        APMState::Quat => Some("QUAT"),
        APMState::Euler => Some("EULER"),
        APMState::AngVel => Some("ANGVEL"),
        APMState::Spin => Some("SPIN"),
        APMState::Inertia => Some("INERTIA"),
        APMState::Man => Some("MAN"),
        APMState::Header | APMState::Metadata | APMState::Data => None,
    };
    if let Some(block) = unterminated_block {
        return Err(ccsds_parse_error(
            "APM",
            &format!("unterminated {} block: missing {}_STOP", block, block),
        ));
    }

    let header = APMHeader {
        format_version: format_version
            .ok_or_else(|| ccsds_missing_field("APM", "CCSDS_APM_VERS"))?,
        classification,
        creation_date: creation_date.ok_or_else(|| ccsds_missing_field("APM", "CREATION_DATE"))?,
        originator: originator.ok_or_else(|| ccsds_missing_field("APM", "ORIGINATOR"))?,
        message_id,
        comments: header_comments,
    };

    let metadata = APMMetadata {
        object_name: object_name.ok_or_else(|| ccsds_missing_field("APM", "OBJECT_NAME"))?,
        object_id: object_id.ok_or_else(|| ccsds_missing_field("APM", "OBJECT_ID"))?,
        center_name,
        time_system: time_system.ok_or_else(|| ccsds_missing_field("APM", "TIME_SYSTEM"))?,
        comments: metadata_comments,
    };

    let apm = APM {
        header,
        metadata,
        epoch: epoch.ok_or_else(|| ccsds_missing_field("APM", "EPOCH"))?,
        comments: data_comments,
        quaternion_states,
        euler_states,
        angular_velocities,
        spins,
        inertias,
        maneuvers,
    };

    if !apm.has_blocks() {
        return Err(ccsds_missing_field("APM", "at least one logical block"));
    }

    Ok(apm)
}

/// Write an APM message to KVN format.
///
/// Requires at least one logical block to be present (504.0-B-2 §3.2.4.3);
/// use [`crate::ccsds::apm::APM::has_blocks`] to check before writing.
pub fn write_apm(apm: &APM) -> Result<String, BraheError> {
    if !apm.has_blocks() {
        return Err(ccsds_missing_field("APM", "at least one logical block"));
    }

    let mut out = String::new();

    // Header. Table 3-1 fixes the field order as VERS, COMMENT,
    // CLASSIFICATION, CREATION_DATE, ORIGINATOR, MESSAGE_ID; comments must
    // precede CLASSIFICATION so the parser (which routes any comment seen
    // after the first non-VERS header keyword to the metadata section)
    // attributes them back to the header on read.
    out.push_str(&format!(
        "CCSDS_APM_VERS = {:.1}\n",
        apm.header.format_version
    ));
    for comment in &apm.header.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }
    if let Some(ref class) = apm.header.classification {
        out.push_str(&format!("CLASSIFICATION = {}\n", class));
    }
    out.push_str(&format!(
        "CREATION_DATE = {}\n",
        format_ccsds_datetime_in(&apm.header.creation_date, &CCSDSTimeSystem::UTC)
    ));
    out.push_str(&format!("ORIGINATOR = {}\n", apm.header.originator));
    if let Some(ref msg_id) = apm.header.message_id {
        out.push_str(&format!("MESSAGE_ID = {}\n", msg_id));
    }
    out.push('\n');

    // Metadata
    for comment in &apm.metadata.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }
    out.push_str(&format!("OBJECT_NAME = {}\n", apm.metadata.object_name));
    out.push_str(&format!("OBJECT_ID = {}\n", apm.metadata.object_id));
    if let Some(ref center) = apm.metadata.center_name {
        out.push_str(&format!("CENTER_NAME = {}\n", center));
    }
    out.push_str(&format!("TIME_SYSTEM = {}\n", apm.metadata.time_system));

    // Data top-level comments and epoch
    for comment in &apm.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }
    out.push_str(&format!(
        "EPOCH = {}\n",
        format_ccsds_datetime_in(&apm.epoch, &apm.metadata.time_system)
    ));

    // Quaternion blocks
    for q in &apm.quaternion_states {
        out.push_str("\nQUAT_START\n");
        for comment in &q.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        out.push_str(&format!("REF_FRAME_A = {}\n", q.ref_frame_a));
        out.push_str(&format!("REF_FRAME_B = {}\n", q.ref_frame_b));
        let v = q.quaternion.to_vector(false);
        out.push_str(&format!("Q1 = {}\n", v[0]));
        out.push_str(&format!("Q2 = {}\n", v[1]));
        out.push_str(&format!("Q3 = {}\n", v[2]));
        out.push_str(&format!("QC = {}\n", v[3]));
        if let Some(d) = q.quaternion_derivative {
            // d is stored scalar-first; wire order is scalar-last.
            out.push_str(&format!("Q1_DOT = {}\n", d[1]));
            out.push_str(&format!("Q2_DOT = {}\n", d[2]));
            out.push_str(&format!("Q3_DOT = {}\n", d[3]));
            out.push_str(&format!("QC_DOT = {}\n", d[0]));
        }
        out.push_str("QUAT_STOP\n");
    }

    // Euler angle blocks
    for e in &apm.euler_states {
        out.push_str("\nEULER_START\n");
        for comment in &e.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        out.push_str(&format!("REF_FRAME_A = {}\n", e.ref_frame_a));
        out.push_str(&format!("REF_FRAME_B = {}\n", e.ref_frame_b));
        out.push_str(&format!(
            "EULER_ROT_SEQ = {}\n",
            format_euler_rot_seq(e.angles.order)
        ));
        out.push_str(&format!("ANGLE_1 = {}\n", e.angles.phi * RAD2DEG));
        out.push_str(&format!("ANGLE_2 = {}\n", e.angles.theta * RAD2DEG));
        out.push_str(&format!("ANGLE_3 = {}\n", e.angles.psi * RAD2DEG));
        if let Some(r) = e.rates {
            out.push_str(&format!("ANGLE_1_DOT = {}\n", r[0] * RAD2DEG));
            out.push_str(&format!("ANGLE_2_DOT = {}\n", r[1] * RAD2DEG));
            out.push_str(&format!("ANGLE_3_DOT = {}\n", r[2] * RAD2DEG));
        }
        out.push_str("EULER_STOP\n");
    }

    // Angular velocity blocks
    for av in &apm.angular_velocities {
        out.push_str("\nANGVEL_START\n");
        for comment in &av.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        out.push_str(&format!("REF_FRAME_A = {}\n", av.ref_frame_a));
        out.push_str(&format!("REF_FRAME_B = {}\n", av.ref_frame_b));
        out.push_str(&format!("ANGVEL_FRAME = {}\n", av.angvel_frame));
        out.push_str(&format!(
            "ANGVEL_X = {}\n",
            av.angular_velocity[0] * RAD2DEG
        ));
        out.push_str(&format!(
            "ANGVEL_Y = {}\n",
            av.angular_velocity[1] * RAD2DEG
        ));
        out.push_str(&format!(
            "ANGVEL_Z = {}\n",
            av.angular_velocity[2] * RAD2DEG
        ));
        out.push_str("ANGVEL_STOP\n");
    }

    // Spin blocks
    for s in &apm.spins {
        out.push_str("\nSPIN_START\n");
        for comment in &s.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        out.push_str(&format!("REF_FRAME_A = {}\n", s.ref_frame_a));
        out.push_str(&format!("REF_FRAME_B = {}\n", s.ref_frame_b));
        out.push_str(&format!("SPIN_ALPHA = {}\n", s.spin_alpha * RAD2DEG));
        out.push_str(&format!("SPIN_DELTA = {}\n", s.spin_delta * RAD2DEG));
        out.push_str(&format!("SPIN_ANGLE = {}\n", s.spin_angle * RAD2DEG));
        out.push_str(&format!(
            "SPIN_ANGLE_VEL = {}\n",
            s.spin_angle_vel * RAD2DEG
        ));
        match &s.nutation {
            APMNutation::None => {}
            APMNutation::Angle {
                nutation,
                nutation_period,
                nutation_phase,
            } => {
                out.push_str(&format!("NUTATION = {}\n", nutation * RAD2DEG));
                out.push_str(&format!("NUTATION_PER = {}\n", nutation_period));
                out.push_str(&format!("NUTATION_PHASE = {}\n", nutation_phase * RAD2DEG));
            }
            APMNutation::Momentum {
                momentum_alpha,
                momentum_delta,
                nutation_vel,
            } => {
                out.push_str(&format!("MOMENTUM_ALPHA = {}\n", momentum_alpha * RAD2DEG));
                out.push_str(&format!("MOMENTUM_DELTA = {}\n", momentum_delta * RAD2DEG));
                out.push_str(&format!("NUTATION_VEL = {}\n", nutation_vel * RAD2DEG));
            }
        }
        out.push_str("SPIN_STOP\n");
    }

    // Inertia blocks
    for i in &apm.inertias {
        out.push_str("\nINERTIA_START\n");
        for comment in &i.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        out.push_str(&format!("INERTIA_REF_FRAME = {}\n", i.inertia_ref_frame));
        out.push_str(&format!("IXX = {}\n", i.ixx));
        out.push_str(&format!("IYY = {}\n", i.iyy));
        out.push_str(&format!("IZZ = {}\n", i.izz));
        out.push_str(&format!("IXY = {}\n", i.ixy));
        out.push_str(&format!("IXZ = {}\n", i.ixz));
        out.push_str(&format!("IYZ = {}\n", i.iyz));
        out.push_str("INERTIA_STOP\n");
    }

    // Maneuver blocks
    for m in &apm.maneuvers {
        out.push_str("\nMAN_START\n");
        for comment in &m.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        out.push_str(&format!(
            "MAN_EPOCH_START = {}\n",
            format_ccsds_datetime_in(&m.epoch_start, &apm.metadata.time_system)
        ));
        out.push_str(&format!("MAN_DURATION = {}\n", m.duration));
        out.push_str(&format!("MAN_REF_FRAME = {}\n", m.ref_frame));
        out.push_str(&format!("MAN_TOR_X = {}\n", m.torque[0]));
        out.push_str(&format!("MAN_TOR_Y = {}\n", m.torque[1]));
        out.push_str(&format!("MAN_TOR_Z = {}\n", m.torque[2]));
        if let Some(dm) = m.delta_mass {
            out.push_str(&format!("MAN_DELTA_MASS = {}\n", dm));
        }
        out.push_str("MAN_STOP\n");
    }

    Ok(out)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::ccsds::common::CCSDSTimeSystem;
    use crate::ccsds::kvn::parse_apm;
    use crate::time::Epoch;
    use serial_test::parallel;

    fn apm_epoch(s: &str) -> Epoch {
        parse_ccsds_datetime(s, &CCSDSTimeSystem::UTC).unwrap()
    }

    #[test]
    #[parallel]
    fn test_parse_apm_example_g1_quaternion() {
        let content = std::fs::read_to_string("test_assets/ccsds/apm/APMExampleG1.txt").unwrap();
        let apm = parse_apm(&content).unwrap();

        // Header
        assert!((apm.header.format_version - 2.0).abs() < 1e-10);
        assert_eq!(apm.header.originator, "GSFC");
        assert_eq!(apm.header.message_id.as_deref(), Some("A7015Z1"));
        assert!((apm.header.creation_date - apm_epoch("2003-09-30T19:23:57")).abs() < 1e-6);
        assert!(apm.header.comments.is_empty());

        // Metadata
        assert_eq!(apm.metadata.object_name, "TRMM");
        assert_eq!(apm.metadata.object_id, "1997-074A");
        assert_eq!(apm.metadata.center_name.as_deref(), Some("EARTH"));
        assert_eq!(apm.metadata.time_system, CCSDSTimeSystem::UTC);
        assert_eq!(
            apm.metadata.comments,
            vec![
                "GEOCENTRIC, CARTESIAN, EARTH FIXED".to_string(),
                "OBJECT_ID: 1997-074A".to_string(),
                "$ITIM = 1997 NOV 21 22:26:18.40000000, $ original launch time".to_string(),
            ]
        );

        // Data top-level
        assert!((apm.epoch - apm_epoch("2003-09-30T14:28:15.1172")).abs() < 1e-6);
        assert_eq!(
            apm.comments,
            vec![
                "Current attitude for orbit 335".to_string(),
                "Attitude state quaternion".to_string(),
                "Accuracy of this attitude is 0.02 deg RSS.".to_string(),
            ]
        );

        // Quaternion block
        assert_eq!(apm.quaternion_states.len(), 1);
        let q = &apm.quaternion_states[0];
        assert_eq!(q.ref_frame_a, ADMReferenceFrame::parse("SC_BODY_1"));
        assert_eq!(q.ref_frame_b, ADMReferenceFrame::parse("ITRF1997"));
        let v = q.quaternion.to_vector(false);
        assert!((v[0] - 0.00005).abs() < 1e-4);
        assert!((v[1] - 0.87543).abs() < 1e-4);
        assert!((v[2] - 0.40949).abs() < 1e-4);
        assert!((v[3] - 0.25678).abs() < 1e-4);
        assert!(q.quaternion_derivative.is_none());
        assert!(q.comments.is_empty());
        assert!(apm.euler_states.is_empty());
        assert!(apm.angular_velocities.is_empty());
        assert!(apm.spins.is_empty());
        assert!(apm.inertias.is_empty());
        assert!(apm.maneuvers.is_empty());
    }

    #[test]
    #[parallel]
    fn test_parse_apm_example_g2_euler_angles() {
        let content = std::fs::read_to_string("test_assets/ccsds/apm/APMExampleG2.txt").unwrap();
        let apm = parse_apm(&content).unwrap();

        assert_eq!(apm.header.originator, "GSFC");
        assert_eq!(apm.header.message_id.as_deref(), Some("A7015Z2"));
        assert_eq!(apm.metadata.object_name, "GOES-P");
        assert_eq!(apm.metadata.object_id, "2006-003A");
        assert_eq!(apm.metadata.center_name.as_deref(), Some("EARTH"));
        assert!(apm.metadata.comments.is_empty());
        assert!((apm.epoch - apm_epoch("2006-03-12T09:56:39.4987")).abs() < 1e-6);
        assert_eq!(
            apm.comments,
            vec![
                "GEOSYNCHRONOUS, CARTESIAN, EARTH FIXED".to_string(),
                "OBJECT_ID: 2006-003A".to_string(),
                "$ITIM = 2006 FEB 5 03:23:45.60000000, $ original launch time".to_string(),
                "Attitude given by Euler angles".to_string(),
            ]
        );

        assert_eq!(apm.euler_states.len(), 1);
        let e = &apm.euler_states[0];
        assert_eq!(e.ref_frame_a, ADMReferenceFrame::parse("BODY_FRAME_A"));
        assert_eq!(e.ref_frame_b, ADMReferenceFrame::parse("ITRF1997"));
        assert!(matches!(e.angles.order, EulerAngleOrder::YXY));
        assert!((e.angles.phi - (-26.78_f64).to_radians()).abs() < 1e-10);
        assert!((e.angles.theta - 46.26_f64.to_radians()).abs() < 1e-10);
        assert!((e.angles.psi - 144.10_f64.to_radians()).abs() < 1e-10);
        assert!(e.rates.is_none());
        assert_eq!(e.comments, vec!["Euler angles".to_string()]);
        assert!(apm.quaternion_states.is_empty());
    }

    #[test]
    #[parallel]
    fn test_parse_apm_example_g3_multi_quat_inertia_maneuver() {
        let content = std::fs::read_to_string("test_assets/ccsds/apm/APMExampleG3.txt").unwrap();
        let apm = parse_apm(&content).unwrap();

        assert_eq!(apm.header.originator, "JPL");
        assert_eq!(apm.header.message_id.as_deref(), Some("900018"));
        assert_eq!(apm.metadata.object_name, "MARS SPIRIT");
        assert!(apm.metadata.comments.is_empty());
        assert_eq!(
            apm.comments,
            vec![
                "GEOCENTRIC, CARTESIAN, EARTH FIXED".to_string(),
                "OBJECT_ID: 2004-003".to_string(),
                "$ITIM = 2004 JAN 14 22:26:18.400000, $ original launch time 14:36".to_string(),
                "Generated by JPL".to_string(),
                "Current attitude for orbit 20 and attitude maneuver".to_string(),
                "planning data.".to_string(),
            ]
        );
        assert!((apm.epoch - apm_epoch("2004-02-14T14:28:15.1172")).abs() < 1e-6);

        // Two quaternion blocks
        assert_eq!(apm.quaternion_states.len(), 2);
        let q0 = &apm.quaternion_states[0];
        assert_eq!(q0.ref_frame_a, ADMReferenceFrame::parse("ITRF1997"));
        assert_eq!(q0.ref_frame_b, ADMReferenceFrame::parse("INSTRUMENT_A"));
        let v0 = q0.quaternion.to_vector(false);
        assert!((v0[0] - 0.03123).abs() < 1e-4);
        assert!((v0[1] - 0.78543).abs() < 1e-4);
        assert!((v0[2] - 0.39158).abs() < 1e-4);
        assert!((v0[3] - 0.47832).abs() < 1e-4);
        assert_eq!(
            q0.comments,
            vec!["Attitude state quaternion (ref frame = ITRF1997)".to_string()]
        );

        let q1 = &apm.quaternion_states[1];
        assert_eq!(q1.ref_frame_a, ADMReferenceFrame::parse("ICRF"));
        let v1 = q1.quaternion.to_vector(false);
        assert!((v1[0] - 0.02478).abs() < 1e-4);
        assert!((v1[1] - 0.78576).abs() < 1e-4);
        assert!((v1[2] - 0.39552).abs() < 1e-4);
        assert!((v1[3] - 0.47491).abs() < 1e-4);
        assert_eq!(
            q1.comments,
            vec!["Attitude state quaternion (ref frame = ICRF)".to_string()]
        );

        // Inertia block
        assert_eq!(apm.inertias.len(), 1);
        let inertia = &apm.inertias[0];
        assert_eq!(
            inertia.inertia_ref_frame,
            ADMReferenceFrame::parse("SC_BODY_1")
        );
        assert!((inertia.ixx - 6080.0).abs() < 1e-9);
        assert!((inertia.iyy - 5245.5).abs() < 1e-9);
        assert!((inertia.izz - 8067.3).abs() < 1e-9);
        assert!((inertia.ixy - (-135.9)).abs() < 1e-9);
        assert!((inertia.ixz - 89.3).abs() < 1e-9);
        assert!((inertia.iyz - (-90.7)).abs() < 1e-9);
        assert_eq!(
            inertia.comments,
            vec!["Spacecraft Inertia Parameters".to_string()]
        );

        // Maneuver block
        assert_eq!(apm.maneuvers.len(), 1);
        let man = &apm.maneuvers[0];
        assert!((man.epoch_start - apm_epoch("2004-02-14T14:29:00.5098")).abs() < 1e-6);
        assert!((man.duration - 3.0).abs() < 1e-9);
        assert_eq!(man.ref_frame, ADMReferenceFrame::parse("ICRF"));
        assert!((man.torque[0] - (-1.25)).abs() < 1e-9);
        assert!((man.torque[1] - (-0.5)).abs() < 1e-9);
        assert!((man.torque[2] - 0.5).abs() < 1e-9);
        assert!(man.delta_mass.is_none());
        assert_eq!(
            man.comments,
            vec![
                "Data follows for 1 planned maneuver.".to_string(),
                "First attitude maneuver for: MARS SPIRIT".to_string(),
                "Impulsive, torque direction fixed in body frame".to_string(),
            ]
        );

        assert!(apm.euler_states.is_empty());
        assert!(apm.angular_velocities.is_empty());
        assert!(apm.spins.is_empty());
    }

    #[test]
    #[parallel]
    fn test_parse_apm_v1_version_rejected() {
        let content = std::fs::read_to_string("test_assets/ccsds/apm/APM-v1-version.txt").unwrap();
        let err = parse_apm(&content).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("version 1.0"), "unexpected message: {}", msg);
        assert!(
            msg.contains("only version 2.0"),
            "unexpected message: {}",
            msg
        );
    }

    #[test]
    #[parallel]
    fn test_parse_apm_no_blocks_rejected() {
        let content = std::fs::read_to_string("test_assets/ccsds/apm/APM-no-blocks.txt").unwrap();
        let err = parse_apm(&content).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("at least one logical block"),
            "unexpected message: {}",
            msg
        );
    }

    #[test]
    #[parallel]
    fn test_parse_apm_unterminated_block_rejected() {
        // G-1's QUAT_START block truncated right after the `Q3 =` line: no
        // QC, no QUAT_STOP, so the parser reaches EOF still inside the
        // quaternion block.
        let content = std::fs::read_to_string("test_assets/ccsds/apm/APMExampleG1.txt").unwrap();
        let cutoff = content.find("Q3").expect("fixture must contain Q3");
        let line_end = content[cutoff..]
            .find('\n')
            .map(|i| cutoff + i)
            .expect("Q3 line must be followed by a newline");
        let truncated = &content[..line_end];

        let err = parse_apm(truncated).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("unterminated") && msg.contains("QUAT"),
            "unexpected message: {}",
            msg
        );
    }

    #[test]
    #[parallel]
    fn test_parse_apm_bad_euler_seq_rejected() {
        let content =
            std::fs::read_to_string("test_assets/ccsds/apm/APM-bad-euler-seq.txt").unwrap();
        let err = parse_apm(&content).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("invalid EULER_ROT_SEQ value '121'"),
            "unexpected message: {}",
            msg
        );
    }

    #[test]
    #[parallel]
    fn test_parse_apm_missing_ref_frame_rejected() {
        let content =
            std::fs::read_to_string("test_assets/ccsds/apm/APM-missing-ref-frame.txt").unwrap();
        let err = parse_apm(&content).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("missing required field 'REF_FRAME_A'"),
            "unexpected message: {}",
            msg
        );
    }

    /// Builds a minimal valid APM header/metadata/EPOCH prefix (through G-1's
    /// values) that a test can append a single logical block to.
    fn apm_prefix() -> String {
        "CCSDS_APM_VERS = 2.0\n\
CREATION_DATE = 2003-09-30T19:23:57\n\
ORIGINATOR = BRAHE\n\
OBJECT_NAME = TESTSAT\n\
OBJECT_ID = 2024-001A\n\
CENTER_NAME = EARTH\n\
TIME_SYSTEM = UTC\n\
EPOCH = 2003-09-30T14:28:15.1172\n"
            .to_string()
    }

    #[test]
    #[parallel]
    fn test_parse_apm_angular_velocity_block() {
        let content = apm_prefix()
            + "ANGVEL_START\n\
REF_FRAME_A = ICRF\n\
REF_FRAME_B = SC_BODY_1\n\
ANGVEL_FRAME = SC_BODY_1\n\
ANGVEL_X = 1.0\n\
ANGVEL_Y = 2.0\n\
ANGVEL_Z = 3.0\n\
ANGVEL_STOP\n";
        let apm = parse_apm(&content).unwrap();
        assert_eq!(apm.angular_velocities.len(), 1);
        let av = &apm.angular_velocities[0];
        assert_eq!(av.angvel_frame, ADMReferenceFrame::parse("SC_BODY_1"));
        assert!((av.angular_velocity[0] - 1.0_f64.to_radians()).abs() < 1e-10);
        assert!((av.angular_velocity[1] - 2.0_f64.to_radians()).abs() < 1e-10);
        assert!((av.angular_velocity[2] - 3.0_f64.to_radians()).abs() < 1e-10);
    }

    #[test]
    #[parallel]
    fn test_parse_apm_spin_nutation_angle_triple() {
        let content = apm_prefix()
            + "SPIN_START\n\
REF_FRAME_A = ICRF\n\
REF_FRAME_B = SC_BODY_1\n\
SPIN_ALPHA = 10.0\n\
SPIN_DELTA = 20.0\n\
SPIN_ANGLE = 30.0\n\
SPIN_ANGLE_VEL = 1.0\n\
NUTATION = 5.0\n\
NUTATION_PER = 100.0\n\
NUTATION_PHASE = 15.0\n\
SPIN_STOP\n";
        let apm = parse_apm(&content).unwrap();
        assert_eq!(apm.spins.len(), 1);
        match &apm.spins[0].nutation {
            APMNutation::Angle {
                nutation,
                nutation_period,
                nutation_phase,
            } => {
                assert!((nutation - 5.0_f64.to_radians()).abs() < 1e-10);
                assert!((nutation_period - 100.0).abs() < 1e-10);
                assert!((nutation_phase - 15.0_f64.to_radians()).abs() < 1e-10);
            }
            other => panic!("expected APMNutation::Angle, got {:?}", other),
        }
    }

    #[test]
    #[parallel]
    fn test_parse_apm_spin_momentum_triple() {
        let content = apm_prefix()
            + "SPIN_START\n\
REF_FRAME_A = ICRF\n\
REF_FRAME_B = SC_BODY_1\n\
SPIN_ALPHA = 10.0\n\
SPIN_DELTA = 20.0\n\
SPIN_ANGLE = 30.0\n\
SPIN_ANGLE_VEL = 1.0\n\
MOMENTUM_ALPHA = 7.0\n\
MOMENTUM_DELTA = 8.0\n\
NUTATION_VEL = 0.5\n\
SPIN_STOP\n";
        let apm = parse_apm(&content).unwrap();
        assert_eq!(apm.spins.len(), 1);
        match &apm.spins[0].nutation {
            APMNutation::Momentum {
                momentum_alpha,
                momentum_delta,
                nutation_vel,
            } => {
                assert!((momentum_alpha - 7.0_f64.to_radians()).abs() < 1e-10);
                assert!((momentum_delta - 8.0_f64.to_radians()).abs() < 1e-10);
                assert!((nutation_vel - 0.5_f64.to_radians()).abs() < 1e-10);
            }
            other => panic!("expected APMNutation::Momentum, got {:?}", other),
        }
    }

    #[test]
    #[parallel]
    fn test_parse_apm_spin_both_triples_rejected() {
        let content = apm_prefix()
            + "SPIN_START\n\
REF_FRAME_A = ICRF\n\
REF_FRAME_B = SC_BODY_1\n\
SPIN_ALPHA = 10.0\n\
SPIN_DELTA = 20.0\n\
SPIN_ANGLE = 30.0\n\
SPIN_ANGLE_VEL = 1.0\n\
NUTATION = 5.0\n\
NUTATION_PER = 100.0\n\
NUTATION_PHASE = 15.0\n\
MOMENTUM_ALPHA = 7.0\n\
MOMENTUM_DELTA = 8.0\n\
NUTATION_VEL = 0.5\n\
SPIN_STOP\n";
        let err = parse_apm(&content).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("cannot contain both"),
            "unexpected message: {}",
            msg
        );
    }

    #[test]
    #[parallel]
    fn test_parse_apm_spin_partial_triple_rejected() {
        let content = apm_prefix()
            + "SPIN_START\n\
REF_FRAME_A = ICRF\n\
REF_FRAME_B = SC_BODY_1\n\
SPIN_ALPHA = 10.0\n\
SPIN_DELTA = 20.0\n\
SPIN_ANGLE = 30.0\n\
SPIN_ANGLE_VEL = 1.0\n\
NUTATION = 5.0\n\
SPIN_STOP\n";
        let err = parse_apm(&content).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("incomplete spin nutation triple"),
            "unexpected message: {}",
            msg
        );
    }

    #[test]
    #[parallel]
    fn test_parse_apm_quaternion_derivative_round_trip() {
        let content = apm_prefix()
            + "QUAT_START\n\
REF_FRAME_A = ICRF\n\
REF_FRAME_B = SC_BODY_1\n\
Q1 = 0.0\n\
Q2 = 0.0\n\
Q3 = 0.0\n\
QC = 1.0\n\
Q1_DOT = 0.1\n\
Q2_DOT = 0.2\n\
Q3_DOT = 0.3\n\
QC_DOT = 0.4\n\
QUAT_STOP\n";
        let apm = parse_apm(&content).unwrap();
        let d = apm.quaternion_states[0].quaternion_derivative.unwrap();
        // Stored scalar-first: [QC_DOT, Q1_DOT, Q2_DOT, Q3_DOT]
        assert!((d[0] - 0.4).abs() < 1e-10);
        assert!((d[1] - 0.1).abs() < 1e-10);
        assert!((d[2] - 0.2).abs() < 1e-10);
        assert!((d[3] - 0.3).abs() < 1e-10);
    }

    #[test]
    #[parallel]
    fn test_parse_apm_quaternion_partial_derivative_rejected() {
        let content = apm_prefix()
            + "QUAT_START\n\
REF_FRAME_A = ICRF\n\
REF_FRAME_B = SC_BODY_1\n\
Q1 = 0.0\n\
Q2 = 0.0\n\
Q3 = 0.0\n\
QC = 1.0\n\
Q1_DOT = 0.1\n\
QUAT_STOP\n";
        let err = parse_apm(&content).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("incomplete quaternion derivative"),
            "unexpected message: {}",
            msg
        );
    }

    #[test]
    #[parallel]
    fn test_parse_apm_user_defined_rejected() {
        // USER_DEFINED_* is not part of APM (504.0-B-2 restricts APM's data
        // section to the six logical blocks in table 3-1; USER_DEFINED_* is
        // ODM/ACM-only per §3.2.4.2), so it must be rejected like any other
        // unrecognized keyword.
        let content = apm_prefix()
            + "QUAT_START\n\
REF_FRAME_A = ICRF\n\
REF_FRAME_B = SC_BODY_1\n\
Q1 = 0.0\n\
Q2 = 0.0\n\
Q3 = 0.0\n\
QC = 1.0\n\
QUAT_STOP\n\
USER_DEFINED_BATTERY_STATE = NOMINAL\n";
        let err = parse_apm(&content).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("USER_DEFINED_BATTERY_STATE") && msg.contains("not part of APM"),
            "unexpected message: {}",
            msg
        );
    }

    #[test]
    #[parallel]
    fn test_parse_apm_unknown_block_keyword_rejected() {
        let content = apm_prefix()
            + "QUAT_START\n\
REF_FRAME_A = ICRF\n\
REF_FRAME_B = SC_BODY_1\n\
BOGUS_KEY = 1.0\n\
Q1 = 0.0\n\
Q2 = 0.0\n\
Q3 = 0.0\n\
QC = 1.0\n\
QUAT_STOP\n";
        let err = parse_apm(&content).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("unexpected keyword 'BOGUS_KEY' in quaternion block"),
            "unexpected message: {}",
            msg
        );
    }

    /// Compares the fields the APM KVN codec round-trips exactly (header,
    /// metadata, epoch, and per-block counts/values), field by field.
    fn assert_apm_matches(a: &APM, b: &APM) {
        assert!((a.header.format_version - b.header.format_version).abs() < 1e-9);
        assert_eq!(a.header.originator, b.header.originator);
        assert_eq!(a.header.message_id, b.header.message_id);
        assert_eq!(a.metadata.object_name, b.metadata.object_name);
        assert_eq!(a.metadata.object_id, b.metadata.object_id);
        assert_eq!(a.metadata.center_name, b.metadata.center_name);
        assert_eq!(a.metadata.time_system, b.metadata.time_system);
        assert!((a.epoch - b.epoch).abs() < 1e-6);

        assert_eq!(a.quaternion_states.len(), b.quaternion_states.len());
        for (qa, qb) in a.quaternion_states.iter().zip(b.quaternion_states.iter()) {
            assert_eq!(qa.ref_frame_a, qb.ref_frame_a);
            assert_eq!(qa.ref_frame_b, qb.ref_frame_b);
            let va = qa.quaternion.to_vector(false);
            let vb = qb.quaternion.to_vector(false);
            for i in 0..4 {
                assert!((va[i] - vb[i]).abs() < 1e-6);
            }
        }

        assert_eq!(a.euler_states.len(), b.euler_states.len());
        for (ea, eb) in a.euler_states.iter().zip(b.euler_states.iter()) {
            assert_eq!(ea.ref_frame_a, eb.ref_frame_a);
            assert_eq!(ea.ref_frame_b, eb.ref_frame_b);
            assert!(ea.angles.order == eb.angles.order);
            assert!((ea.angles.phi - eb.angles.phi).abs() < 1e-9);
            assert!((ea.angles.theta - eb.angles.theta).abs() < 1e-9);
            assert!((ea.angles.psi - eb.angles.psi).abs() < 1e-9);
        }

        assert_eq!(a.inertias.len(), b.inertias.len());
        for (ia, ib) in a.inertias.iter().zip(b.inertias.iter()) {
            assert_eq!(ia.inertia_ref_frame, ib.inertia_ref_frame);
            assert!((ia.ixx - ib.ixx).abs() < 1e-6);
            assert!((ia.iyy - ib.iyy).abs() < 1e-6);
            assert!((ia.izz - ib.izz).abs() < 1e-6);
            assert!((ia.ixy - ib.ixy).abs() < 1e-6);
            assert!((ia.ixz - ib.ixz).abs() < 1e-6);
            assert!((ia.iyz - ib.iyz).abs() < 1e-6);
        }

        assert_eq!(a.maneuvers.len(), b.maneuvers.len());
        for (ma, mb) in a.maneuvers.iter().zip(b.maneuvers.iter()) {
            assert!((ma.epoch_start - mb.epoch_start).abs() < 1e-6);
            assert!((ma.duration - mb.duration).abs() < 1e-9);
            assert_eq!(ma.ref_frame, mb.ref_frame);
            for i in 0..3 {
                assert!((ma.torque[i] - mb.torque[i]).abs() < 1e-9);
            }
            assert_eq!(ma.delta_mass, mb.delta_mass);
        }
    }

    #[test]
    #[parallel]
    fn test_apm_kvn_round_trip_g1() {
        let content = std::fs::read_to_string("test_assets/ccsds/apm/APMExampleG1.txt").unwrap();
        let apm1 = parse_apm(&content).unwrap();
        let written = write_apm(&apm1).unwrap();
        let apm2 = parse_apm(&written).unwrap();
        assert_apm_matches(&apm1, &apm2);
    }

    #[test]
    #[parallel]
    fn test_apm_kvn_round_trip_g2() {
        let content = std::fs::read_to_string("test_assets/ccsds/apm/APMExampleG2.txt").unwrap();
        let apm1 = parse_apm(&content).unwrap();
        let written = write_apm(&apm1).unwrap();
        let apm2 = parse_apm(&written).unwrap();
        assert_apm_matches(&apm1, &apm2);
    }

    #[test]
    #[parallel]
    fn test_apm_kvn_round_trip_g3() {
        let content = std::fs::read_to_string("test_assets/ccsds/apm/APMExampleG3.txt").unwrap();
        let apm1 = parse_apm(&content).unwrap();
        let written = write_apm(&apm1).unwrap();
        let apm2 = parse_apm(&written).unwrap();
        assert_apm_matches(&apm1, &apm2);
    }

    #[test]
    #[parallel]
    fn test_apm_header_comments_and_classification_round_trip() {
        use crate::attitude::attitude_types::Quaternion;
        use crate::ccsds::apm::APMMetadata;
        use crate::ccsds::common::CCSDSTimeSystem;
        use crate::ccsds::frames::ADMReferenceFrame;
        use crate::time::Epoch;

        let metadata = APMMetadata::new("SAT1", "2024-001A", CCSDSTimeSystem::UTC);
        let mut apm = APM::new("BRAHE", metadata, Epoch::now());
        apm.header = apm
            .header
            .with_classification("UNCLASSIFIED")
            .with_comments(vec![
                "first header comment".to_string(),
                "second header comment".to_string(),
            ]);
        apm.push_quaternion_state(crate::ccsds::apm::APMQuaternionState::new(
            ADMReferenceFrame::parse("ICRF"),
            ADMReferenceFrame::parse("SC_BODY_1"),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
        ));

        let written = write_apm(&apm).unwrap();
        let vers_pos = written.find("CCSDS_APM_VERS").unwrap();
        let comment_pos = written.find("COMMENT first header comment").unwrap();
        let classification_pos = written.find("CLASSIFICATION").unwrap();
        assert!(vers_pos < comment_pos, "VERS must precede COMMENT");
        assert!(
            comment_pos < classification_pos,
            "COMMENT must precede CLASSIFICATION per table 3-1"
        );

        let parsed = parse_apm(&written).unwrap();
        assert_eq!(
            parsed.header.classification.as_deref(),
            Some("UNCLASSIFIED")
        );
        assert_eq!(
            parsed.header.comments,
            vec![
                "first header comment".to_string(),
                "second header comment".to_string(),
            ]
        );
        assert!(parsed.metadata.comments.is_empty());
    }

    #[test]
    #[parallel]
    fn test_apm_write_no_blocks_rejected() {
        use crate::ccsds::apm::APMMetadata;
        use crate::ccsds::common::CCSDSTimeSystem;
        use crate::time::Epoch;

        let metadata = APMMetadata::new("SAT1", "2024-001A", CCSDSTimeSystem::UTC);
        let apm = APM::new("BRAHE", metadata, Epoch::now());
        let err = write_apm(&apm).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("at least one logical block"),
            "unexpected message: {}",
            msg
        );
    }
}
