/*!
 * KVN reader and writer for the Attitude Ephemeris Message (AEM).
 *
 * Reference: CCSDS 504.0-B-2 (Attitude Data Messages), section 4
 */

use nalgebra::{Vector3, Vector4};

use crate::attitude::attitude_types::{EulerAngle, EulerAngleOrder, Quaternion};
use crate::ccsds::aem::{
    AEM, AEMAttitudeData, AEMAttitudeState, AEMAttitudeType, AEMHeader, AEMInterpolationMethod,
    AEMMetadata, AEMSegment,
};
use crate::ccsds::common::{
    CCSDSTimeSystem, format_ccsds_datetime, format_ccsds_datetime_in, format_euler_rot_seq,
    parse_ccsds_datetime, parse_euler_rot_seq,
};
use crate::ccsds::error::{ccsds_missing_field, ccsds_parse_error};
use crate::ccsds::frames::ADMReferenceFrame;
use crate::ccsds::kvn::common::{KVNToken, tokenize_line};
use crate::constants::{AngleFormat, DEG2RAD, RAD2DEG};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

/// Parser state for AEM KVN parsing.
#[derive(Debug, PartialEq)]
enum AemState {
    Header,
    Metadata,
    /// After META_STOP, before DATA_START.
    AwaitingDataStart,
    /// Between DATA_START and DATA_STOP.
    DataBlock,
    /// After DATA_STOP, before the next META_START or a clean EOF.
    AfterDataStop,
}

/// Parses a single AEM ephemeris data line into an [`AEMAttitudeState`].
///
/// Column layout and units are fixed per `attitude_type` (504.0-B-2 table
/// 4-4). `euler_seq` supplies the rotation sequence for Euler angle types
/// (guaranteed present by [`AEMMetadata::validate`], which runs at
/// `META_STOP` before any data line is parsed).
fn parse_aem_data_line(
    parts: &[String],
    attitude_type: AEMAttitudeType,
    euler_seq: Option<EulerAngleOrder>,
    time_system: &CCSDSTimeSystem,
) -> Result<AEMAttitudeState, BraheError> {
    let expected_columns = match attitude_type {
        AEMAttitudeType::Quaternion => 5,
        AEMAttitudeType::QuaternionDerivative => 9,
        AEMAttitudeType::QuaternionAngVel => 8,
        AEMAttitudeType::EulerAngle => 4,
        AEMAttitudeType::EulerAngleDerivative => 7,
        AEMAttitudeType::EulerAngleAngVel => 7,
        AEMAttitudeType::Spin => 5,
        AEMAttitudeType::SpinNutation => 8,
        AEMAttitudeType::SpinNutationMom => 8,
    };
    if parts.len() != expected_columns {
        return Err(ccsds_parse_error(
            "AEM",
            &format!(
                "data line has {} columns, expected {} for ATTITUDE_TYPE '{}'",
                parts.len(),
                expected_columns,
                attitude_type
            ),
        ));
    }

    let epoch = parse_ccsds_datetime(&parts[0], time_system)?;
    let col = |i: usize, name: &str| -> Result<f64, BraheError> {
        parts[i].parse::<f64>().map_err(|_| {
            ccsds_parse_error("AEM", &format!("invalid {} value '{}'", name, parts[i]))
        })
    };

    let data = match attitude_type {
        AEMAttitudeType::Quaternion => {
            let (q1, q2, q3, qc) = (col(1, "Q1")?, col(2, "Q2")?, col(3, "Q3")?, col(4, "QC")?);
            AEMAttitudeData::Quaternion {
                quaternion: Quaternion::from_vector(Vector4::new(q1, q2, q3, qc), false),
            }
        }
        AEMAttitudeType::QuaternionDerivative => {
            let (q1, q2, q3, qc) = (col(1, "Q1")?, col(2, "Q2")?, col(3, "Q3")?, col(4, "QC")?);
            let (q1d, q2d, q3d, qcd) = (
                col(5, "Q1_DOT")?,
                col(6, "Q2_DOT")?,
                col(7, "Q3_DOT")?,
                col(8, "QC_DOT")?,
            );
            AEMAttitudeData::QuaternionDerivative {
                quaternion: Quaternion::from_vector(Vector4::new(q1, q2, q3, qc), false),
                // derivative is stored scalar-first; wire order is scalar-last.
                derivative: Vector4::new(qcd, q1d, q2d, q3d),
            }
        }
        AEMAttitudeType::QuaternionAngVel => {
            let (q1, q2, q3, qc) = (col(1, "Q1")?, col(2, "Q2")?, col(3, "Q3")?, col(4, "QC")?);
            let (wx, wy, wz) = (
                col(5, "ANGVEL_X")?,
                col(6, "ANGVEL_Y")?,
                col(7, "ANGVEL_Z")?,
            );
            AEMAttitudeData::QuaternionAngVel {
                quaternion: Quaternion::from_vector(Vector4::new(q1, q2, q3, qc), false),
                angular_velocity: Vector3::new(wx * DEG2RAD, wy * DEG2RAD, wz * DEG2RAD),
            }
        }
        AEMAttitudeType::EulerAngle => {
            let seq = euler_seq.ok_or_else(|| ccsds_missing_field("AEM", "EULER_ROT_SEQ"))?;
            let (a1, a2, a3) = (col(1, "ANGLE_1")?, col(2, "ANGLE_2")?, col(3, "ANGLE_3")?);
            AEMAttitudeData::EulerAngle {
                angles: EulerAngle::new(seq, a1, a2, a3, AngleFormat::Degrees),
            }
        }
        AEMAttitudeType::EulerAngleDerivative => {
            let seq = euler_seq.ok_or_else(|| ccsds_missing_field("AEM", "EULER_ROT_SEQ"))?;
            let (a1, a2, a3) = (col(1, "ANGLE_1")?, col(2, "ANGLE_2")?, col(3, "ANGLE_3")?);
            let (a1d, a2d, a3d) = (
                col(4, "ANGLE_1_DOT")?,
                col(5, "ANGLE_2_DOT")?,
                col(6, "ANGLE_3_DOT")?,
            );
            AEMAttitudeData::EulerAngleDerivative {
                angles: EulerAngle::new(seq, a1, a2, a3, AngleFormat::Degrees),
                rates: Vector3::new(a1d * DEG2RAD, a2d * DEG2RAD, a3d * DEG2RAD),
            }
        }
        AEMAttitudeType::EulerAngleAngVel => {
            let seq = euler_seq.ok_or_else(|| ccsds_missing_field("AEM", "EULER_ROT_SEQ"))?;
            let (a1, a2, a3) = (col(1, "ANGLE_1")?, col(2, "ANGLE_2")?, col(3, "ANGLE_3")?);
            let (wx, wy, wz) = (
                col(4, "ANGVEL_X")?,
                col(5, "ANGVEL_Y")?,
                col(6, "ANGVEL_Z")?,
            );
            AEMAttitudeData::EulerAngleAngVel {
                angles: EulerAngle::new(seq, a1, a2, a3, AngleFormat::Degrees),
                angular_velocity: Vector3::new(wx * DEG2RAD, wy * DEG2RAD, wz * DEG2RAD),
            }
        }
        AEMAttitudeType::Spin => AEMAttitudeData::Spin {
            spin_alpha: col(1, "SPIN_ALPHA")? * DEG2RAD,
            spin_delta: col(2, "SPIN_DELTA")? * DEG2RAD,
            spin_angle: col(3, "SPIN_ANGLE")? * DEG2RAD,
            spin_angle_vel: col(4, "SPIN_ANGLE_VEL")? * DEG2RAD,
        },
        AEMAttitudeType::SpinNutation => AEMAttitudeData::SpinNutation {
            spin_alpha: col(1, "SPIN_ALPHA")? * DEG2RAD,
            spin_delta: col(2, "SPIN_DELTA")? * DEG2RAD,
            spin_angle: col(3, "SPIN_ANGLE")? * DEG2RAD,
            spin_angle_vel: col(4, "SPIN_ANGLE_VEL")? * DEG2RAD,
            nutation: col(5, "NUTATION")? * DEG2RAD,
            nutation_period: col(6, "NUTATION_PER")?,
            nutation_phase: col(7, "NUTATION_PHASE")? * DEG2RAD,
        },
        AEMAttitudeType::SpinNutationMom => AEMAttitudeData::SpinNutationMom {
            spin_alpha: col(1, "SPIN_ALPHA")? * DEG2RAD,
            spin_delta: col(2, "SPIN_DELTA")? * DEG2RAD,
            spin_angle: col(3, "SPIN_ANGLE")? * DEG2RAD,
            spin_angle_vel: col(4, "SPIN_ANGLE_VEL")? * DEG2RAD,
            momentum_alpha: col(5, "MOMENTUM_ALPHA")? * DEG2RAD,
            momentum_delta: col(6, "MOMENTUM_DELTA")? * DEG2RAD,
            nutation_vel: col(7, "NUTATION_VEL")? * DEG2RAD,
        },
    };

    Ok(AEMAttitudeState { epoch, data })
}

/// Parse an AEM message from KVN format.
pub fn parse_aem(content: &str) -> Result<AEM, BraheError> {
    let mut state = AemState::Header;

    // Header fields
    let mut format_version: Option<f64> = None;
    let mut classification: Option<String> = None;
    let mut creation_date: Option<Epoch> = None;
    let mut originator: Option<String> = None;
    let mut message_id: Option<String> = None;
    let mut header_comments: Vec<String> = Vec::new();

    // Metadata fields being built
    let mut meta_object_name: Option<String> = None;
    let mut meta_object_id: Option<String> = None;
    let mut meta_center_name: Option<String> = None;
    let mut meta_ref_frame_a: Option<ADMReferenceFrame> = None;
    let mut meta_ref_frame_b: Option<ADMReferenceFrame> = None;
    let mut meta_time_system: Option<CCSDSTimeSystem> = None;
    let mut meta_start_time: Option<Epoch> = None;
    let mut meta_useable_start_time: Option<Epoch> = None;
    let mut meta_useable_stop_time: Option<Epoch> = None;
    let mut meta_stop_time: Option<Epoch> = None;
    let mut meta_attitude_type: Option<AEMAttitudeType> = None;
    let mut meta_euler_rot_seq: Option<EulerAngleOrder> = None;
    let mut meta_angvel_frame: Option<ADMReferenceFrame> = None;
    let mut meta_interpolation_method: Option<AEMInterpolationMethod> = None;
    let mut meta_interpolation_degree: Option<u32> = None;
    let mut metadata_comments: Vec<String> = Vec::new();

    // We need the time system from the current segment's metadata to parse
    // epochs in its data block.
    let mut active_time_system = CCSDSTimeSystem::UTC;

    let mut segments: Vec<AEMSegment> = Vec::new();
    let mut current_segment: Option<AEMSegment> = None;

    for line in content.lines() {
        let token = tokenize_line(line);

        match (&state, token) {
            // === HEADER STATE ===
            (AemState::Header, KVNToken::KeyValue { key, value }) => match key.as_str() {
                "CCSDS_AEM_VERS" => {
                    let v: f64 = value.parse().map_err(|_| {
                        ccsds_parse_error("AEM", &format!("invalid version '{}'", value))
                    })?;
                    if (v - 1.0).abs() < 1e-9 {
                        return Err(ccsds_parse_error(
                            "AEM",
                            "version 1.0 (504.0-B-1) files are not supported; only version 2.0",
                        ));
                    }
                    format_version = Some(v);
                }
                "CLASSIFICATION" => classification = Some(value),
                "CREATION_DATE" => {
                    creation_date = Some(parse_ccsds_datetime(&value, &CCSDSTimeSystem::UTC)?);
                }
                "ORIGINATOR" => originator = Some(value),
                "MESSAGE_ID" => message_id = Some(value),
                "META_START" => {
                    state = AemState::Metadata;
                }
                _ => {
                    return Err(ccsds_parse_error(
                        "AEM",
                        &format!("unexpected header keyword '{}'", key),
                    ));
                }
            },
            (AemState::Header, KVNToken::Comment(text)) => header_comments.push(text),
            (AemState::Header, KVNToken::Empty) => {}

            // === METADATA STATE ===
            (AemState::Metadata, KVNToken::KeyValue { key, value }) => match key.as_str() {
                "META_START" => {
                    // Already in metadata (re-entry handled by state machine)
                }
                "OBJECT_NAME" => meta_object_name = Some(value),
                "OBJECT_ID" => meta_object_id = Some(value),
                "CENTER_NAME" => meta_center_name = Some(value),
                "REF_FRAME_A" => meta_ref_frame_a = Some(ADMReferenceFrame::parse(&value)),
                "REF_FRAME_B" => meta_ref_frame_b = Some(ADMReferenceFrame::parse(&value)),
                "TIME_SYSTEM" => {
                    let ts = CCSDSTimeSystem::parse(&value)?;
                    active_time_system = ts.clone();
                    meta_time_system = Some(ts);
                }
                "START_TIME" => {
                    let ts = meta_time_system.as_ref().unwrap_or(&CCSDSTimeSystem::UTC);
                    meta_start_time = Some(parse_ccsds_datetime(&value, ts)?);
                }
                "USEABLE_START_TIME" => {
                    let ts = meta_time_system.as_ref().unwrap_or(&CCSDSTimeSystem::UTC);
                    meta_useable_start_time = Some(parse_ccsds_datetime(&value, ts)?);
                }
                "USEABLE_STOP_TIME" => {
                    let ts = meta_time_system.as_ref().unwrap_or(&CCSDSTimeSystem::UTC);
                    meta_useable_stop_time = Some(parse_ccsds_datetime(&value, ts)?);
                }
                "STOP_TIME" => {
                    let ts = meta_time_system.as_ref().unwrap_or(&CCSDSTimeSystem::UTC);
                    meta_stop_time = Some(parse_ccsds_datetime(&value, ts)?);
                }
                "ATTITUDE_TYPE" => {
                    meta_attitude_type = Some(AEMAttitudeType::parse(&value)?);
                }
                "EULER_ROT_SEQ" => {
                    meta_euler_rot_seq = Some(parse_euler_rot_seq(&value)?);
                }
                "ANGVEL_FRAME" => meta_angvel_frame = Some(ADMReferenceFrame::parse(&value)),
                "INTERPOLATION_METHOD" => {
                    meta_interpolation_method = Some(AEMInterpolationMethod::parse(&value)?);
                }
                "INTERPOLATION_DEGREE" => {
                    meta_interpolation_degree = Some(value.parse::<u32>().map_err(|_| {
                        ccsds_parse_error(
                            "AEM",
                            &format!("invalid interpolation degree '{}'", value),
                        )
                    })?);
                }
                "META_STOP" => {
                    let metadata = AEMMetadata {
                        object_name: meta_object_name
                            .take()
                            .ok_or_else(|| ccsds_missing_field("AEM", "OBJECT_NAME"))?,
                        object_id: meta_object_id
                            .take()
                            .ok_or_else(|| ccsds_missing_field("AEM", "OBJECT_ID"))?,
                        center_name: meta_center_name.take(),
                        ref_frame_a: meta_ref_frame_a
                            .take()
                            .ok_or_else(|| ccsds_missing_field("AEM", "REF_FRAME_A"))?,
                        ref_frame_b: meta_ref_frame_b
                            .take()
                            .ok_or_else(|| ccsds_missing_field("AEM", "REF_FRAME_B"))?,
                        time_system: meta_time_system
                            .take()
                            .ok_or_else(|| ccsds_missing_field("AEM", "TIME_SYSTEM"))?,
                        start_time: meta_start_time
                            .take()
                            .ok_or_else(|| ccsds_missing_field("AEM", "START_TIME"))?,
                        stop_time: meta_stop_time
                            .take()
                            .ok_or_else(|| ccsds_missing_field("AEM", "STOP_TIME"))?,
                        useable_start_time: meta_useable_start_time.take(),
                        useable_stop_time: meta_useable_stop_time.take(),
                        attitude_type: meta_attitude_type
                            .take()
                            .ok_or_else(|| ccsds_missing_field("AEM", "ATTITUDE_TYPE"))?,
                        euler_rot_seq: meta_euler_rot_seq.take(),
                        angvel_frame: meta_angvel_frame.take(),
                        interpolation_method: meta_interpolation_method.take(),
                        interpolation_degree: meta_interpolation_degree.take(),
                        comments: std::mem::take(&mut metadata_comments),
                    };
                    metadata.validate()?;

                    current_segment = Some(AEMSegment::new(metadata));
                    state = AemState::AwaitingDataStart;
                }
                _ => {
                    return Err(ccsds_parse_error(
                        "AEM",
                        &format!("unexpected metadata keyword '{}'", key),
                    ));
                }
            },
            (AemState::Metadata, KVNToken::Comment(text)) => metadata_comments.push(text),
            (AemState::Metadata, KVNToken::Empty) => {}

            // === AWAITING DATA_START ===
            (AemState::AwaitingDataStart, KVNToken::KeyValue { key, value: _ }) => {
                match key.as_str() {
                    "DATA_START" => state = AemState::DataBlock,
                    _ => {
                        return Err(ccsds_parse_error(
                            "AEM",
                            &format!("unexpected keyword '{}' before DATA_START", key),
                        ));
                    }
                }
            }
            (AemState::AwaitingDataStart, KVNToken::Comment(text)) => {
                current_segment.as_mut().unwrap().comments.push(text);
            }
            (AemState::AwaitingDataStart, KVNToken::Empty) => {}

            // === DATA BLOCK ===
            (AemState::DataBlock, KVNToken::DataLine(parts)) => {
                let segment = current_segment.as_mut().unwrap();
                let attitude_type = segment.metadata.attitude_type;
                let euler_seq = segment.metadata.euler_rot_seq;
                let attitude_state =
                    parse_aem_data_line(&parts, attitude_type, euler_seq, &active_time_system)?;
                segment.push_state(attitude_state)?;
            }
            (AemState::DataBlock, KVNToken::KeyValue { key, value: _ }) => match key.as_str() {
                "DATA_STOP" => state = AemState::AfterDataStop,
                _ => {
                    return Err(ccsds_parse_error(
                        "AEM",
                        &format!("unexpected keyword '{}' in data block", key),
                    ));
                }
            },
            (AemState::DataBlock, KVNToken::Comment(text)) => {
                current_segment.as_mut().unwrap().comments.push(text);
            }
            (AemState::DataBlock, KVNToken::Empty) => {}

            // === AFTER DATA_STOP ===
            (AemState::AfterDataStop, KVNToken::KeyValue { key, value: _ }) => match key.as_str() {
                "META_START" => {
                    if let Some(seg) = current_segment.take() {
                        segments.push(seg);
                    }
                    state = AemState::Metadata;
                }
                _ => {
                    return Err(ccsds_parse_error(
                        "AEM",
                        &format!("unexpected keyword '{}' after DATA_STOP", key),
                    ));
                }
            },
            (AemState::AfterDataStop, KVNToken::Comment(text)) => {
                current_segment.as_mut().unwrap().comments.push(text);
            }
            (AemState::AfterDataStop, KVNToken::Empty) => {}

            // Catch unexpected tokens
            (st, token) => {
                return Err(ccsds_parse_error(
                    "AEM",
                    &format!("unexpected token {:?} in state {:?}", token, st),
                ));
            }
        }
    }

    // A file that ends while a segment is not cleanly closed out (metadata
    // still open, or its data block never started or never closed) is
    // malformed per 504.0-B-2 §4.2.3.2 / §4.2.4.1.
    let unterminated = match state {
        AemState::Metadata => Some("META block: missing META_STOP"),
        AemState::AwaitingDataStart => Some("DATA block: missing DATA_START"),
        AemState::DataBlock => Some("DATA block: missing DATA_STOP"),
        AemState::Header | AemState::AfterDataStop => None,
    };
    if let Some(detail) = unterminated {
        return Err(ccsds_parse_error(
            "AEM",
            &format!("unterminated {}", detail),
        ));
    }

    if let Some(seg) = current_segment.take() {
        segments.push(seg);
    }

    let header = AEMHeader {
        format_version: format_version
            .ok_or_else(|| ccsds_missing_field("AEM", "CCSDS_AEM_VERS"))?,
        classification,
        creation_date: creation_date.ok_or_else(|| ccsds_missing_field("AEM", "CREATION_DATE"))?,
        originator: originator.ok_or_else(|| ccsds_missing_field("AEM", "ORIGINATOR"))?,
        message_id,
        comments: header_comments,
    };

    Ok(AEM { header, segments })
}


/// Write an AEM message to KVN format.
pub fn write_aem(aem: &AEM) -> Result<String, BraheError> {
    let mut out = String::new();

    // Header. Table 4-2 fixes the field order as VERS, COMMENT,
    // CLASSIFICATION, CREATION_DATE, ORIGINATOR, MESSAGE_ID; comments must
    // precede CLASSIFICATION so the parser (which routes any comment seen
    // after the first non-VERS header keyword to the metadata section)
    // attributes them back to the header on read.
    out.push_str(&format!(
        "CCSDS_AEM_VERS = {:.1}\n",
        aem.header.format_version
    ));
    for comment in &aem.header.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }
    if let Some(ref class) = aem.header.classification {
        out.push_str(&format!("CLASSIFICATION = {}\n", class));
    }
    out.push_str(&format!(
        "CREATION_DATE = {}\n",
        format_ccsds_datetime_in(&aem.header.creation_date, &CCSDSTimeSystem::UTC)
    ));
    out.push_str(&format!("ORIGINATOR = {}\n", aem.header.originator));
    if let Some(ref msg_id) = aem.header.message_id {
        out.push_str(&format!("MESSAGE_ID = {}\n", msg_id));
    }

    for segment in &aem.segments {
        out.push('\n');

        // Epochs are written in the segment's own metadata TIME_SYSTEM
        // (504.0-B-2 §4.2.4.4), not the `Epoch`'s internal time system. A
        // handful of CCSDS time systems (SCLK, MET, MRT, GMST, TDR) have no
        // corresponding `crate::time::TimeSystem` — they are spacecraft- or
        // mission-specific clocks with no fixed relationship to the
        // physical time systems `Epoch` represents — so for those the
        // epoch is written as stored, unconverted.
        let write_ts = segment.metadata.time_system.to_time_system();
        let epoch_for_write = |e: &crate::time::Epoch| -> crate::time::Epoch {
            match write_ts {
                Some(ts) => e.to_time_system(ts),
                None => *e,
            }
        };

        out.push_str("META_START\n");
        for comment in &segment.metadata.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        out.push_str(&format!("OBJECT_NAME = {}\n", segment.metadata.object_name));
        out.push_str(&format!("OBJECT_ID = {}\n", segment.metadata.object_id));
        if let Some(ref center) = segment.metadata.center_name {
            out.push_str(&format!("CENTER_NAME = {}\n", center));
        }
        out.push_str(&format!("REF_FRAME_A = {}\n", segment.metadata.ref_frame_a));
        out.push_str(&format!("REF_FRAME_B = {}\n", segment.metadata.ref_frame_b));
        out.push_str(&format!("TIME_SYSTEM = {}\n", segment.metadata.time_system));
        out.push_str(&format!(
            "START_TIME = {}\n",
            format_ccsds_datetime(&epoch_for_write(&segment.metadata.start_time))
        ));
        if let Some(ref t) = segment.metadata.useable_start_time {
            out.push_str(&format!(
                "USEABLE_START_TIME = {}\n",
                format_ccsds_datetime(&epoch_for_write(t))
            ));
        }
        if let Some(ref t) = segment.metadata.useable_stop_time {
            out.push_str(&format!(
                "USEABLE_STOP_TIME = {}\n",
                format_ccsds_datetime(&epoch_for_write(t))
            ));
        }
        out.push_str(&format!(
            "STOP_TIME = {}\n",
            format_ccsds_datetime(&epoch_for_write(&segment.metadata.stop_time))
        ));
        out.push_str(&format!(
            "ATTITUDE_TYPE = {}\n",
            segment.metadata.attitude_type
        ));
        if let Some(seq) = segment.metadata.euler_rot_seq {
            out.push_str(&format!("EULER_ROT_SEQ = {}\n", format_euler_rot_seq(seq)));
        }
        if let Some(ref frame) = segment.metadata.angvel_frame {
            out.push_str(&format!("ANGVEL_FRAME = {}\n", frame));
        }
        if let Some(method) = segment.metadata.interpolation_method {
            out.push_str(&format!("INTERPOLATION_METHOD = {}\n", method));
        }
        if let Some(degree) = segment.metadata.interpolation_degree {
            out.push_str(&format!("INTERPOLATION_DEGREE = {}\n", degree));
        }
        out.push_str("META_STOP\n");

        out.push_str("\nDATA_START\n");
        for comment in &segment.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }

        for attitude_state in &segment.states {
            let epoch_str = format_ccsds_datetime(&epoch_for_write(&attitude_state.epoch));
            match &attitude_state.data {
                AEMAttitudeData::Quaternion { quaternion } => {
                    let v = quaternion.to_vector(false);
                    out.push_str(&format!(
                        "{} {} {} {} {}\n",
                        epoch_str, v[0], v[1], v[2], v[3]
                    ));
                }
                AEMAttitudeData::QuaternionDerivative {
                    quaternion,
                    derivative,
                } => {
                    let v = quaternion.to_vector(false);
                    // derivative is stored scalar-first; wire order is scalar-last.
                    out.push_str(&format!(
                        "{} {} {} {} {} {} {} {} {}\n",
                        epoch_str,
                        v[0],
                        v[1],
                        v[2],
                        v[3],
                        derivative[1],
                        derivative[2],
                        derivative[3],
                        derivative[0]
                    ));
                }
                AEMAttitudeData::QuaternionAngVel {
                    quaternion,
                    angular_velocity,
                } => {
                    let v = quaternion.to_vector(false);
                    out.push_str(&format!(
                        "{} {} {} {} {} {} {} {}\n",
                        epoch_str,
                        v[0],
                        v[1],
                        v[2],
                        v[3],
                        angular_velocity[0] * RAD2DEG,
                        angular_velocity[1] * RAD2DEG,
                        angular_velocity[2] * RAD2DEG
                    ));
                }
                AEMAttitudeData::EulerAngle { angles } => {
                    out.push_str(&format!(
                        "{} {} {} {}\n",
                        epoch_str,
                        angles.phi * RAD2DEG,
                        angles.theta * RAD2DEG,
                        angles.psi * RAD2DEG
                    ));
                }
                AEMAttitudeData::EulerAngleDerivative { angles, rates } => {
                    out.push_str(&format!(
                        "{} {} {} {} {} {} {}\n",
                        epoch_str,
                        angles.phi * RAD2DEG,
                        angles.theta * RAD2DEG,
                        angles.psi * RAD2DEG,
                        rates[0] * RAD2DEG,
                        rates[1] * RAD2DEG,
                        rates[2] * RAD2DEG
                    ));
                }
                AEMAttitudeData::EulerAngleAngVel {
                    angles,
                    angular_velocity,
                } => {
                    out.push_str(&format!(
                        "{} {} {} {} {} {} {}\n",
                        epoch_str,
                        angles.phi * RAD2DEG,
                        angles.theta * RAD2DEG,
                        angles.psi * RAD2DEG,
                        angular_velocity[0] * RAD2DEG,
                        angular_velocity[1] * RAD2DEG,
                        angular_velocity[2] * RAD2DEG
                    ));
                }
                AEMAttitudeData::Spin {
                    spin_alpha,
                    spin_delta,
                    spin_angle,
                    spin_angle_vel,
                } => {
                    out.push_str(&format!(
                        "{} {} {} {} {}\n",
                        epoch_str,
                        spin_alpha * RAD2DEG,
                        spin_delta * RAD2DEG,
                        spin_angle * RAD2DEG,
                        spin_angle_vel * RAD2DEG
                    ));
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
                    out.push_str(&format!(
                        "{} {} {} {} {} {} {} {}\n",
                        epoch_str,
                        spin_alpha * RAD2DEG,
                        spin_delta * RAD2DEG,
                        spin_angle * RAD2DEG,
                        spin_angle_vel * RAD2DEG,
                        nutation * RAD2DEG,
                        nutation_period,
                        nutation_phase * RAD2DEG
                    ));
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
                    out.push_str(&format!(
                        "{} {} {} {} {} {} {} {}\n",
                        epoch_str,
                        spin_alpha * RAD2DEG,
                        spin_delta * RAD2DEG,
                        spin_angle * RAD2DEG,
                        spin_angle_vel * RAD2DEG,
                        momentum_alpha * RAD2DEG,
                        momentum_delta * RAD2DEG,
                        nutation_vel * RAD2DEG
                    ));
                }
            }
        }
        out.push_str("DATA_STOP\n");
    }

    Ok(out)
}


#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::parallel;

    // ------------------------------------------------------------------
    // AEM
    // ------------------------------------------------------------------

    fn aem_epoch(s: &str) -> Epoch {
        parse_ccsds_datetime(s, &CCSDSTimeSystem::UTC).unwrap()
    }

    #[test]
    #[parallel]
    fn test_parse_aem_example_g4_quaternion_two_segments() {
        let content = std::fs::read_to_string("test_assets/ccsds/aem/AEMExampleG4.txt").unwrap();
        let aem = parse_aem(&content).unwrap();

        // Header
        assert!((aem.header.format_version - 2.0).abs() < 1e-10);
        assert_eq!(aem.header.originator, "NASA/JPL");
        assert_eq!(aem.header.message_id.as_deref(), Some("A7015Z3"));
        assert!((aem.header.creation_date - aem_epoch("2002-11-04T17:22:31")).abs() < 1e-6);
        assert!(aem.header.comments.is_empty());

        assert_eq!(aem.segments.len(), 2);

        // --- Segment 1 ---
        let seg0 = &aem.segments[0];
        assert_eq!(seg0.metadata.object_name, "MARS GLOBAL SURVEYOR");
        assert_eq!(seg0.metadata.object_id, "1996-062A");
        assert_eq!(
            seg0.metadata.center_name.as_deref(),
            Some("MARS BARYCENTER")
        );
        assert_eq!(
            seg0.metadata.ref_frame_a,
            ADMReferenceFrame::parse("EME2000")
        );
        assert_eq!(
            seg0.metadata.ref_frame_b,
            ADMReferenceFrame::parse("SC_BODY_1")
        );
        assert_eq!(seg0.metadata.time_system, CCSDSTimeSystem::UTC);
        assert!((seg0.metadata.start_time - aem_epoch("1996-11-28T21:29:07.2555")).abs() < 1e-6);
        assert!(
            (seg0.metadata.useable_start_time.unwrap() - aem_epoch("1996-11-28T22:08:02.5555"))
                .abs()
                < 1e-6
        );
        assert!(
            (seg0.metadata.useable_stop_time.unwrap() - aem_epoch("1996-11-30T01:18:02.5555"))
                .abs()
                < 1e-6
        );
        assert!((seg0.metadata.stop_time - aem_epoch("1996-11-30T01:28:02.5555")).abs() < 1e-6);
        assert_eq!(seg0.metadata.attitude_type, AEMAttitudeType::Quaternion);
        assert_eq!(
            seg0.metadata.interpolation_method,
            Some(AEMInterpolationMethod::Hermite)
        );
        assert_eq!(seg0.metadata.interpolation_degree, Some(7));
        assert_eq!(
            seg0.metadata.comments,
            vec![
                "This file was produced by M.R. Somebody, MSOO NAV/JPL.".to_string(),
                "It is to be used for attitude reconstruction only. The relative accuracy of \
                 these"
                    .to_string(),
                "attitudes is 0.1 degrees per axis.".to_string(),
            ]
        );
        assert!(seg0.comments.is_empty());

        assert_eq!(seg0.states.len(), 4);
        let first = &seg0.states[0];
        assert!((first.epoch - aem_epoch("1996-11-28T21:29:07.2555")).abs() < 1e-6);
        match &first.data {
            AEMAttitudeData::Quaternion { quaternion } => {
                let v = quaternion.to_vector(false);
                assert!((v[0] - 0.56748).abs() < 1e-4);
                assert!((v[1] - 0.03146).abs() < 1e-4);
                assert!((v[2] - 0.45689).abs() < 1e-4);
                assert!((v[3] - 0.68427).abs() < 1e-4);
            }
            other => panic!("expected Quaternion, got {:?}", other),
        }
        let last = &seg0.states[3];
        assert!((last.epoch - aem_epoch("1996-11-30T01:28:02.5555")).abs() < 1e-6);
        match &last.data {
            AEMAttitudeData::Quaternion { quaternion } => {
                let v = quaternion.to_vector(false);
                assert!((v[0] - 0.74563).abs() < 1e-4);
                assert!((v[1] - (-0.45375)).abs() < 1e-4);
                assert!((v[2] - 0.36875).abs() < 1e-4);
                assert!((v[3] - 0.31964).abs() < 1e-4);
            }
            other => panic!("expected Quaternion, got {:?}", other),
        }

        // --- Segment 2 ---
        let seg1 = &aem.segments[1];
        assert_eq!(seg1.metadata.object_name, "mars global surveyor");
        assert_eq!(seg1.metadata.object_id, "1996-062A");
        assert_eq!(
            seg1.metadata.center_name.as_deref(),
            Some("MARS BARYCENTER")
        );
        assert_eq!(
            seg1.metadata.ref_frame_a,
            ADMReferenceFrame::parse("EME2000")
        );
        assert_eq!(
            seg1.metadata.ref_frame_b,
            ADMReferenceFrame::parse("SC_BODY_1")
        );
        assert!((seg1.metadata.start_time - aem_epoch("1996-12-18T12:05:00.5555")).abs() < 1e-6);
        assert!(
            (seg1.metadata.useable_start_time.unwrap() - aem_epoch("1996-12-18T12:10:00.5555"))
                .abs()
                < 1e-6
        );
        assert!(
            (seg1.metadata.useable_stop_time.unwrap() - aem_epoch("1996-12-28T21:23:00.5555"))
                .abs()
                < 1e-6
        );
        assert!((seg1.metadata.stop_time - aem_epoch("1996-12-28T21:28:00.5555")).abs() < 1e-6);
        assert_eq!(seg1.metadata.attitude_type, AEMAttitudeType::Quaternion);
        assert!(seg1.metadata.interpolation_method.is_none());
        assert!(seg1.metadata.interpolation_degree.is_none());
        assert_eq!(
            seg1.metadata.comments,
            vec!["This block begins after trajectory correction maneuver TCM-3.".to_string()]
        );

        assert_eq!(seg1.states.len(), 4);
        let first = &seg1.states[0];
        assert!((first.epoch - aem_epoch("1996-12-18T12:05:00.5555")).abs() < 1e-6);
        match &first.data {
            AEMAttitudeData::Quaternion { quaternion } => {
                let v = quaternion.to_vector(false);
                assert!((v[0] - (-0.64585)).abs() < 1e-4);
                assert!((v[1] - 0.018542).abs() < 1e-4);
                assert!((v[2] - (-0.23854)).abs() < 1e-4);
                assert!((v[3] - 0.72501).abs() < 1e-4);
            }
            other => panic!("expected Quaternion, got {:?}", other),
        }
        let last = &seg1.states[3];
        assert!((last.epoch - aem_epoch("1996-12-28T21:28:00.5555")).abs() < 1e-6);
        match &last.data {
            AEMAttitudeData::Quaternion { quaternion } => {
                let v = quaternion.to_vector(false);
                assert!((v[0] - (-0.25485)).abs() < 1e-4);
                assert!((v[1] - 0.58745).abs() < 1e-4);
                assert!((v[2] - (-0.36845)).abs() < 1e-4);
                assert!((v[3] - 0.67394).abs() < 1e-4);
            }
            other => panic!("expected Quaternion, got {:?}", other),
        }
    }

    #[test]
    #[parallel]
    fn test_parse_aem_example_g5_spin() {
        let content = std::fs::read_to_string("test_assets/ccsds/aem/AEMExampleG5.txt").unwrap();
        let aem = parse_aem(&content).unwrap();

        assert_eq!(aem.header.originator, "GSFC");
        assert_eq!(aem.header.message_id.as_deref(), Some("7077456"));

        assert_eq!(aem.segments.len(), 1);
        let seg = &aem.segments[0];
        assert_eq!(seg.metadata.object_name, "ST5-224");
        assert_eq!(seg.metadata.object_id, "2006-224A");
        assert_eq!(seg.metadata.center_name.as_deref(), Some("EARTH"));
        assert_eq!(seg.metadata.ref_frame_a, ADMReferenceFrame::parse("J2000"));
        assert_eq!(
            seg.metadata.ref_frame_b,
            ADMReferenceFrame::parse("SC_BODY_1")
        );
        assert_eq!(seg.metadata.attitude_type, AEMAttitudeType::Spin);
        assert_eq!(
            seg.comments,
            vec!["Spin KF ground solution, SPINKF rates".to_string()]
        );

        assert_eq!(seg.states.len(), 8);
        let first = &seg.states[0];
        match &first.data {
            AEMAttitudeData::Spin {
                spin_alpha,
                spin_delta,
                spin_angle,
                spin_angle_vel,
            } => {
                assert!((*spin_alpha - 268.62511_f64.to_radians()).abs() < 1e-9);
                assert!((*spin_delta - 68.448486_f64.to_radians()).abs() < 1e-9);
                assert!((*spin_angle - 159.69509_f64.to_radians()).abs() < 1e-9);
                assert!((*spin_angle_vel - (-109.96528_f64).to_radians()).abs() < 1e-9);
            }
            other => panic!("expected Spin, got {:?}", other),
        }
        let last = &seg.states[7];
        match &last.data {
            AEMAttitudeData::Spin {
                spin_alpha,
                spin_delta,
                spin_angle,
                spin_angle_vel,
            } => {
                assert!((*spin_alpha - 268.43571_f64.to_radians()).abs() < 1e-9);
                assert!((*spin_delta - 68.332398_f64.to_radians()).abs() < 1e-9);
                assert!((*spin_angle - 63.662262_f64.to_radians()).abs() < 1e-9);
                assert!((*spin_angle_vel - (-109.96304_f64).to_radians()).abs() < 1e-9);
            }
            other => panic!("expected Spin, got {:?}", other),
        }
    }

    #[test]
    #[parallel]
    fn test_parse_aem_v1_version_rejected() {
        let content = std::fs::read_to_string("test_assets/ccsds/aem/AEM-v1-version.txt").unwrap();
        let err = parse_aem(&content).unwrap_err();
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
    fn test_parse_aem_decreasing_epochs_rejected() {
        let content =
            std::fs::read_to_string("test_assets/ccsds/aem/AEM-decreasing-epochs.txt").unwrap();
        let err = parse_aem(&content).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("not strictly increasing"),
            "unexpected message: {}",
            msg
        );
    }

    #[test]
    #[parallel]
    fn test_parse_aem_wrong_columns_rejected() {
        let content =
            std::fs::read_to_string("test_assets/ccsds/aem/AEM-wrong-columns.txt").unwrap();
        let err = parse_aem(&content).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("4 columns") && msg.contains("expected 5") && msg.contains("QUATERNION"),
            "unexpected message: {}",
            msg
        );
    }

    #[test]
    #[parallel]
    fn test_parse_aem_missing_euler_seq_rejected() {
        let content =
            std::fs::read_to_string("test_assets/ccsds/aem/AEM-missing-euler-seq.txt").unwrap();
        let err = parse_aem(&content).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("EULER_ROT_SEQ") && msg.contains("EULER_ANGLE"),
            "unexpected message: {}",
            msg
        );
    }

    /// Builds a minimal valid AEM header/metadata prefix (through G-4's
    /// segment-1 values, minus interpolation) that a test can append a
    /// DATA_START/data/DATA_STOP block to.
    fn aem_prefix() -> String {
        "CCSDS_AEM_VERS = 2.0\n\
CREATION_DATE = 2002-11-04T17:22:31\n\
ORIGINATOR = BRAHE\n\
\n\
META_START\n\
OBJECT_NAME = TESTSAT\n\
OBJECT_ID = 2024-001A\n\
CENTER_NAME = EARTH\n\
REF_FRAME_A = EME2000\n\
REF_FRAME_B = SC_BODY_1\n\
TIME_SYSTEM = UTC\n\
START_TIME = 1996-11-28T21:29:07.2555\n\
STOP_TIME = 1996-11-30T01:28:02.5555\n\
ATTITUDE_TYPE = QUATERNION\n\
META_STOP\n"
            .to_string()
    }

    #[test]
    #[parallel]
    fn test_parse_aem_unterminated_metadata_rejected() {
        let content = "CCSDS_AEM_VERS = 2.0\n\
CREATION_DATE = 2002-11-04T17:22:31\n\
ORIGINATOR = BRAHE\n\
\n\
META_START\n\
OBJECT_NAME = TESTSAT\n";
        let err = parse_aem(content).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("unterminated") && msg.contains("META") && msg.contains("META_STOP"),
            "unexpected message: {}",
            msg
        );
    }

    #[test]
    #[parallel]
    fn test_parse_aem_unterminated_missing_data_start_rejected() {
        let content = aem_prefix();
        let err = parse_aem(&content).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("unterminated") && msg.contains("DATA") && msg.contains("DATA_START"),
            "unexpected message: {}",
            msg
        );
    }

    #[test]
    #[parallel]
    fn test_parse_aem_unterminated_data_block_rejected() {
        let content = aem_prefix()
            + "\nDATA_START\n\
1996-11-28T21:29:07.2555 0.56748  0.03146  0.45689  0.68427\n";
        let err = parse_aem(&content).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("unterminated") && msg.contains("DATA") && msg.contains("DATA_STOP"),
            "unexpected message: {}",
            msg
        );
    }

    #[test]
    #[parallel]
    fn test_parse_aem_missing_object_name_rejected() {
        let content = "CCSDS_AEM_VERS = 2.0\n\
CREATION_DATE = 2002-11-04T17:22:31\n\
ORIGINATOR = BRAHE\n\
\n\
META_START\n\
OBJECT_ID = 2024-001A\n\
CENTER_NAME = EARTH\n\
REF_FRAME_A = EME2000\n\
REF_FRAME_B = SC_BODY_1\n\
TIME_SYSTEM = UTC\n\
START_TIME = 1996-11-28T21:29:07.2555\n\
STOP_TIME = 1996-11-30T01:28:02.5555\n\
ATTITUDE_TYPE = QUATERNION\n\
META_STOP\n";
        let err = parse_aem(content).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("missing required field 'OBJECT_NAME'"),
            "unexpected message: {}",
            msg
        );
    }

    #[test]
    #[parallel]
    fn test_parse_aem_angvel_frame_mismatch_rejected() {
        let content = "CCSDS_AEM_VERS = 2.0\n\
CREATION_DATE = 2002-11-04T17:22:31\n\
ORIGINATOR = BRAHE\n\
\n\
META_START\n\
OBJECT_NAME = TESTSAT\n\
OBJECT_ID = 2024-001A\n\
CENTER_NAME = EARTH\n\
REF_FRAME_A = EME2000\n\
REF_FRAME_B = SC_BODY_1\n\
TIME_SYSTEM = UTC\n\
START_TIME = 1996-11-28T21:29:07.2555\n\
STOP_TIME = 1996-11-30T01:28:02.5555\n\
ATTITUDE_TYPE = QUATERNION/ANGVEL\n\
ANGVEL_FRAME = INSTRUMENT_A\n\
META_STOP\n";
        let err = parse_aem(content).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("must equal REF_FRAME_A"),
            "unexpected message: {}",
            msg
        );
    }

    // ------------------------------------------------------------------
    // AEM
    // ------------------------------------------------------------------

    use crate::ccsds::kvn::parse_aem;

    /// Compares the fields the AEM KVN codec round-trips exactly (header,
    /// per-segment metadata, and per-state data), field by field.
    fn assert_aem_matches(a: &AEM, b: &AEM) {
        assert!((a.header.format_version - b.header.format_version).abs() < 1e-9);
        assert_eq!(a.header.originator, b.header.originator);
        assert_eq!(a.header.message_id, b.header.message_id);
        assert_eq!(a.segments.len(), b.segments.len());

        for (sa, sb) in a.segments.iter().zip(b.segments.iter()) {
            assert_eq!(sa.metadata.object_name, sb.metadata.object_name);
            assert_eq!(sa.metadata.object_id, sb.metadata.object_id);
            assert_eq!(sa.metadata.center_name, sb.metadata.center_name);
            assert_eq!(sa.metadata.ref_frame_a, sb.metadata.ref_frame_a);
            assert_eq!(sa.metadata.ref_frame_b, sb.metadata.ref_frame_b);
            assert_eq!(sa.metadata.time_system, sb.metadata.time_system);
            assert!((sa.metadata.start_time - sb.metadata.start_time).abs() < 1e-6);
            assert!((sa.metadata.stop_time - sb.metadata.stop_time).abs() < 1e-6);
            assert_eq!(sa.metadata.attitude_type, sb.metadata.attitude_type);
            assert_eq!(
                sa.metadata.euler_rot_seq.is_some(),
                sb.metadata.euler_rot_seq.is_some()
            );
            assert_eq!(sa.metadata.angvel_frame, sb.metadata.angvel_frame);
            assert_eq!(
                sa.metadata.interpolation_method,
                sb.metadata.interpolation_method
            );
            assert_eq!(
                sa.metadata.interpolation_degree,
                sb.metadata.interpolation_degree
            );

            assert_eq!(sa.states.len(), sb.states.len());
            for (state_a, state_b) in sa.states.iter().zip(sb.states.iter()) {
                assert!((state_a.epoch - state_b.epoch).abs() < 1e-6);
                match (&state_a.data, &state_b.data) {
                    (
                        AEMAttitudeData::Quaternion { quaternion: qa },
                        AEMAttitudeData::Quaternion { quaternion: qb },
                    ) => {
                        let va = qa.to_vector(false);
                        let vb = qb.to_vector(false);
                        for i in 0..4 {
                            assert!((va[i] - vb[i]).abs() < 1e-6);
                        }
                    }
                    (
                        AEMAttitudeData::Spin {
                            spin_alpha: aa,
                            spin_delta: da,
                            spin_angle: ga,
                            spin_angle_vel: va,
                        },
                        AEMAttitudeData::Spin {
                            spin_alpha: ab,
                            spin_delta: db,
                            spin_angle: gb,
                            spin_angle_vel: vb,
                        },
                    ) => {
                        assert!((aa - ab).abs() < 1e-9);
                        assert!((da - db).abs() < 1e-9);
                        assert!((ga - gb).abs() < 1e-9);
                        assert!((va - vb).abs() < 1e-9);
                    }
                    (data_a, data_b) => {
                        panic!(
                            "unexpected data variant mismatch: {:?} vs {:?}",
                            data_a, data_b
                        )
                    }
                }
            }
        }
    }

    #[test]
    #[parallel]
    fn test_aem_kvn_round_trip_g4() {
        let content = std::fs::read_to_string("test_assets/ccsds/aem/AEMExampleG4.txt").unwrap();
        let aem1 = parse_aem(&content).unwrap();
        let written = write_aem(&aem1).unwrap();
        let aem2 = parse_aem(&written).unwrap();
        assert_aem_matches(&aem1, &aem2);
    }

    #[test]
    #[parallel]
    fn test_aem_kvn_round_trip_g5() {
        let content = std::fs::read_to_string("test_assets/ccsds/aem/AEMExampleG5.txt").unwrap();
        let aem1 = parse_aem(&content).unwrap();
        let written = write_aem(&aem1).unwrap();
        let aem2 = parse_aem(&written).unwrap();
        assert_aem_matches(&aem1, &aem2);
    }

    #[test]
    #[parallel]
    fn test_aem_write_interpolation_method_all_upper() {
        let content = std::fs::read_to_string("test_assets/ccsds/aem/AEMExampleG4.txt").unwrap();
        let aem = parse_aem(&content).unwrap();
        let written = write_aem(&aem).unwrap();
        assert!(written.contains("INTERPOLATION_METHOD = HERMITE"));
    }

    #[test]
    #[parallel]
    fn test_aem_write_quaternion_wire_order() {
        // The wire order is Q1 Q2 Q3 QC (vector-first, scalar-last); the
        // writer must not reorder to scalar-first.
        let content = std::fs::read_to_string("test_assets/ccsds/aem/AEMExampleG4.txt").unwrap();
        let aem = parse_aem(&content).unwrap();
        let written = write_aem(&aem).unwrap();
        let data_line = written
            .lines()
            .find(|l| l.starts_with("1996-11-28T21:29:07"))
            .expect("first data line must be present");
        let cols: Vec<&str> = data_line.split_whitespace().collect();
        assert_eq!(cols.len(), 5);
        let qc: f64 = cols[4].parse().unwrap();
        assert!((qc - 0.68427).abs() < 1e-4);
    }
    #[test]
    #[parallel]
    fn test_aem_header_comments_and_classification_round_trip() {
        use crate::attitude::attitude_types::Quaternion;
        use crate::ccsds::aem::{AEMAttitudeState, AEMAttitudeType, AEMMetadata, AEMSegment};
        use crate::ccsds::common::CCSDSTimeSystem;
        use crate::ccsds::frames::ADMReferenceFrame;
        use crate::time::{Epoch, TimeSystem};

        let metadata = AEMMetadata::new(
            "SAT1",
            "2024-001A",
            ADMReferenceFrame::parse("ICRF"),
            ADMReferenceFrame::parse("SC_BODY_1"),
            CCSDSTimeSystem::UTC,
            Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC),
            Epoch::from_datetime(2024, 1, 1, 1, 0, 0.0, 0.0, TimeSystem::UTC),
            AEMAttitudeType::Quaternion,
        );
        let mut segment = AEMSegment::new(metadata);
        segment
            .push_state(AEMAttitudeState {
                epoch: Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC),
                data: AEMAttitudeData::Quaternion {
                    quaternion: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                },
            })
            .unwrap();

        let mut aem = AEM::new("BRAHE");
        aem.header = aem
            .header
            .with_classification("UNCLASSIFIED")
            .with_comments(vec![
                "first header comment".to_string(),
                "second header comment".to_string(),
            ]);
        aem.push_segment(segment);

        let written = write_aem(&aem).unwrap();
        let vers_pos = written.find("CCSDS_AEM_VERS").unwrap();
        let comment_pos = written.find("COMMENT first header comment").unwrap();
        let classification_pos = written.find("CLASSIFICATION").unwrap();
        assert!(vers_pos < comment_pos, "VERS must precede COMMENT");
        assert!(
            comment_pos < classification_pos,
            "COMMENT must precede CLASSIFICATION per table 4-2"
        );

        let parsed = parse_aem(&written).unwrap();
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
        assert!(parsed.segments[0].metadata.comments.is_empty());
    }
}
