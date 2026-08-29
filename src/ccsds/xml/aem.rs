/*!
 * XML reader and writer for the Attitude Ephemeris Message (AEM).
 *
 * Reference: CCSDS 504.0-B-2 (Attitude Data Messages), section 4
 */

use nalgebra::{Vector3, Vector4};
use serde::Deserialize;

use crate::attitude::attitude_types::{EulerAngle, EulerAngleOrder, Quaternion};
use crate::ccsds::aem::{
    AEM, AEMAttitudeData, AEMAttitudeState, AEMAttitudeType, AEMHeader, AEMInterpolationMethod,
    AEMMetadata, AEMSegment, resolve_angvel_frame_token,
};
use crate::ccsds::common::{
    CCSDSTimeSystem, format_ccsds_datetime, format_ccsds_datetime_in, format_euler_rot_seq,
    parse_ccsds_datetime, parse_euler_rot_seq,
};
use crate::ccsds::error::{ccsds_missing_field, ccsds_parse_error};
use crate::ccsds::frames::ADMReferenceFrame;
use crate::ccsds::xml::common::{
    XMLHeader, XMLQuaternion, XMLQuaternionDot, XMLValue, escape_xml_text,
};
use crate::constants::{AngleFormat, DEG2RAD, RAD2DEG};
use crate::utils::errors::BraheError;

// ============================================================================
// Intermediate XML structs for AEM
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(rename = "aem")]
#[allow(clippy::upper_case_acronyms)]
struct XMLAEM {
    #[serde(rename = "@version")]
    version: Option<String>,
    header: XMLHeader,
    body: XMLAEMBody,
}

#[derive(Debug, Deserialize)]
struct XMLAEMBody {
    #[serde(rename = "segment")]
    segments: Vec<XMLAEMSegment>,
}

#[derive(Debug, Deserialize)]
struct XMLAEMSegment {
    metadata: XMLAEMMetadata,
    data: XMLAEMData,
}

#[derive(Debug, Deserialize)]
struct XMLAEMMetadata {
    #[serde(rename = "$value")]
    items: Vec<XMLAEMMetadataItem>,
}

#[derive(Debug, Deserialize)]
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
enum XMLAEMMetadataItem {
    OBJECT_NAME(String),
    OBJECT_ID(String),
    CENTER_NAME(String),
    REF_FRAME_A(String),
    REF_FRAME_B(String),
    TIME_SYSTEM(String),
    START_TIME(String),
    USEABLE_START_TIME(String),
    USEABLE_STOP_TIME(String),
    STOP_TIME(String),
    ATTITUDE_TYPE(String),
    EULER_ROT_SEQ(String),
    ANGVEL_FRAME(String),
    INTERPOLATION_METHOD(String),
    INTERPOLATION_DEGREE(u32),
    COMMENT(String),
}

impl XMLAEMMetadata {
    fn find_str(&self, variant: &str) -> Option<&str> {
        self.items.iter().find_map(|item| match item {
            XMLAEMMetadataItem::OBJECT_NAME(s) if variant == "OBJECT_NAME" => Some(s.as_str()),
            XMLAEMMetadataItem::OBJECT_ID(s) if variant == "OBJECT_ID" => Some(s.as_str()),
            XMLAEMMetadataItem::CENTER_NAME(s) if variant == "CENTER_NAME" => Some(s.as_str()),
            XMLAEMMetadataItem::REF_FRAME_A(s) if variant == "REF_FRAME_A" => Some(s.as_str()),
            XMLAEMMetadataItem::REF_FRAME_B(s) if variant == "REF_FRAME_B" => Some(s.as_str()),
            XMLAEMMetadataItem::TIME_SYSTEM(s) if variant == "TIME_SYSTEM" => Some(s.as_str()),
            XMLAEMMetadataItem::START_TIME(s) if variant == "START_TIME" => Some(s.as_str()),
            XMLAEMMetadataItem::USEABLE_START_TIME(s) if variant == "USEABLE_START_TIME" => {
                Some(s.as_str())
            }
            XMLAEMMetadataItem::USEABLE_STOP_TIME(s) if variant == "USEABLE_STOP_TIME" => {
                Some(s.as_str())
            }
            XMLAEMMetadataItem::STOP_TIME(s) if variant == "STOP_TIME" => Some(s.as_str()),
            XMLAEMMetadataItem::ATTITUDE_TYPE(s) if variant == "ATTITUDE_TYPE" => Some(s.as_str()),
            XMLAEMMetadataItem::EULER_ROT_SEQ(s) if variant == "EULER_ROT_SEQ" => Some(s.as_str()),
            XMLAEMMetadataItem::ANGVEL_FRAME(s) if variant == "ANGVEL_FRAME" => Some(s.as_str()),
            XMLAEMMetadataItem::INTERPOLATION_METHOD(s) if variant == "INTERPOLATION_METHOD" => {
                Some(s.as_str())
            }
            _ => None,
        })
    }

    fn interpolation_degree(&self) -> Option<u32> {
        self.items.iter().find_map(|item| {
            if let XMLAEMMetadataItem::INTERPOLATION_DEGREE(v) = item {
                Some(*v)
            } else {
                None
            }
        })
    }

    fn comments(&self) -> Vec<String> {
        self.items
            .iter()
            .filter_map(|item| {
                if let XMLAEMMetadataItem::COMMENT(s) = item {
                    Some(s.trim().to_string())
                } else {
                    None
                }
            })
            .collect()
    }
}

/// AEM data block containing comments and the ordered sequence of
/// `<attitudeState>` elements.
///
/// Uses `$value` to capture all child elements as a flat sequence, since
/// quick-xml cannot handle multiple `<COMMENT>`/`<attitudeState>` elements as
/// a `Vec` directly.
#[derive(Debug, Deserialize)]
struct XMLAEMData {
    #[serde(rename = "$value", default)]
    items: Vec<XMLAEMDataItem>,
}

#[derive(Debug, Deserialize)]
#[allow(clippy::large_enum_variant)]
enum XMLAEMDataItem {
    #[serde(rename = "COMMENT")]
    Comment(String),
    #[serde(rename = "attitudeState")]
    AttitudeState(XMLAEMAttitudeState),
}

impl XMLAEMData {
    fn comments(&self) -> Vec<String> {
        self.items
            .iter()
            .filter_map(|item| {
                if let XMLAEMDataItem::Comment(s) = item {
                    Some(s.trim().to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    fn attitude_states(&self) -> Vec<&XMLAEMAttitudeState> {
        self.items
            .iter()
            .filter_map(|item| {
                if let XMLAEMDataItem::AttitudeState(a) = item {
                    Some(a)
                } else {
                    None
                }
            })
            .collect()
    }
}

/// A single `<attitudeState>` element, wrapping exactly one of the nine
/// per-type child elements (504.0-B-2 table 7-5).
#[derive(Debug, Deserialize)]
struct XMLAEMAttitudeState {
    #[serde(rename = "$value")]
    kind: XMLAEMAttitudeStateKind,
}

#[derive(Debug, Deserialize)]
#[allow(clippy::large_enum_variant)]
enum XMLAEMAttitudeStateKind {
    #[serde(rename = "quaternionEphemeris")]
    QuaternionEphemeris(XMLAEMQuaternionEphemeris),
    #[serde(rename = "quaternionDerivative")]
    QuaternionDerivative(XMLAEMQuaternionDerivative),
    #[serde(rename = "quaternionAngVel")]
    QuaternionAngVel(XMLAEMQuaternionAngVel),
    #[serde(rename = "eulerAngle")]
    EulerAngle(XMLAEMEulerAngle),
    #[serde(rename = "eulerAngleDerivative")]
    EulerAngleDerivative(XMLAEMEulerAngleDerivative),
    #[serde(rename = "eulerAngleAngVel")]
    EulerAngleAngVel(XMLAEMEulerAngleAngVel),
    #[serde(rename = "spin")]
    Spin(XMLAEMSpin),
    #[serde(rename = "spinNutation")]
    SpinNutation(XMLAEMSpinNutation),
    #[serde(rename = "spinNutationMom")]
    SpinNutationMom(XMLAEMSpinNutationMom),
}

#[derive(Debug, Deserialize)]
struct XMLAEMAngVel {
    #[serde(rename = "ANGVEL_X")]
    angvel_x: XMLValue,
    #[serde(rename = "ANGVEL_Y")]
    angvel_y: XMLValue,
    #[serde(rename = "ANGVEL_Z")]
    angvel_z: XMLValue,
}

#[derive(Debug, Deserialize)]
struct XMLAEMQuaternionEphemeris {
    #[serde(rename = "EPOCH")]
    epoch: String,
    quaternion: XMLQuaternion,
}

#[derive(Debug, Deserialize)]
struct XMLAEMQuaternionDerivative {
    #[serde(rename = "EPOCH")]
    epoch: String,
    quaternion: XMLQuaternion,
    #[serde(rename = "quaternionDot")]
    quaternion_dot: XMLQuaternionDot,
}

#[derive(Debug, Deserialize)]
struct XMLAEMQuaternionAngVel {
    #[serde(rename = "EPOCH")]
    epoch: String,
    quaternion: XMLQuaternion,
    #[serde(rename = "angVel")]
    ang_vel: XMLAEMAngVel,
}

#[derive(Debug, Deserialize)]
struct XMLAEMEulerAngle {
    #[serde(rename = "EPOCH")]
    epoch: String,
    #[serde(rename = "ANGLE_1")]
    angle_1: XMLValue,
    #[serde(rename = "ANGLE_2")]
    angle_2: XMLValue,
    #[serde(rename = "ANGLE_3")]
    angle_3: XMLValue,
}

#[derive(Debug, Deserialize)]
struct XMLAEMEulerAngleDerivative {
    #[serde(rename = "EPOCH")]
    epoch: String,
    #[serde(rename = "ANGLE_1")]
    angle_1: XMLValue,
    #[serde(rename = "ANGLE_2")]
    angle_2: XMLValue,
    #[serde(rename = "ANGLE_3")]
    angle_3: XMLValue,
    #[serde(rename = "ANGLE_1_DOT")]
    angle_1_dot: XMLValue,
    #[serde(rename = "ANGLE_2_DOT")]
    angle_2_dot: XMLValue,
    #[serde(rename = "ANGLE_3_DOT")]
    angle_3_dot: XMLValue,
}

#[derive(Debug, Deserialize)]
struct XMLAEMEulerAngleAngVel {
    #[serde(rename = "EPOCH")]
    epoch: String,
    #[serde(rename = "ANGLE_1")]
    angle_1: XMLValue,
    #[serde(rename = "ANGLE_2")]
    angle_2: XMLValue,
    #[serde(rename = "ANGLE_3")]
    angle_3: XMLValue,
    #[serde(rename = "angVel")]
    ang_vel: XMLAEMAngVel,
}

#[derive(Debug, Deserialize)]
struct XMLAEMSpin {
    #[serde(rename = "EPOCH")]
    epoch: String,
    #[serde(rename = "SPIN_ALPHA")]
    spin_alpha: XMLValue,
    #[serde(rename = "SPIN_DELTA")]
    spin_delta: XMLValue,
    #[serde(rename = "SPIN_ANGLE")]
    spin_angle: XMLValue,
    #[serde(rename = "SPIN_ANGLE_VEL")]
    spin_angle_vel: XMLValue,
}

#[derive(Debug, Deserialize)]
struct XMLAEMSpinNutation {
    #[serde(rename = "EPOCH")]
    epoch: String,
    #[serde(rename = "SPIN_ALPHA")]
    spin_alpha: XMLValue,
    #[serde(rename = "SPIN_DELTA")]
    spin_delta: XMLValue,
    #[serde(rename = "SPIN_ANGLE")]
    spin_angle: XMLValue,
    #[serde(rename = "SPIN_ANGLE_VEL")]
    spin_angle_vel: XMLValue,
    #[serde(rename = "NUTATION")]
    nutation: XMLValue,
    #[serde(rename = "NUTATION_PER")]
    nutation_per: XMLValue,
    #[serde(rename = "NUTATION_PHASE")]
    nutation_phase: XMLValue,
}

#[derive(Debug, Deserialize)]
struct XMLAEMSpinNutationMom {
    #[serde(rename = "EPOCH")]
    epoch: String,
    #[serde(rename = "SPIN_ALPHA")]
    spin_alpha: XMLValue,
    #[serde(rename = "SPIN_DELTA")]
    spin_delta: XMLValue,
    #[serde(rename = "SPIN_ANGLE")]
    spin_angle: XMLValue,
    #[serde(rename = "SPIN_ANGLE_VEL")]
    spin_angle_vel: XMLValue,
    #[serde(rename = "MOMENTUM_ALPHA")]
    momentum_alpha: XMLValue,
    #[serde(rename = "MOMENTUM_DELTA")]
    momentum_delta: XMLValue,
    #[serde(rename = "NUTATION_VEL")]
    nutation_vel: XMLValue,
}

// ============================================================================
// AEM XML Parser
// ============================================================================

fn xml_quaternion_from_element(q: &XMLQuaternion) -> Result<Quaternion, BraheError> {
    let v = Vector4::new(
        q.q1.parse_f64()?,
        q.q2.parse_f64()?,
        q.q3.parse_f64()?,
        q.qc.parse_f64()?,
    );
    Ok(Quaternion::from_vector(v, false))
}

/// Converts a single `<attitudeState>` element to an [`AEMAttitudeState`].
///
/// `euler_seq` supplies the rotation sequence for Euler angle types
/// (guaranteed present by [`AEMMetadata::validate`], which runs before any
/// attitude state is converted).
fn convert_xml_aem_attitude_state(
    state: &XMLAEMAttitudeState,
    euler_seq: Option<EulerAngleOrder>,
    time_system: &CCSDSTimeSystem,
) -> Result<AEMAttitudeState, BraheError> {
    match &state.kind {
        XMLAEMAttitudeStateKind::QuaternionEphemeris(q) => {
            let epoch = parse_ccsds_datetime(&q.epoch, time_system)?;
            let quaternion = xml_quaternion_from_element(&q.quaternion)?;
            Ok(AEMAttitudeState {
                epoch,
                data: AEMAttitudeData::Quaternion { quaternion },
            })
        }
        XMLAEMAttitudeStateKind::QuaternionDerivative(q) => {
            let epoch = parse_ccsds_datetime(&q.epoch, time_system)?;
            let quaternion = xml_quaternion_from_element(&q.quaternion)?;
            // derivative is stored scalar-first; wire order is scalar-last.
            let derivative = Vector4::new(
                q.quaternion_dot.qc_dot.parse_f64()?,
                q.quaternion_dot.q1_dot.parse_f64()?,
                q.quaternion_dot.q2_dot.parse_f64()?,
                q.quaternion_dot.q3_dot.parse_f64()?,
            );
            Ok(AEMAttitudeState {
                epoch,
                data: AEMAttitudeData::QuaternionDerivative {
                    quaternion,
                    derivative,
                },
            })
        }
        XMLAEMAttitudeStateKind::QuaternionAngVel(q) => {
            let epoch = parse_ccsds_datetime(&q.epoch, time_system)?;
            let quaternion = xml_quaternion_from_element(&q.quaternion)?;
            let angular_velocity = Vector3::new(
                q.ang_vel.angvel_x.parse_f64()? * DEG2RAD,
                q.ang_vel.angvel_y.parse_f64()? * DEG2RAD,
                q.ang_vel.angvel_z.parse_f64()? * DEG2RAD,
            );
            Ok(AEMAttitudeState {
                epoch,
                data: AEMAttitudeData::QuaternionAngVel {
                    quaternion,
                    angular_velocity,
                },
            })
        }
        XMLAEMAttitudeStateKind::EulerAngle(e) => {
            let epoch = parse_ccsds_datetime(&e.epoch, time_system)?;
            let seq = euler_seq.ok_or_else(|| ccsds_missing_field("AEM", "EULER_ROT_SEQ"))?;
            let angles = EulerAngle::new(
                seq,
                e.angle_1.parse_f64()?,
                e.angle_2.parse_f64()?,
                e.angle_3.parse_f64()?,
                AngleFormat::Degrees,
            );
            Ok(AEMAttitudeState {
                epoch,
                data: AEMAttitudeData::EulerAngle { angles },
            })
        }
        XMLAEMAttitudeStateKind::EulerAngleDerivative(e) => {
            let epoch = parse_ccsds_datetime(&e.epoch, time_system)?;
            let seq = euler_seq.ok_or_else(|| ccsds_missing_field("AEM", "EULER_ROT_SEQ"))?;
            let angles = EulerAngle::new(
                seq,
                e.angle_1.parse_f64()?,
                e.angle_2.parse_f64()?,
                e.angle_3.parse_f64()?,
                AngleFormat::Degrees,
            );
            let rates = Vector3::new(
                e.angle_1_dot.parse_f64()? * DEG2RAD,
                e.angle_2_dot.parse_f64()? * DEG2RAD,
                e.angle_3_dot.parse_f64()? * DEG2RAD,
            );
            Ok(AEMAttitudeState {
                epoch,
                data: AEMAttitudeData::EulerAngleDerivative { angles, rates },
            })
        }
        XMLAEMAttitudeStateKind::EulerAngleAngVel(e) => {
            let epoch = parse_ccsds_datetime(&e.epoch, time_system)?;
            let seq = euler_seq.ok_or_else(|| ccsds_missing_field("AEM", "EULER_ROT_SEQ"))?;
            let angles = EulerAngle::new(
                seq,
                e.angle_1.parse_f64()?,
                e.angle_2.parse_f64()?,
                e.angle_3.parse_f64()?,
                AngleFormat::Degrees,
            );
            let angular_velocity = Vector3::new(
                e.ang_vel.angvel_x.parse_f64()? * DEG2RAD,
                e.ang_vel.angvel_y.parse_f64()? * DEG2RAD,
                e.ang_vel.angvel_z.parse_f64()? * DEG2RAD,
            );
            Ok(AEMAttitudeState {
                epoch,
                data: AEMAttitudeData::EulerAngleAngVel {
                    angles,
                    angular_velocity,
                },
            })
        }
        XMLAEMAttitudeStateKind::Spin(s) => {
            let epoch = parse_ccsds_datetime(&s.epoch, time_system)?;
            Ok(AEMAttitudeState {
                epoch,
                data: AEMAttitudeData::Spin {
                    spin_alpha: s.spin_alpha.parse_f64()? * DEG2RAD,
                    spin_delta: s.spin_delta.parse_f64()? * DEG2RAD,
                    spin_angle: s.spin_angle.parse_f64()? * DEG2RAD,
                    spin_angle_vel: s.spin_angle_vel.parse_f64()? * DEG2RAD,
                },
            })
        }
        XMLAEMAttitudeStateKind::SpinNutation(s) => {
            let epoch = parse_ccsds_datetime(&s.epoch, time_system)?;
            Ok(AEMAttitudeState {
                epoch,
                data: AEMAttitudeData::SpinNutation {
                    spin_alpha: s.spin_alpha.parse_f64()? * DEG2RAD,
                    spin_delta: s.spin_delta.parse_f64()? * DEG2RAD,
                    spin_angle: s.spin_angle.parse_f64()? * DEG2RAD,
                    spin_angle_vel: s.spin_angle_vel.parse_f64()? * DEG2RAD,
                    nutation: s.nutation.parse_f64()? * DEG2RAD,
                    nutation_period: s.nutation_per.parse_f64()?,
                    nutation_phase: s.nutation_phase.parse_f64()? * DEG2RAD,
                },
            })
        }
        XMLAEMAttitudeStateKind::SpinNutationMom(s) => {
            let epoch = parse_ccsds_datetime(&s.epoch, time_system)?;
            Ok(AEMAttitudeState {
                epoch,
                data: AEMAttitudeData::SpinNutationMom {
                    spin_alpha: s.spin_alpha.parse_f64()? * DEG2RAD,
                    spin_delta: s.spin_delta.parse_f64()? * DEG2RAD,
                    spin_angle: s.spin_angle.parse_f64()? * DEG2RAD,
                    spin_angle_vel: s.spin_angle_vel.parse_f64()? * DEG2RAD,
                    momentum_alpha: s.momentum_alpha.parse_f64()? * DEG2RAD,
                    momentum_delta: s.momentum_delta.parse_f64()? * DEG2RAD,
                    nutation_vel: s.nutation_vel.parse_f64()? * DEG2RAD,
                },
            })
        }
    }
}

/// Parse an AEM message from XML format.
pub fn parse_aem_xml(content: &str) -> Result<AEM, BraheError> {
    let xml_aem: XMLAEM = quick_xml::de::from_str(content)
        .map_err(|e| ccsds_parse_error("AEM", &format!("XML parse error: {}", e)))?;

    let format_version = xml_aem
        .version
        .as_ref()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(2.0);
    if (format_version - 1.0).abs() < 1e-9 {
        return Err(ccsds_parse_error(
            "AEM",
            "version 1.0 (504.0-B-1) files are not supported; only version 2.0",
        ));
    }

    let creation_date_str = xml_aem
        .header
        .creation_date()
        .ok_or_else(|| ccsds_parse_error("AEM", "missing CREATION_DATE in header"))?;
    let originator = xml_aem
        .header
        .originator()
        .ok_or_else(|| ccsds_parse_error("AEM", "missing ORIGINATOR in header"))?
        .to_string();

    let header = AEMHeader {
        format_version,
        classification: xml_aem.header.classification(),
        creation_date: parse_ccsds_datetime(creation_date_str, &CCSDSTimeSystem::UTC)?,
        originator,
        message_id: xml_aem.header.message_id(),
        comments: xml_aem.header.comments(),
    };

    let mut segments = Vec::new();
    for seg in &xml_aem.body.segments {
        let meta = &seg.metadata;
        let time_system = CCSDSTimeSystem::parse(
            meta.find_str("TIME_SYSTEM")
                .ok_or_else(|| ccsds_parse_error("AEM", "missing TIME_SYSTEM in metadata"))?,
        )?;

        let attitude_type = AEMAttitudeType::parse(
            meta.find_str("ATTITUDE_TYPE")
                .ok_or_else(|| ccsds_parse_error("AEM", "missing ATTITUDE_TYPE in metadata"))?,
        )?;
        let euler_rot_seq = meta
            .find_str("EULER_ROT_SEQ")
            .map(parse_euler_rot_seq)
            .transpose()?;
        let ref_frame_a = ADMReferenceFrame::parse(
            meta.find_str("REF_FRAME_A")
                .ok_or_else(|| ccsds_parse_error("AEM", "missing REF_FRAME_A"))?,
        );
        let ref_frame_b = ADMReferenceFrame::parse(
            meta.find_str("REF_FRAME_B")
                .ok_or_else(|| ccsds_parse_error("AEM", "missing REF_FRAME_B"))?,
        );
        // CCSDS 504.0-B-2 Annex G-13 uses the literal tokens REF_FRAME_A/
        // REF_FRAME_B to mean "same as REF_FRAME_A/B"; resolve those
        // aliases against the frames just parsed.
        let angvel_frame = meta
            .find_str("ANGVEL_FRAME")
            .map(|raw| resolve_angvel_frame_token(raw, &ref_frame_a, &ref_frame_b));
        let interpolation_method = meta
            .find_str("INTERPOLATION_METHOD")
            .map(AEMInterpolationMethod::parse)
            .transpose()?;
        let interpolation_degree = meta.interpolation_degree();

        let metadata = AEMMetadata {
            object_name: meta
                .find_str("OBJECT_NAME")
                .ok_or_else(|| ccsds_parse_error("AEM", "missing OBJECT_NAME"))?
                .to_string(),
            object_id: meta
                .find_str("OBJECT_ID")
                .ok_or_else(|| ccsds_parse_error("AEM", "missing OBJECT_ID"))?
                .to_string(),
            center_name: meta.find_str("CENTER_NAME").map(|s| s.to_string()),
            ref_frame_a,
            ref_frame_b,
            time_system: time_system.clone(),
            start_time: parse_ccsds_datetime(
                meta.find_str("START_TIME")
                    .ok_or_else(|| ccsds_parse_error("AEM", "missing START_TIME"))?,
                &time_system,
            )?,
            stop_time: parse_ccsds_datetime(
                meta.find_str("STOP_TIME")
                    .ok_or_else(|| ccsds_parse_error("AEM", "missing STOP_TIME"))?,
                &time_system,
            )?,
            useable_start_time: meta
                .find_str("USEABLE_START_TIME")
                .map(|s| parse_ccsds_datetime(s, &time_system))
                .transpose()?,
            useable_stop_time: meta
                .find_str("USEABLE_STOP_TIME")
                .map(|s| parse_ccsds_datetime(s, &time_system))
                .transpose()?,
            attitude_type,
            euler_rot_seq,
            angvel_frame,
            interpolation_method,
            interpolation_degree,
            comments: meta.comments(),
        };
        metadata.validate()?;

        let mut segment = AEMSegment::new(metadata);
        segment.comments = seg.data.comments();

        for state_xml in seg.data.attitude_states() {
            let state = convert_xml_aem_attitude_state(state_xml, euler_rot_seq, &time_system)?;
            segment.push_state(state)?;
        }

        segments.push(segment);
    }

    Ok(AEM { header, segments })
}


// ============================================================================
// AEM XML Writer
// ============================================================================

/// Write an XML `<quaternion>` block (Q1/Q2/Q3/QC, scalar-last).
fn write_xml_quaternion(out: &mut String, q: &Quaternion, i_block: &str, i_elem: &str) {
    let v = q.to_vector(false);
    out.push_str(&format!("{}<quaternion>\n", i_block));
    out.push_str(&format!("{}<Q1>{}</Q1>\n", i_elem, v[0]));
    out.push_str(&format!("{}<Q2>{}</Q2>\n", i_elem, v[1]));
    out.push_str(&format!("{}<Q3>{}</Q3>\n", i_elem, v[2]));
    out.push_str(&format!("{}<QC>{}</QC>\n", i_elem, v[3]));
    out.push_str(&format!("{}</quaternion>\n", i_block));
}

/// Write an AEM message to XML format.
///
/// Requires at least one segment, each with at least one attitude state,
/// and metadata that passes [`crate::ccsds::aem::AEMMetadata::validate`];
/// see [`crate::ccsds::aem::AEM::validate_for_write`].
pub fn write_aem_xml(aem: &AEM) -> Result<String, BraheError> {
    aem.validate_for_write()?;

    let mut out = String::new();
    let i1 = "  ";
    let i2 = "    ";
    let i3 = "      ";
    let i4 = "        ";
    let i5 = "          ";
    let i6 = "            ";
    let i7 = "              ";

    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str(&format!(
        "<aem xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" id=\"CCSDS_AEM_VERS\" version=\"{:.1}\">\n",
        aem.header.format_version
    ));

    // Header
    out.push_str(&format!("{}<header>\n", i1));
    for c in &aem.header.comments {
        out.push_str(&format!(
            "{}<COMMENT>{}</COMMENT>\n",
            i2,
            escape_xml_text(c)
        ));
    }
    if let Some(ref cl) = aem.header.classification {
        out.push_str(&format!(
            "{}<CLASSIFICATION>{}</CLASSIFICATION>\n",
            i2,
            escape_xml_text(cl)
        ));
    }
    out.push_str(&format!(
        "{}<CREATION_DATE>{}</CREATION_DATE>\n",
        i2,
        format_ccsds_datetime_in(&aem.header.creation_date, &CCSDSTimeSystem::UTC)
    ));
    out.push_str(&format!(
        "{}<ORIGINATOR>{}</ORIGINATOR>\n",
        i2,
        escape_xml_text(&aem.header.originator)
    ));
    if let Some(ref mid) = aem.header.message_id {
        out.push_str(&format!(
            "{}<MESSAGE_ID>{}</MESSAGE_ID>\n",
            i2,
            escape_xml_text(mid)
        ));
    }
    out.push_str(&format!("{}</header>\n", i1));

    out.push_str(&format!("{}<body>\n", i1));

    for segment in &aem.segments {
        out.push_str(&format!("{}<segment>\n", i2));

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

        // Metadata
        out.push_str(&format!("{}<metadata>\n", i3));
        for c in &segment.metadata.comments {
            out.push_str(&format!(
                "{}<COMMENT>{}</COMMENT>\n",
                i4,
                escape_xml_text(c)
            ));
        }
        out.push_str(&format!(
            "{}<OBJECT_NAME>{}</OBJECT_NAME>\n",
            i4,
            escape_xml_text(&segment.metadata.object_name)
        ));
        out.push_str(&format!(
            "{}<OBJECT_ID>{}</OBJECT_ID>\n",
            i4,
            escape_xml_text(&segment.metadata.object_id)
        ));
        if let Some(ref center) = segment.metadata.center_name {
            out.push_str(&format!(
                "{}<CENTER_NAME>{}</CENTER_NAME>\n",
                i4,
                escape_xml_text(center)
            ));
        }
        out.push_str(&format!(
            "{}<REF_FRAME_A>{}</REF_FRAME_A>\n",
            i4,
            escape_xml_text(&format!("{}", segment.metadata.ref_frame_a))
        ));
        out.push_str(&format!(
            "{}<REF_FRAME_B>{}</REF_FRAME_B>\n",
            i4,
            escape_xml_text(&format!("{}", segment.metadata.ref_frame_b))
        ));
        out.push_str(&format!(
            "{}<TIME_SYSTEM>{}</TIME_SYSTEM>\n",
            i4, segment.metadata.time_system
        ));
        out.push_str(&format!(
            "{}<START_TIME>{}</START_TIME>\n",
            i4,
            format_ccsds_datetime(&epoch_for_write(&segment.metadata.start_time))
        ));
        if let Some(ref t) = segment.metadata.useable_start_time {
            out.push_str(&format!(
                "{}<USEABLE_START_TIME>{}</USEABLE_START_TIME>\n",
                i4,
                format_ccsds_datetime(&epoch_for_write(t))
            ));
        }
        if let Some(ref t) = segment.metadata.useable_stop_time {
            out.push_str(&format!(
                "{}<USEABLE_STOP_TIME>{}</USEABLE_STOP_TIME>\n",
                i4,
                format_ccsds_datetime(&epoch_for_write(t))
            ));
        }
        out.push_str(&format!(
            "{}<STOP_TIME>{}</STOP_TIME>\n",
            i4,
            format_ccsds_datetime(&epoch_for_write(&segment.metadata.stop_time))
        ));
        out.push_str(&format!(
            "{}<ATTITUDE_TYPE>{}</ATTITUDE_TYPE>\n",
            i4, segment.metadata.attitude_type
        ));
        if let Some(seq) = segment.metadata.euler_rot_seq {
            out.push_str(&format!(
                "{}<EULER_ROT_SEQ>{}</EULER_ROT_SEQ>\n",
                i4,
                format_euler_rot_seq(seq)
            ));
        }
        if let Some(ref frame) = segment.metadata.angvel_frame {
            out.push_str(&format!(
                "{}<ANGVEL_FRAME>{}</ANGVEL_FRAME>\n",
                i4,
                escape_xml_text(&format!("{}", frame))
            ));
        }
        if let Some(method) = segment.metadata.interpolation_method {
            out.push_str(&format!(
                "{}<INTERPOLATION_METHOD>{}</INTERPOLATION_METHOD>\n",
                i4, method
            ));
        }
        if let Some(degree) = segment.metadata.interpolation_degree {
            out.push_str(&format!(
                "{}<INTERPOLATION_DEGREE>{}</INTERPOLATION_DEGREE>\n",
                i4, degree
            ));
        }
        out.push_str(&format!("{}</metadata>\n", i3));

        // Data
        out.push_str(&format!("{}<data>\n", i3));
        for c in &segment.comments {
            out.push_str(&format!(
                "{}<COMMENT>{}</COMMENT>\n",
                i4,
                escape_xml_text(c)
            ));
        }

        for state in &segment.states {
            let epoch_str = format_ccsds_datetime(&epoch_for_write(&state.epoch));
            out.push_str(&format!("{}<attitudeState>\n", i4));
            match &state.data {
                AEMAttitudeData::Quaternion { quaternion } => {
                    out.push_str(&format!("{}<quaternionEphemeris>\n", i5));
                    out.push_str(&format!("{}<EPOCH>{}</EPOCH>\n", i6, epoch_str));
                    write_xml_quaternion(&mut out, quaternion, i6, i7);
                    out.push_str(&format!("{}</quaternionEphemeris>\n", i5));
                }
                AEMAttitudeData::QuaternionDerivative {
                    quaternion,
                    derivative,
                } => {
                    out.push_str(&format!("{}<quaternionDerivative>\n", i5));
                    out.push_str(&format!("{}<EPOCH>{}</EPOCH>\n", i6, epoch_str));
                    write_xml_quaternion(&mut out, quaternion, i6, i7);
                    // derivative is stored scalar-first; wire order is scalar-last.
                    out.push_str(&format!("{}<quaternionDot>\n", i6));
                    out.push_str(&format!("{}<Q1_DOT>{}</Q1_DOT>\n", i7, derivative[1]));
                    out.push_str(&format!("{}<Q2_DOT>{}</Q2_DOT>\n", i7, derivative[2]));
                    out.push_str(&format!("{}<Q3_DOT>{}</Q3_DOT>\n", i7, derivative[3]));
                    out.push_str(&format!("{}<QC_DOT>{}</QC_DOT>\n", i7, derivative[0]));
                    out.push_str(&format!("{}</quaternionDot>\n", i6));
                    out.push_str(&format!("{}</quaternionDerivative>\n", i5));
                }
                AEMAttitudeData::QuaternionAngVel {
                    quaternion,
                    angular_velocity,
                } => {
                    out.push_str(&format!("{}<quaternionAngVel>\n", i5));
                    out.push_str(&format!("{}<EPOCH>{}</EPOCH>\n", i6, epoch_str));
                    write_xml_quaternion(&mut out, quaternion, i6, i7);
                    out.push_str(&format!("{}<angVel>\n", i6));
                    out.push_str(&format!(
                        "{}<ANGVEL_X>{}</ANGVEL_X>\n",
                        i7,
                        angular_velocity[0] * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<ANGVEL_Y>{}</ANGVEL_Y>\n",
                        i7,
                        angular_velocity[1] * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<ANGVEL_Z>{}</ANGVEL_Z>\n",
                        i7,
                        angular_velocity[2] * RAD2DEG
                    ));
                    out.push_str(&format!("{}</angVel>\n", i6));
                    out.push_str(&format!("{}</quaternionAngVel>\n", i5));
                }
                AEMAttitudeData::EulerAngle { angles } => {
                    out.push_str(&format!("{}<eulerAngle>\n", i5));
                    out.push_str(&format!("{}<EPOCH>{}</EPOCH>\n", i6, epoch_str));
                    out.push_str(&format!(
                        "{}<ANGLE_1>{}</ANGLE_1>\n",
                        i6,
                        angles.phi * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<ANGLE_2>{}</ANGLE_2>\n",
                        i6,
                        angles.theta * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<ANGLE_3>{}</ANGLE_3>\n",
                        i6,
                        angles.psi * RAD2DEG
                    ));
                    out.push_str(&format!("{}</eulerAngle>\n", i5));
                }
                AEMAttitudeData::EulerAngleDerivative { angles, rates } => {
                    out.push_str(&format!("{}<eulerAngleDerivative>\n", i5));
                    out.push_str(&format!("{}<EPOCH>{}</EPOCH>\n", i6, epoch_str));
                    out.push_str(&format!(
                        "{}<ANGLE_1>{}</ANGLE_1>\n",
                        i6,
                        angles.phi * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<ANGLE_2>{}</ANGLE_2>\n",
                        i6,
                        angles.theta * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<ANGLE_3>{}</ANGLE_3>\n",
                        i6,
                        angles.psi * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<ANGLE_1_DOT>{}</ANGLE_1_DOT>\n",
                        i6,
                        rates[0] * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<ANGLE_2_DOT>{}</ANGLE_2_DOT>\n",
                        i6,
                        rates[1] * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<ANGLE_3_DOT>{}</ANGLE_3_DOT>\n",
                        i6,
                        rates[2] * RAD2DEG
                    ));
                    out.push_str(&format!("{}</eulerAngleDerivative>\n", i5));
                }
                AEMAttitudeData::EulerAngleAngVel {
                    angles,
                    angular_velocity,
                } => {
                    out.push_str(&format!("{}<eulerAngleAngVel>\n", i5));
                    out.push_str(&format!("{}<EPOCH>{}</EPOCH>\n", i6, epoch_str));
                    out.push_str(&format!(
                        "{}<ANGLE_1>{}</ANGLE_1>\n",
                        i6,
                        angles.phi * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<ANGLE_2>{}</ANGLE_2>\n",
                        i6,
                        angles.theta * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<ANGLE_3>{}</ANGLE_3>\n",
                        i6,
                        angles.psi * RAD2DEG
                    ));
                    out.push_str(&format!("{}<angVel>\n", i6));
                    out.push_str(&format!(
                        "{}<ANGVEL_X>{}</ANGVEL_X>\n",
                        i7,
                        angular_velocity[0] * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<ANGVEL_Y>{}</ANGVEL_Y>\n",
                        i7,
                        angular_velocity[1] * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<ANGVEL_Z>{}</ANGVEL_Z>\n",
                        i7,
                        angular_velocity[2] * RAD2DEG
                    ));
                    out.push_str(&format!("{}</angVel>\n", i6));
                    out.push_str(&format!("{}</eulerAngleAngVel>\n", i5));
                }
                AEMAttitudeData::Spin {
                    spin_alpha,
                    spin_delta,
                    spin_angle,
                    spin_angle_vel,
                } => {
                    out.push_str(&format!("{}<spin>\n", i5));
                    out.push_str(&format!("{}<EPOCH>{}</EPOCH>\n", i6, epoch_str));
                    out.push_str(&format!(
                        "{}<SPIN_ALPHA>{}</SPIN_ALPHA>\n",
                        i6,
                        spin_alpha * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<SPIN_DELTA>{}</SPIN_DELTA>\n",
                        i6,
                        spin_delta * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<SPIN_ANGLE>{}</SPIN_ANGLE>\n",
                        i6,
                        spin_angle * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<SPIN_ANGLE_VEL>{}</SPIN_ANGLE_VEL>\n",
                        i6,
                        spin_angle_vel * RAD2DEG
                    ));
                    out.push_str(&format!("{}</spin>\n", i5));
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
                    out.push_str(&format!("{}<spinNutation>\n", i5));
                    out.push_str(&format!("{}<EPOCH>{}</EPOCH>\n", i6, epoch_str));
                    out.push_str(&format!(
                        "{}<SPIN_ALPHA>{}</SPIN_ALPHA>\n",
                        i6,
                        spin_alpha * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<SPIN_DELTA>{}</SPIN_DELTA>\n",
                        i6,
                        spin_delta * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<SPIN_ANGLE>{}</SPIN_ANGLE>\n",
                        i6,
                        spin_angle * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<SPIN_ANGLE_VEL>{}</SPIN_ANGLE_VEL>\n",
                        i6,
                        spin_angle_vel * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<NUTATION>{}</NUTATION>\n",
                        i6,
                        nutation * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<NUTATION_PER>{}</NUTATION_PER>\n",
                        i6, nutation_period
                    ));
                    out.push_str(&format!(
                        "{}<NUTATION_PHASE>{}</NUTATION_PHASE>\n",
                        i6,
                        nutation_phase * RAD2DEG
                    ));
                    out.push_str(&format!("{}</spinNutation>\n", i5));
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
                    out.push_str(&format!("{}<spinNutationMom>\n", i5));
                    out.push_str(&format!("{}<EPOCH>{}</EPOCH>\n", i6, epoch_str));
                    out.push_str(&format!(
                        "{}<SPIN_ALPHA>{}</SPIN_ALPHA>\n",
                        i6,
                        spin_alpha * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<SPIN_DELTA>{}</SPIN_DELTA>\n",
                        i6,
                        spin_delta * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<SPIN_ANGLE>{}</SPIN_ANGLE>\n",
                        i6,
                        spin_angle * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<SPIN_ANGLE_VEL>{}</SPIN_ANGLE_VEL>\n",
                        i6,
                        spin_angle_vel * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<MOMENTUM_ALPHA>{}</MOMENTUM_ALPHA>\n",
                        i6,
                        momentum_alpha * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<MOMENTUM_DELTA>{}</MOMENTUM_DELTA>\n",
                        i6,
                        momentum_delta * RAD2DEG
                    ));
                    out.push_str(&format!(
                        "{}<NUTATION_VEL>{}</NUTATION_VEL>\n",
                        i6,
                        nutation_vel * RAD2DEG
                    ));
                    out.push_str(&format!("{}</spinNutationMom>\n", i5));
                }
            }
            out.push_str(&format!("{}</attitudeState>\n", i4));
        }

        out.push_str(&format!("{}</data>\n", i3));
        out.push_str(&format!("{}</segment>\n", i2));
    }

    out.push_str(&format!("{}</body>\n", i1));
    out.push_str("</aem>\n");

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::parallel;

    #[test]
    #[parallel]
    fn test_parse_aem_xml_v1_version_rejected() {
        let content = std::fs::read_to_string("test_assets/ccsds/aem/AEMExampleG11.xml")
            .unwrap()
            .replace(
                "id=\"CCSDS_AEM_VERS\" version=\"2.0\"",
                "id=\"CCSDS_AEM_VERS\" version=\"1.0\"",
            );
        let result = parse_aem_xml(&content);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("version 1.0"));
        assert!(err.contains("504.0-B-1"));
    }

    #[test]
    #[parallel]
    fn test_parse_aem_xml_angvel_frame_literal_ref_frame_a_resolves() {
        // CCSDS 504.0-B-2 Annex G-13 uses the literal token REF_FRAME_A to
        // mean "same as REF_FRAME_A", not a SANA registry token spelled
        // "REF_FRAME_A".
        use crate::time::{Epoch, TimeSystem};

        let ref_frame_a = ADMReferenceFrame::parse("EME2000");
        let ref_frame_b = ADMReferenceFrame::parse("SC_BODY_1");
        let t0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;

        let metadata = AEMMetadata::new(
            "SAT1",
            "2024-001A",
            ref_frame_a.clone(),
            ref_frame_b.clone(),
            CCSDSTimeSystem::UTC,
            t0,
            t1,
            AEMAttitudeType::QuaternionAngVel,
        )
        .with_angvel_frame(ref_frame_a.clone());

        let mut segment = AEMSegment::new(metadata);
        segment
            .push_state(AEMAttitudeState {
                epoch: t0,
                data: AEMAttitudeData::QuaternionAngVel {
                    quaternion: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                    angular_velocity: Vector3::new(0.001, 0.002, 0.003),
                },
            })
            .unwrap();

        let mut aem = AEM::new("BRAHE");
        aem.push_segment(segment);

        let xml = super::super::writer::write_aem_xml(&aem).unwrap();
        assert!(xml.contains("<ANGVEL_FRAME>EME2000</ANGVEL_FRAME>"));
        let xml_literal = xml.replace(
            "<ANGVEL_FRAME>EME2000</ANGVEL_FRAME>",
            "<ANGVEL_FRAME>REF_FRAME_A</ANGVEL_FRAME>",
        );

        let parsed = parse_aem_xml(&xml_literal).unwrap();
        let parsed_metadata = &parsed.segments[0].metadata;
        assert_eq!(
            parsed_metadata.angvel_frame,
            Some(parsed_metadata.ref_frame_a.clone())
        );
    }
}
