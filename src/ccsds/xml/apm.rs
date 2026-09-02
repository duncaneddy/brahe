/*!
 * XML reader and writer for the Attitude Parameter Message (APM).
 *
 * Reference: CCSDS 504.0-B-2 (Attitude Data Messages), section 3
 */

use nalgebra::{Vector3, Vector4};
use serde::Deserialize;

use crate::attitude::attitude_types::{EulerAngle, Quaternion};
use crate::ccsds::apm::{
    APM, APMAngularVelocity, APMEulerState, APMHeader, APMInertia, APMManeuver, APMMetadata,
    APMNutation, APMQuaternionState, APMSpin,
};
use crate::ccsds::common::{
    CCSDSTimeSystem, format_ccsds_datetime_in, format_euler_rot_seq, parse_ccsds_datetime,
    parse_euler_rot_seq,
};
use crate::ccsds::error::{ccsds_missing_field, ccsds_parse_error};
use crate::ccsds::frames::ADMReferenceFrame;
use crate::ccsds::xml::common::{XMLHeader, XMLValue, escape_xml_text, validate_xml_characters};
use crate::constants::{AngleFormat, DEG2RAD, RAD2DEG};
use crate::utils::errors::BraheError;

// ============================================================================
// Intermediate XML structs for APM
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(rename = "apm")]
#[allow(clippy::upper_case_acronyms)]
struct XMLAPM {
    #[serde(rename = "@version")]
    version: Option<String>,
    header: XMLHeader,
    body: XMLAPMBody,
}

#[derive(Debug, Deserialize)]
struct XMLAPMBody {
    segment: XMLAPMSegment,
}

#[derive(Debug, Deserialize)]
struct XMLAPMSegment {
    metadata: XMLAPMMetadata,
    data: XMLAPMData,
}

#[derive(Debug, Deserialize)]
struct XMLAPMMetadata {
    #[serde(rename = "$value")]
    items: Vec<XMLAPMMetadataItem>,
}

#[derive(Debug, Deserialize)]
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
enum XMLAPMMetadataItem {
    OBJECT_NAME(String),
    OBJECT_ID(String),
    CENTER_NAME(String),
    TIME_SYSTEM(String),
    COMMENT(String),
}

impl XMLAPMMetadata {
    fn find_str(&self, variant: &str) -> Option<&str> {
        self.items.iter().find_map(|item| match item {
            XMLAPMMetadataItem::OBJECT_NAME(s) if variant == "OBJECT_NAME" => Some(s.as_str()),
            XMLAPMMetadataItem::OBJECT_ID(s) if variant == "OBJECT_ID" => Some(s.as_str()),
            XMLAPMMetadataItem::CENTER_NAME(s) if variant == "CENTER_NAME" => Some(s.as_str()),
            XMLAPMMetadataItem::TIME_SYSTEM(s) if variant == "TIME_SYSTEM" => Some(s.as_str()),
            _ => None,
        })
    }

    fn comments(&self) -> Vec<String> {
        self.items
            .iter()
            .filter_map(|item| {
                if let XMLAPMMetadataItem::COMMENT(s) = item {
                    Some(s.trim().to_string())
                } else {
                    None
                }
            })
            .collect()
    }
}

/// APM data block containing the epoch and the six repeatable logical
/// blocks, in any mix and order.
///
/// Uses `$value` to capture all child elements as a flat sequence, since
/// quick-xml cannot handle interleaved repeated block elements (e.g.
/// multiple `quaternionState` blocks interspersed with `inertia` or
/// `maneuverParameters`) any other way.
#[derive(Debug, Deserialize)]
struct XMLAPMData {
    #[serde(rename = "$value", default)]
    items: Vec<XMLAPMDataItem>,
}

#[derive(Debug, Deserialize)]
#[allow(clippy::large_enum_variant)]
enum XMLAPMDataItem {
    #[serde(rename = "COMMENT")]
    Comment(String),
    #[serde(rename = "EPOCH")]
    Epoch(String),
    #[serde(rename = "quaternionState")]
    QuaternionState(XMLAPMQuaternionState),
    #[serde(rename = "eulerAngleState")]
    EulerAngleState(XMLAPMEulerState),
    #[serde(rename = "angularVelocity")]
    AngularVelocity(XMLAPMAngularVelocity),
    #[serde(rename = "spin")]
    Spin(XMLAPMSpin),
    #[serde(rename = "inertia")]
    Inertia(XMLAPMInertia),
    #[serde(rename = "maneuverParameters")]
    ManeuverParameters(XMLAPMManeuver),
}

impl XMLAPMData {
    fn comments(&self) -> Vec<String> {
        self.items
            .iter()
            .filter_map(|item| {
                if let XMLAPMDataItem::Comment(s) = item {
                    Some(s.trim().to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    fn epoch(&self) -> Option<&str> {
        self.items.iter().find_map(|item| {
            if let XMLAPMDataItem::Epoch(s) = item {
                Some(s.as_str())
            } else {
                None
            }
        })
    }

    fn quaternion_states(&self) -> Vec<&XMLAPMQuaternionState> {
        self.items
            .iter()
            .filter_map(|item| {
                if let XMLAPMDataItem::QuaternionState(q) = item {
                    Some(q)
                } else {
                    None
                }
            })
            .collect()
    }

    fn euler_states(&self) -> Vec<&XMLAPMEulerState> {
        self.items
            .iter()
            .filter_map(|item| {
                if let XMLAPMDataItem::EulerAngleState(e) = item {
                    Some(e)
                } else {
                    None
                }
            })
            .collect()
    }

    fn angular_velocities(&self) -> Vec<&XMLAPMAngularVelocity> {
        self.items
            .iter()
            .filter_map(|item| {
                if let XMLAPMDataItem::AngularVelocity(a) = item {
                    Some(a)
                } else {
                    None
                }
            })
            .collect()
    }

    fn spins(&self) -> Vec<&XMLAPMSpin> {
        self.items
            .iter()
            .filter_map(|item| {
                if let XMLAPMDataItem::Spin(s) = item {
                    Some(s)
                } else {
                    None
                }
            })
            .collect()
    }

    fn inertias(&self) -> Vec<&XMLAPMInertia> {
        self.items
            .iter()
            .filter_map(|item| {
                if let XMLAPMDataItem::Inertia(i) = item {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    fn maneuvers(&self) -> Vec<&XMLAPMManeuver> {
        self.items
            .iter()
            .filter_map(|item| {
                if let XMLAPMDataItem::ManeuverParameters(m) = item {
                    Some(m)
                } else {
                    None
                }
            })
            .collect()
    }
}

#[derive(Debug, Deserialize)]
struct XMLAPMQuaternion {
    #[serde(rename = "Q1")]
    q1: XMLValue,
    #[serde(rename = "Q2")]
    q2: XMLValue,
    #[serde(rename = "Q3")]
    q3: XMLValue,
    #[serde(rename = "QC")]
    qc: XMLValue,
}

#[derive(Debug, Deserialize)]
struct XMLAPMQuaternionDot {
    #[serde(rename = "Q1_DOT")]
    q1_dot: XMLValue,
    #[serde(rename = "Q2_DOT")]
    q2_dot: XMLValue,
    #[serde(rename = "Q3_DOT")]
    q3_dot: XMLValue,
    #[serde(rename = "QC_DOT")]
    qc_dot: XMLValue,
}

#[derive(Debug, Deserialize)]
struct XMLAPMQuaternionState {
    #[serde(rename = "COMMENT", default)]
    comments: Vec<String>,
    #[serde(rename = "REF_FRAME_A")]
    ref_frame_a: String,
    #[serde(rename = "REF_FRAME_B")]
    ref_frame_b: String,
    quaternion: XMLAPMQuaternion,
    #[serde(rename = "quaternionDot", default)]
    quaternion_dot: Option<XMLAPMQuaternionDot>,
}

#[derive(Debug, Deserialize)]
struct XMLAPMEulerState {
    #[serde(rename = "COMMENT", default)]
    comments: Vec<String>,
    #[serde(rename = "REF_FRAME_A")]
    ref_frame_a: String,
    #[serde(rename = "REF_FRAME_B")]
    ref_frame_b: String,
    #[serde(rename = "EULER_ROT_SEQ")]
    euler_rot_seq: String,
    #[serde(rename = "ANGLE_1")]
    angle_1: XMLValue,
    #[serde(rename = "ANGLE_2")]
    angle_2: XMLValue,
    #[serde(rename = "ANGLE_3")]
    angle_3: XMLValue,
    #[serde(rename = "ANGLE_1_DOT")]
    angle_1_dot: Option<XMLValue>,
    #[serde(rename = "ANGLE_2_DOT")]
    angle_2_dot: Option<XMLValue>,
    #[serde(rename = "ANGLE_3_DOT")]
    angle_3_dot: Option<XMLValue>,
}

#[derive(Debug, Deserialize)]
struct XMLAPMAngularVelocity {
    #[serde(rename = "COMMENT", default)]
    comments: Vec<String>,
    #[serde(rename = "REF_FRAME_A")]
    ref_frame_a: String,
    #[serde(rename = "REF_FRAME_B")]
    ref_frame_b: String,
    #[serde(rename = "ANGVEL_FRAME")]
    angvel_frame: String,
    #[serde(rename = "ANGVEL_X")]
    angvel_x: XMLValue,
    #[serde(rename = "ANGVEL_Y")]
    angvel_y: XMLValue,
    #[serde(rename = "ANGVEL_Z")]
    angvel_z: XMLValue,
}

#[derive(Debug, Deserialize)]
struct XMLAPMSpin {
    #[serde(rename = "COMMENT", default)]
    comments: Vec<String>,
    #[serde(rename = "REF_FRAME_A")]
    ref_frame_a: String,
    #[serde(rename = "REF_FRAME_B")]
    ref_frame_b: String,
    #[serde(rename = "SPIN_ALPHA")]
    spin_alpha: XMLValue,
    #[serde(rename = "SPIN_DELTA")]
    spin_delta: XMLValue,
    #[serde(rename = "SPIN_ANGLE")]
    spin_angle: XMLValue,
    #[serde(rename = "SPIN_ANGLE_VEL")]
    spin_angle_vel: XMLValue,
    #[serde(rename = "NUTATION")]
    nutation: Option<XMLValue>,
    #[serde(rename = "NUTATION_PER")]
    nutation_per: Option<XMLValue>,
    #[serde(rename = "NUTATION_PHASE")]
    nutation_phase: Option<XMLValue>,
    #[serde(rename = "MOMENTUM_ALPHA")]
    momentum_alpha: Option<XMLValue>,
    #[serde(rename = "MOMENTUM_DELTA")]
    momentum_delta: Option<XMLValue>,
    #[serde(rename = "NUTATION_VEL")]
    nutation_vel: Option<XMLValue>,
}

#[derive(Debug, Deserialize)]
struct XMLAPMInertia {
    #[serde(rename = "COMMENT", default)]
    comments: Vec<String>,
    #[serde(rename = "INERTIA_REF_FRAME")]
    inertia_ref_frame: String,
    #[serde(rename = "IXX")]
    ixx: XMLValue,
    #[serde(rename = "IYY")]
    iyy: XMLValue,
    #[serde(rename = "IZZ")]
    izz: XMLValue,
    #[serde(rename = "IXY")]
    ixy: XMLValue,
    #[serde(rename = "IXZ")]
    ixz: XMLValue,
    #[serde(rename = "IYZ")]
    iyz: XMLValue,
}

#[derive(Debug, Deserialize)]
struct XMLAPMManeuver {
    #[serde(rename = "COMMENT", default)]
    comments: Vec<String>,
    #[serde(rename = "MAN_EPOCH_START")]
    epoch_start: String,
    #[serde(rename = "MAN_DURATION")]
    duration: XMLValue,
    #[serde(rename = "MAN_REF_FRAME")]
    ref_frame: String,
    #[serde(rename = "MAN_TOR_X")]
    tor_x: XMLValue,
    #[serde(rename = "MAN_TOR_Y")]
    tor_y: XMLValue,
    #[serde(rename = "MAN_TOR_Z")]
    tor_z: XMLValue,
    #[serde(rename = "MAN_DELTA_MASS")]
    delta_mass: Option<XMLValue>,
}

// ============================================================================
// APM XML Parser
// ============================================================================

/// Parse an APM message from XML format.
pub fn parse_apm_xml(content: &str) -> Result<APM, BraheError> {
    let xml_apm: XMLAPM = quick_xml::de::from_str(content)
        .map_err(|e| ccsds_parse_error("APM", &format!("XML parse error: {}", e)))?;

    let format_version = xml_apm
        .version
        .as_ref()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(2.0);
    if (format_version - 1.0).abs() < 1e-9 {
        return Err(ccsds_parse_error(
            "APM",
            "version 1.0 (504.0-B-1) files are not supported; only version 2.0",
        ));
    }

    let creation_date_str = xml_apm
        .header
        .creation_date()
        .ok_or_else(|| ccsds_parse_error("APM", "missing CREATION_DATE in header"))?;
    let originator = xml_apm
        .header
        .originator()
        .ok_or_else(|| ccsds_parse_error("APM", "missing ORIGINATOR in header"))?
        .to_string();

    let header = APMHeader {
        format_version,
        classification: xml_apm.header.classification(),
        creation_date: parse_ccsds_datetime(creation_date_str, &CCSDSTimeSystem::UTC)?,
        originator,
        message_id: xml_apm.header.message_id(),
        comments: xml_apm.header.comments(),
    };

    let meta = &xml_apm.body.segment.metadata;
    let time_system = CCSDSTimeSystem::parse(
        meta.find_str("TIME_SYSTEM")
            .ok_or_else(|| ccsds_parse_error("APM", "missing TIME_SYSTEM in metadata"))?,
    )?;

    let metadata = APMMetadata {
        object_name: meta
            .find_str("OBJECT_NAME")
            .ok_or_else(|| ccsds_parse_error("APM", "missing OBJECT_NAME"))?
            .to_string(),
        object_id: meta
            .find_str("OBJECT_ID")
            .ok_or_else(|| ccsds_parse_error("APM", "missing OBJECT_ID"))?
            .to_string(),
        center_name: meta.find_str("CENTER_NAME").map(|s| s.to_string()),
        time_system: time_system.clone(),
        comments: meta.comments(),
    };

    let data = &xml_apm.body.segment.data;
    let epoch_str = data
        .epoch()
        .ok_or_else(|| ccsds_missing_field("APM", "EPOCH"))?;
    let epoch = parse_ccsds_datetime(epoch_str, &time_system)?;

    let mut quaternion_states = Vec::new();
    for q in data.quaternion_states() {
        let ref_frame_a = ADMReferenceFrame::parse(&q.ref_frame_a);
        let ref_frame_b = ADMReferenceFrame::parse(&q.ref_frame_b);
        let q1 = q.quaternion.q1.parse_f64()?;
        let q2 = q.quaternion.q2.parse_f64()?;
        let q3 = q.quaternion.q3.parse_f64()?;
        let qc = q.quaternion.qc.parse_f64()?;
        let quaternion = Quaternion::from_vector(Vector4::new(q1, q2, q3, qc), false);

        let mut state = APMQuaternionState::new(ref_frame_a, ref_frame_b, quaternion);
        if let Some(ref qd) = q.quaternion_dot {
            let d1 = qd.q1_dot.parse_f64()?;
            let d2 = qd.q2_dot.parse_f64()?;
            let d3 = qd.q3_dot.parse_f64()?;
            let dc = qd.qc_dot.parse_f64()?;
            state = state.with_derivative(Vector4::new(dc, d1, d2, d3));
        }
        state.comments = q.comments.clone();
        quaternion_states.push(state);
    }

    let mut euler_states = Vec::new();
    for e in data.euler_states() {
        let ref_frame_a = ADMReferenceFrame::parse(&e.ref_frame_a);
        let ref_frame_b = ADMReferenceFrame::parse(&e.ref_frame_b);
        let seq = parse_euler_rot_seq(&e.euler_rot_seq)?;
        let a1 = e.angle_1.parse_f64()?;
        let a2 = e.angle_2.parse_f64()?;
        let a3 = e.angle_3.parse_f64()?;
        let angles = EulerAngle::new(seq, a1, a2, a3, AngleFormat::Degrees);

        let mut state = APMEulerState::new(ref_frame_a, ref_frame_b, angles);
        let rates = match (&e.angle_1_dot, &e.angle_2_dot, &e.angle_3_dot) {
            (Some(a), Some(b), Some(c)) => Some(Vector3::new(
                a.parse_f64()? * DEG2RAD,
                b.parse_f64()? * DEG2RAD,
                c.parse_f64()? * DEG2RAD,
            )),
            (None, None, None) => None,
            _ => {
                return Err(ccsds_parse_error(
                    "APM",
                    "incomplete Euler angle rates: ANGLE_1_DOT/ANGLE_2_DOT/ANGLE_3_DOT must all be present or all absent",
                ));
            }
        };
        if let Some(r) = rates {
            state = state.with_rates(r);
        }
        state.comments = e.comments.clone();
        euler_states.push(state);
    }

    let mut angular_velocities = Vec::new();
    for av in data.angular_velocities() {
        let ref_frame_a = ADMReferenceFrame::parse(&av.ref_frame_a);
        let ref_frame_b = ADMReferenceFrame::parse(&av.ref_frame_b);
        let angvel_frame = ADMReferenceFrame::parse(&av.angvel_frame);
        let vel = Vector3::new(
            av.angvel_x.parse_f64()? * DEG2RAD,
            av.angvel_y.parse_f64()? * DEG2RAD,
            av.angvel_z.parse_f64()? * DEG2RAD,
        );
        let mut block = APMAngularVelocity::new(ref_frame_a, ref_frame_b, angvel_frame, vel);
        block.comments = av.comments.clone();
        angular_velocities.push(block);
    }

    let mut spins = Vec::new();
    for s in data.spins() {
        let ref_frame_a = ADMReferenceFrame::parse(&s.ref_frame_a);
        let ref_frame_b = ADMReferenceFrame::parse(&s.ref_frame_b);
        let alpha = s.spin_alpha.parse_f64()?;
        let delta = s.spin_delta.parse_f64()?;
        let angle = s.spin_angle.parse_f64()?;
        let angle_vel = s.spin_angle_vel.parse_f64()?;
        let mut spin = APMSpin::new(
            ref_frame_a,
            ref_frame_b,
            alpha,
            delta,
            angle,
            angle_vel,
            AngleFormat::Degrees,
        );

        let nut_triple = (&s.nutation, &s.nutation_per, &s.nutation_phase);
        let mom_triple = (&s.momentum_alpha, &s.momentum_delta, &s.nutation_vel);
        let nut_complete = matches!(nut_triple, (Some(_), Some(_), Some(_)));
        let nut_partial = !nut_complete
            && (nut_triple.0.is_some() || nut_triple.1.is_some() || nut_triple.2.is_some());
        let mom_complete = matches!(mom_triple, (Some(_), Some(_), Some(_)));
        let mom_partial = !mom_complete
            && (mom_triple.0.is_some() || mom_triple.1.is_some() || mom_triple.2.is_some());

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
                nut_triple.0.as_ref().unwrap().parse_f64()?,
                nut_triple.1.as_ref().unwrap().parse_f64()?,
                nut_triple.2.as_ref().unwrap().parse_f64()?,
                AngleFormat::Degrees,
            );
        } else if mom_complete {
            spin = spin.with_nutation_momentum(
                mom_triple.0.as_ref().unwrap().parse_f64()?,
                mom_triple.1.as_ref().unwrap().parse_f64()?,
                mom_triple.2.as_ref().unwrap().parse_f64()?,
                AngleFormat::Degrees,
            );
        }
        spin.comments = s.comments.clone();
        spins.push(spin);
    }

    let mut inertias = Vec::new();
    for i in data.inertias() {
        let frame = ADMReferenceFrame::parse(&i.inertia_ref_frame);
        let mut inertia = APMInertia::new(
            frame,
            i.ixx.parse_f64()?,
            i.iyy.parse_f64()?,
            i.izz.parse_f64()?,
            i.ixy.parse_f64()?,
            i.ixz.parse_f64()?,
            i.iyz.parse_f64()?,
        );
        inertia.comments = i.comments.clone();
        inertias.push(inertia);
    }

    let mut maneuvers = Vec::new();
    for m in data.maneuvers() {
        let epoch_start = parse_ccsds_datetime(&m.epoch_start, &time_system)?;
        let frame = ADMReferenceFrame::parse(&m.ref_frame);
        let torque = Vector3::new(
            m.tor_x.parse_f64()?,
            m.tor_y.parse_f64()?,
            m.tor_z.parse_f64()?,
        );
        let mut man = APMManeuver::new(epoch_start, m.duration.parse_f64()?, frame, torque);
        if let Some(ref dm) = m.delta_mass {
            man = man.with_delta_mass(dm.parse_f64()?)?;
        }
        man.comments = m.comments.clone();
        maneuvers.push(man);
    }

    let apm = APM {
        header,
        metadata,
        epoch,
        comments: data.comments(),
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

// ============================================================================
// APM XML Writer
// ============================================================================

/// Write an APM message to XML format.
pub fn write_apm_xml(apm: &APM) -> Result<String, BraheError> {
    if !apm.has_blocks() {
        return Err(ccsds_missing_field("APM", "at least one logical block"));
    }
    apm.validate_maneuvers()?;

    let mut out = String::new();
    let i1 = "  ";
    let i2 = "    ";
    let i3 = "      ";
    let i4 = "        ";
    let i5 = "          ";
    let i6 = "            ";

    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str(&format!(
        "<apm xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" id=\"CCSDS_APM_VERS\" version=\"{:.1}\">\n",
        apm.header.format_version
    ));

    // Header
    out.push_str(&format!("{}<header>\n", i1));
    for c in &apm.header.comments {
        out.push_str(&format!(
            "{}<COMMENT>{}</COMMENT>\n",
            i2,
            escape_xml_text(c)
        ));
    }
    if let Some(ref cl) = apm.header.classification {
        out.push_str(&format!(
            "{}<CLASSIFICATION>{}</CLASSIFICATION>\n",
            i2,
            escape_xml_text(cl)
        ));
    }
    out.push_str(&format!(
        "{}<CREATION_DATE>{}</CREATION_DATE>\n",
        i2,
        format_ccsds_datetime_in(&apm.header.creation_date, &CCSDSTimeSystem::UTC)
    ));
    out.push_str(&format!(
        "{}<ORIGINATOR>{}</ORIGINATOR>\n",
        i2,
        escape_xml_text(&apm.header.originator)
    ));
    if let Some(ref mid) = apm.header.message_id {
        out.push_str(&format!(
            "{}<MESSAGE_ID>{}</MESSAGE_ID>\n",
            i2,
            escape_xml_text(mid)
        ));
    }
    out.push_str(&format!("{}</header>\n", i1));

    out.push_str(&format!("{}<body>\n", i1));
    out.push_str(&format!("{}<segment>\n", i2));

    // Metadata
    out.push_str(&format!("{}<metadata>\n", i3));
    for c in &apm.metadata.comments {
        out.push_str(&format!(
            "{}<COMMENT>{}</COMMENT>\n",
            i4,
            escape_xml_text(c)
        ));
    }
    out.push_str(&format!(
        "{}<OBJECT_NAME>{}</OBJECT_NAME>\n",
        i4,
        escape_xml_text(&apm.metadata.object_name)
    ));
    out.push_str(&format!(
        "{}<OBJECT_ID>{}</OBJECT_ID>\n",
        i4,
        escape_xml_text(&apm.metadata.object_id)
    ));
    if let Some(ref center) = apm.metadata.center_name {
        out.push_str(&format!(
            "{}<CENTER_NAME>{}</CENTER_NAME>\n",
            i4,
            escape_xml_text(center)
        ));
    }
    out.push_str(&format!(
        "{}<TIME_SYSTEM>{}</TIME_SYSTEM>\n",
        i4, apm.metadata.time_system
    ));
    out.push_str(&format!("{}</metadata>\n", i3));

    // Data
    out.push_str(&format!("{}<data>\n", i3));
    for c in &apm.comments {
        out.push_str(&format!(
            "{}<COMMENT>{}</COMMENT>\n",
            i4,
            escape_xml_text(c)
        ));
    }
    out.push_str(&format!(
        "{}<EPOCH>{}</EPOCH>\n",
        i4,
        format_ccsds_datetime_in(&apm.epoch, &apm.metadata.time_system)
    ));

    // Quaternion blocks
    for q in &apm.quaternion_states {
        out.push_str(&format!("{}<quaternionState>\n", i4));
        for c in &q.comments {
            out.push_str(&format!(
                "{}<COMMENT>{}</COMMENT>\n",
                i5,
                escape_xml_text(c)
            ));
        }
        out.push_str(&format!(
            "{}<REF_FRAME_A>{}</REF_FRAME_A>\n",
            i5,
            escape_xml_text(&format!("{}", q.ref_frame_a))
        ));
        out.push_str(&format!(
            "{}<REF_FRAME_B>{}</REF_FRAME_B>\n",
            i5,
            escape_xml_text(&format!("{}", q.ref_frame_b))
        ));
        let v = q.quaternion.to_vector(false);
        out.push_str(&format!("{}<quaternion>\n", i5));
        out.push_str(&format!("{}<Q1>{}</Q1>\n", i6, v[0]));
        out.push_str(&format!("{}<Q2>{}</Q2>\n", i6, v[1]));
        out.push_str(&format!("{}<Q3>{}</Q3>\n", i6, v[2]));
        out.push_str(&format!("{}<QC>{}</QC>\n", i6, v[3]));
        out.push_str(&format!("{}</quaternion>\n", i5));
        if let Some(d) = q.quaternion_derivative {
            // d is stored scalar-first; wire order is scalar-last.
            out.push_str(&format!("{}<quaternionDot>\n", i5));
            out.push_str(&format!("{}<Q1_DOT>{}</Q1_DOT>\n", i6, d[1]));
            out.push_str(&format!("{}<Q2_DOT>{}</Q2_DOT>\n", i6, d[2]));
            out.push_str(&format!("{}<Q3_DOT>{}</Q3_DOT>\n", i6, d[3]));
            out.push_str(&format!("{}<QC_DOT>{}</QC_DOT>\n", i6, d[0]));
            out.push_str(&format!("{}</quaternionDot>\n", i5));
        }
        out.push_str(&format!("{}</quaternionState>\n", i4));
    }

    // Euler angle blocks
    for e in &apm.euler_states {
        out.push_str(&format!("{}<eulerAngleState>\n", i4));
        for c in &e.comments {
            out.push_str(&format!(
                "{}<COMMENT>{}</COMMENT>\n",
                i5,
                escape_xml_text(c)
            ));
        }
        out.push_str(&format!(
            "{}<REF_FRAME_A>{}</REF_FRAME_A>\n",
            i5,
            escape_xml_text(&format!("{}", e.ref_frame_a))
        ));
        out.push_str(&format!(
            "{}<REF_FRAME_B>{}</REF_FRAME_B>\n",
            i5,
            escape_xml_text(&format!("{}", e.ref_frame_b))
        ));
        out.push_str(&format!(
            "{}<EULER_ROT_SEQ>{}</EULER_ROT_SEQ>\n",
            i5,
            format_euler_rot_seq(e.angles.order)
        ));
        out.push_str(&format!(
            "{}<ANGLE_1>{}</ANGLE_1>\n",
            i5,
            e.angles.phi * RAD2DEG
        ));
        out.push_str(&format!(
            "{}<ANGLE_2>{}</ANGLE_2>\n",
            i5,
            e.angles.theta * RAD2DEG
        ));
        out.push_str(&format!(
            "{}<ANGLE_3>{}</ANGLE_3>\n",
            i5,
            e.angles.psi * RAD2DEG
        ));
        if let Some(r) = e.rates {
            out.push_str(&format!(
                "{}<ANGLE_1_DOT>{}</ANGLE_1_DOT>\n",
                i5,
                r[0] * RAD2DEG
            ));
            out.push_str(&format!(
                "{}<ANGLE_2_DOT>{}</ANGLE_2_DOT>\n",
                i5,
                r[1] * RAD2DEG
            ));
            out.push_str(&format!(
                "{}<ANGLE_3_DOT>{}</ANGLE_3_DOT>\n",
                i5,
                r[2] * RAD2DEG
            ));
        }
        out.push_str(&format!("{}</eulerAngleState>\n", i4));
    }

    // Angular velocity blocks
    for av in &apm.angular_velocities {
        out.push_str(&format!("{}<angularVelocity>\n", i4));
        for c in &av.comments {
            out.push_str(&format!(
                "{}<COMMENT>{}</COMMENT>\n",
                i5,
                escape_xml_text(c)
            ));
        }
        out.push_str(&format!(
            "{}<REF_FRAME_A>{}</REF_FRAME_A>\n",
            i5,
            escape_xml_text(&format!("{}", av.ref_frame_a))
        ));
        out.push_str(&format!(
            "{}<REF_FRAME_B>{}</REF_FRAME_B>\n",
            i5,
            escape_xml_text(&format!("{}", av.ref_frame_b))
        ));
        out.push_str(&format!(
            "{}<ANGVEL_FRAME>{}</ANGVEL_FRAME>\n",
            i5,
            escape_xml_text(&format!("{}", av.angvel_frame))
        ));
        out.push_str(&format!(
            "{}<ANGVEL_X>{}</ANGVEL_X>\n",
            i5,
            av.angular_velocity[0] * RAD2DEG
        ));
        out.push_str(&format!(
            "{}<ANGVEL_Y>{}</ANGVEL_Y>\n",
            i5,
            av.angular_velocity[1] * RAD2DEG
        ));
        out.push_str(&format!(
            "{}<ANGVEL_Z>{}</ANGVEL_Z>\n",
            i5,
            av.angular_velocity[2] * RAD2DEG
        ));
        out.push_str(&format!("{}</angularVelocity>\n", i4));
    }

    // Spin blocks
    for s in &apm.spins {
        out.push_str(&format!("{}<spin>\n", i4));
        for c in &s.comments {
            out.push_str(&format!(
                "{}<COMMENT>{}</COMMENT>\n",
                i5,
                escape_xml_text(c)
            ));
        }
        out.push_str(&format!(
            "{}<REF_FRAME_A>{}</REF_FRAME_A>\n",
            i5,
            escape_xml_text(&format!("{}", s.ref_frame_a))
        ));
        out.push_str(&format!(
            "{}<REF_FRAME_B>{}</REF_FRAME_B>\n",
            i5,
            escape_xml_text(&format!("{}", s.ref_frame_b))
        ));
        out.push_str(&format!(
            "{}<SPIN_ALPHA>{}</SPIN_ALPHA>\n",
            i5,
            s.spin_alpha * RAD2DEG
        ));
        out.push_str(&format!(
            "{}<SPIN_DELTA>{}</SPIN_DELTA>\n",
            i5,
            s.spin_delta * RAD2DEG
        ));
        out.push_str(&format!(
            "{}<SPIN_ANGLE>{}</SPIN_ANGLE>\n",
            i5,
            s.spin_angle * RAD2DEG
        ));
        out.push_str(&format!(
            "{}<SPIN_ANGLE_VEL>{}</SPIN_ANGLE_VEL>\n",
            i5,
            s.spin_angle_vel * RAD2DEG
        ));
        match &s.nutation {
            APMNutation::None => {}
            APMNutation::Angle {
                nutation,
                nutation_period,
                nutation_phase,
            } => {
                out.push_str(&format!(
                    "{}<NUTATION>{}</NUTATION>\n",
                    i5,
                    nutation * RAD2DEG
                ));
                out.push_str(&format!(
                    "{}<NUTATION_PER>{}</NUTATION_PER>\n",
                    i5, nutation_period
                ));
                out.push_str(&format!(
                    "{}<NUTATION_PHASE>{}</NUTATION_PHASE>\n",
                    i5,
                    nutation_phase * RAD2DEG
                ));
            }
            APMNutation::Momentum {
                momentum_alpha,
                momentum_delta,
                nutation_vel,
            } => {
                out.push_str(&format!(
                    "{}<MOMENTUM_ALPHA>{}</MOMENTUM_ALPHA>\n",
                    i5,
                    momentum_alpha * RAD2DEG
                ));
                out.push_str(&format!(
                    "{}<MOMENTUM_DELTA>{}</MOMENTUM_DELTA>\n",
                    i5,
                    momentum_delta * RAD2DEG
                ));
                out.push_str(&format!(
                    "{}<NUTATION_VEL>{}</NUTATION_VEL>\n",
                    i5,
                    nutation_vel * RAD2DEG
                ));
            }
        }
        out.push_str(&format!("{}</spin>\n", i4));
    }

    // Inertia blocks
    for i_blk in &apm.inertias {
        out.push_str(&format!("{}<inertia>\n", i4));
        for c in &i_blk.comments {
            out.push_str(&format!(
                "{}<COMMENT>{}</COMMENT>\n",
                i5,
                escape_xml_text(c)
            ));
        }
        out.push_str(&format!(
            "{}<INERTIA_REF_FRAME>{}</INERTIA_REF_FRAME>\n",
            i5,
            escape_xml_text(&format!("{}", i_blk.inertia_ref_frame))
        ));
        out.push_str(&format!("{}<IXX>{}</IXX>\n", i5, i_blk.ixx));
        out.push_str(&format!("{}<IYY>{}</IYY>\n", i5, i_blk.iyy));
        out.push_str(&format!("{}<IZZ>{}</IZZ>\n", i5, i_blk.izz));
        out.push_str(&format!("{}<IXY>{}</IXY>\n", i5, i_blk.ixy));
        out.push_str(&format!("{}<IXZ>{}</IXZ>\n", i5, i_blk.ixz));
        out.push_str(&format!("{}<IYZ>{}</IYZ>\n", i5, i_blk.iyz));
        out.push_str(&format!("{}</inertia>\n", i4));
    }

    // Maneuver blocks
    for m in &apm.maneuvers {
        out.push_str(&format!("{}<maneuverParameters>\n", i4));
        for c in &m.comments {
            out.push_str(&format!(
                "{}<COMMENT>{}</COMMENT>\n",
                i5,
                escape_xml_text(c)
            ));
        }
        out.push_str(&format!(
            "{}<MAN_EPOCH_START>{}</MAN_EPOCH_START>\n",
            i5,
            format_ccsds_datetime_in(&m.epoch_start, &apm.metadata.time_system)
        ));
        out.push_str(&format!(
            "{}<MAN_DURATION>{}</MAN_DURATION>\n",
            i5, m.duration
        ));
        out.push_str(&format!(
            "{}<MAN_REF_FRAME>{}</MAN_REF_FRAME>\n",
            i5,
            escape_xml_text(&format!("{}", m.ref_frame))
        ));
        out.push_str(&format!("{}<MAN_TOR_X>{}</MAN_TOR_X>\n", i5, m.torque[0]));
        out.push_str(&format!("{}<MAN_TOR_Y>{}</MAN_TOR_Y>\n", i5, m.torque[1]));
        out.push_str(&format!("{}<MAN_TOR_Z>{}</MAN_TOR_Z>\n", i5, m.torque[2]));
        if let Some(dm) = m.delta_mass {
            out.push_str(&format!("{}<MAN_DELTA_MASS>{}</MAN_DELTA_MASS>\n", i5, dm));
        }
        out.push_str(&format!("{}</maneuverParameters>\n", i4));
    }

    out.push_str(&format!("{}</data>\n", i3));
    out.push_str(&format!("{}</segment>\n", i2));

    out.push_str(&format!("{}</body>\n", i1));
    out.push_str("</apm>\n");

    validate_xml_characters("APM", &out)?;

    Ok(out)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    use crate::ccsds::xml::{parse_apm_xml, write_apm_xml};
    use serial_test::parallel;

    #[test]
    #[parallel]
    fn test_parse_apm_xml_example_g10() {
        let content = std::fs::read_to_string("test_assets/ccsds/apm/APMExampleG10.xml").unwrap();
        let apm = parse_apm_xml(&content).unwrap();

        // Header
        assert!((apm.header.format_version - 2.0).abs() < 1e-10);
        assert_eq!(apm.header.originator, "GSFC");
        assert_eq!(apm.header.message_id.as_deref(), Some("A7015Z1"));
        assert!(apm.header.classification.is_none());
        assert!(apm.header.comments.is_empty());
        let creation_date =
            parse_ccsds_datetime("2003-09-30T19:23:57", &CCSDSTimeSystem::UTC).unwrap();
        assert!((apm.header.creation_date - creation_date).abs() < 1e-6);

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
        let epoch =
            parse_ccsds_datetime("2003-09-30T14:28:15.1172", &CCSDSTimeSystem::UTC).unwrap();
        assert!((apm.epoch - epoch).abs() < 1e-6);
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
        assert_eq!(
            q.comments,
            vec!["Attitude state vector quaternion".to_string()]
        );
        let v = q.quaternion.to_vector(false);
        assert!((v[0] - 0.00005).abs() < 1e-4);
        assert!((v[1] - 0.87543).abs() < 1e-4);
        assert!((v[2] - 0.40949).abs() < 1e-4);
        assert!((v[3] - 0.25678).abs() < 1e-4);
        assert!(q.quaternion_derivative.is_none());
        assert!(apm.euler_states.is_empty());
        assert!(apm.angular_velocities.is_empty());
        assert!(apm.spins.is_empty());
        assert!(apm.inertias.is_empty());
        assert!(apm.maneuvers.is_empty());
    }

    #[test]
    #[parallel]
    fn test_apm_xml_round_trip() {
        let content = std::fs::read_to_string("test_assets/ccsds/apm/APMExampleG10.xml").unwrap();
        let apm = parse_apm_xml(&content).unwrap();
        let xml = write_apm_xml(&apm).unwrap();
        let apm2 = parse_apm_xml(&xml).unwrap();

        assert!((apm.header.format_version - apm2.header.format_version).abs() < 1e-10);
        assert_eq!(apm.header.originator, apm2.header.originator);
        assert_eq!(apm.header.message_id, apm2.header.message_id);
        assert_eq!(apm.header.comments, apm2.header.comments);
        assert!((apm.header.creation_date - apm2.header.creation_date).abs() < 1e-6);

        assert_eq!(apm.metadata.object_name, apm2.metadata.object_name);
        assert_eq!(apm.metadata.object_id, apm2.metadata.object_id);
        assert_eq!(apm.metadata.center_name, apm2.metadata.center_name);
        assert_eq!(apm.metadata.time_system, apm2.metadata.time_system);
        assert_eq!(apm.metadata.comments, apm2.metadata.comments);

        assert!((apm.epoch - apm2.epoch).abs() < 1e-6);
        assert_eq!(apm.comments, apm2.comments);

        assert_eq!(apm.quaternion_states.len(), apm2.quaternion_states.len());
        let (q1, q2) = (&apm.quaternion_states[0], &apm2.quaternion_states[0]);
        assert_eq!(q1.ref_frame_a, q2.ref_frame_a);
        assert_eq!(q1.ref_frame_b, q2.ref_frame_b);
        assert_eq!(q1.comments, q2.comments);
        let (v1, v2) = (
            q1.quaternion.to_vector(false),
            q2.quaternion.to_vector(false),
        );
        for i in 0..4 {
            assert!((v1[i] - v2[i]).abs() < 1e-12);
        }
        assert_eq!(
            q1.quaternion_derivative.is_some(),
            q2.quaternion_derivative.is_some()
        );
    }

    #[test]
    #[parallel]
    fn test_apm_kvn_to_xml_cross_format() {
        let kvn_content =
            std::fs::read_to_string("test_assets/ccsds/apm/APMExampleG1.txt").unwrap();
        let apm_kvn = crate::ccsds::kvn::parse_apm(&kvn_content).unwrap();

        let xml = write_apm_xml(&apm_kvn).unwrap();
        let apm_xml = parse_apm_xml(&xml).unwrap();

        assert_eq!(apm_kvn.header.originator, apm_xml.header.originator);
        assert_eq!(apm_kvn.header.message_id, apm_xml.header.message_id);
        assert!((apm_kvn.header.creation_date - apm_xml.header.creation_date).abs() < 1e-6);

        assert_eq!(apm_kvn.metadata.object_name, apm_xml.metadata.object_name);
        assert_eq!(apm_kvn.metadata.object_id, apm_xml.metadata.object_id);
        assert_eq!(apm_kvn.metadata.center_name, apm_xml.metadata.center_name);
        assert_eq!(apm_kvn.metadata.time_system, apm_xml.metadata.time_system);

        assert!((apm_kvn.epoch - apm_xml.epoch).abs() < 1e-6);

        assert_eq!(
            apm_kvn.quaternion_states.len(),
            apm_xml.quaternion_states.len()
        );
        let (q1, q2) = (&apm_kvn.quaternion_states[0], &apm_xml.quaternion_states[0]);
        assert_eq!(q1.ref_frame_a, q2.ref_frame_a);
        assert_eq!(q1.ref_frame_b, q2.ref_frame_b);
        let (v1, v2) = (
            q1.quaternion.to_vector(false),
            q2.quaternion.to_vector(false),
        );
        for i in 0..4 {
            assert!((v1[i] - v2[i]).abs() < 1e-9);
        }
    }

    #[test]
    #[parallel]
    fn test_parse_apm_xml_quaternion_derivative() {
        let content = r#"<?xml version="1.0" encoding="UTF-8"?>
<apm id="CCSDS_APM_VERS" version="2.0">
   <header>
      <CREATION_DATE>2003-09-30T19:23:57</CREATION_DATE>
      <ORIGINATOR>GSFC</ORIGINATOR>
   </header>
   <body>
      <segment>
         <metadata>
            <OBJECT_NAME>TRMM</OBJECT_NAME>
            <OBJECT_ID>1997-074A</OBJECT_ID>
            <TIME_SYSTEM>UTC</TIME_SYSTEM>
         </metadata>
         <data>
            <EPOCH>2003-09-30T14:28:15.1172</EPOCH>
            <quaternionState>
               <REF_FRAME_A>ICRF</REF_FRAME_A>
               <REF_FRAME_B>SC_BODY_1</REF_FRAME_B>
               <quaternion>
                  <Q1>0.0</Q1>
                  <Q2>0.0</Q2>
                  <Q3>0.0</Q3>
                  <QC>1.0</QC>
               </quaternion>
               <quaternionDot>
                  <Q1_DOT>0.1</Q1_DOT>
                  <Q2_DOT>0.2</Q2_DOT>
                  <Q3_DOT>0.3</Q3_DOT>
                  <QC_DOT>0.4</QC_DOT>
               </quaternionDot>
            </quaternionState>
         </data>
      </segment>
   </body>
</apm>
"#;
        let apm = parse_apm_xml(content).unwrap();
        let d = apm.quaternion_states[0].quaternion_derivative.unwrap();
        // Stored scalar-first: [QC_DOT, Q1_DOT, Q2_DOT, Q3_DOT]
        assert!((d[0] - 0.4).abs() < 1e-10);
        assert!((d[1] - 0.1).abs() < 1e-10);
        assert!((d[2] - 0.2).abs() < 1e-10);
        assert!((d[3] - 0.3).abs() < 1e-10);

        // Round-trip through the writer preserves the derivative.
        let xml = write_apm_xml(&apm).unwrap();
        assert!(xml.contains("<quaternionDot>"));
        let apm2 = parse_apm_xml(&xml).unwrap();
        let d2 = apm2.quaternion_states[0].quaternion_derivative.unwrap();
        assert!((d2[0] - 0.4).abs() < 1e-10);
        assert!((d2[1] - 0.1).abs() < 1e-10);
        assert!((d2[2] - 0.2).abs() < 1e-10);
        assert!((d2[3] - 0.3).abs() < 1e-10);
    }

    #[test]
    #[parallel]
    fn test_parse_apm_xml_v1_version_rejected() {
        let content = std::fs::read_to_string("test_assets/ccsds/apm/APMExampleG10.xml")
            .unwrap()
            .replace("version=\"2.0\"", "version=\"1.0\"");
        let result = parse_apm_xml(&content);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("version 1.0"));
        assert!(err.contains("504.0-B-1"));
    }

    fn apm_for_xml_test() -> APM {
        let metadata = APMMetadata::new("SAT1", "2024-001A", CCSDSTimeSystem::UTC);
        let epoch = crate::time::Epoch::from_datetime(
            2024,
            3,
            1,
            0,
            0,
            0.0,
            0.0,
            crate::time::TimeSystem::UTC,
        );
        let mut apm = APM::new("BRAHE", metadata, epoch);
        apm.push_quaternion_state(APMQuaternionState::new(
            ADMReferenceFrame::parse("ICRF"),
            ADMReferenceFrame::parse("SC_BODY_1"),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
        ));
        apm
    }

    #[test]
    #[parallel]
    fn test_write_apm_xml_root_has_xmlns_xsi() {
        let apm = apm_for_xml_test();
        let xml = write_apm_xml(&apm).unwrap();
        assert!(
            xml.contains("<apm xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" id=\"CCSDS_APM_VERS\" version=\"2.0\">"),
            "unexpected root element: {}",
            xml.lines().nth(1).unwrap_or("")
        );

        let reparsed = parse_apm_xml(&xml).unwrap();
        assert_eq!(reparsed.metadata.object_name, "SAT1");
    }

    #[test]
    #[parallel]
    fn test_write_apm_xml_escapes_free_text() {
        let mut apm = apm_for_xml_test();
        apm.header.originator = "A & B <test>".to_string();

        let xml = write_apm_xml(&apm).unwrap();
        assert!(xml.contains("<ORIGINATOR>A &amp; B &lt;test&gt;</ORIGINATOR>"));
        assert!(!xml.contains("<ORIGINATOR>A & B <test></ORIGINATOR>"));

        let reparsed = parse_apm_xml(&xml).unwrap();
        assert_eq!(reparsed.header.originator, "A & B <test>");
    }
}
