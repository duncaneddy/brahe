/*!
 * XML parser for CCSDS OEM, OMM, and OPM messages.
 *
 * Uses `quick-xml` with serde to deserialize into intermediate structs,
 * then converts to the public CCSDS types with unit conversion.
 */

use std::collections::HashMap;

use nalgebra::{Vector3, Vector4};
use serde::Deserialize;

use crate::attitude::attitude_types::{EulerAngle, Quaternion};
use crate::ccsds::apm::{
    APM, APMAngularVelocity, APMEulerState, APMHeader, APMInertia, APMManeuver, APMMetadata,
    APMQuaternionState, APMSpin,
};
use crate::ccsds::common::{
    CCSDSCovariance, CCSDSRefFrame, CCSDSTimeSystem, CCSDSUserDefined, ODMHeader,
    covariance_from_lower_triangular, parse_ccsds_datetime, parse_euler_rot_seq,
};
use crate::ccsds::error::{ccsds_missing_field, ccsds_parse_error};
use crate::ccsds::frames::ADMReferenceFrame;
use crate::ccsds::oem::{OEM, OEMMetadata, OEMSegment, OEMStateVector};
use crate::constants::{AngleFormat, DEG2RAD};
use crate::utils::errors::BraheError;

// ============================================================================
// Intermediate XML structs for OEM
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(rename = "oem")]
#[allow(clippy::upper_case_acronyms)]
struct XMLOEM {
    #[serde(rename = "@version")]
    version: Option<String>,
    header: XMLHeader,
    body: XMLOEMBody,
}

#[derive(Debug, Deserialize)]
struct XMLHeader {
    #[serde(rename = "$value")]
    items: Vec<XMLHeaderItem>,
}

#[derive(Debug, Deserialize)]
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
enum XMLHeaderItem {
    CREATION_DATE(String),
    ORIGINATOR(String),
    MESSAGE_ID(String),
    CLASSIFICATION(String),
    COMMENT(String),
}

impl XMLHeader {
    fn creation_date(&self) -> Option<&str> {
        self.items.iter().find_map(|item| {
            if let XMLHeaderItem::CREATION_DATE(s) = item {
                Some(s.as_str())
            } else {
                None
            }
        })
    }

    fn originator(&self) -> Option<&str> {
        self.items.iter().find_map(|item| {
            if let XMLHeaderItem::ORIGINATOR(s) = item {
                Some(s.as_str())
            } else {
                None
            }
        })
    }

    fn message_id(&self) -> Option<String> {
        self.items.iter().find_map(|item| {
            if let XMLHeaderItem::MESSAGE_ID(s) = item {
                Some(s.clone())
            } else {
                None
            }
        })
    }

    fn classification(&self) -> Option<String> {
        self.items.iter().find_map(|item| {
            if let XMLHeaderItem::CLASSIFICATION(s) = item {
                Some(s.clone())
            } else {
                None
            }
        })
    }

    fn comments(&self) -> Vec<String> {
        self.items
            .iter()
            .filter_map(|item| {
                if let XMLHeaderItem::COMMENT(s) = item {
                    Some(s.trim().to_string())
                } else {
                    None
                }
            })
            .collect()
    }
}

#[derive(Debug, Deserialize)]
struct XMLOEMBody {
    #[serde(rename = "segment")]
    segments: Vec<XMLOEMSegment>,
}

#[derive(Debug, Deserialize)]
struct XMLOEMSegment {
    metadata: XMLOEMMetadata,
    data: XMLOEMData,
}

#[derive(Debug, Deserialize)]
struct XMLOEMMetadata {
    #[serde(rename = "$value")]
    items: Vec<XMLOEMMetadataItem>,
}

#[derive(Debug, Deserialize)]
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
enum XMLOEMMetadataItem {
    OBJECT_NAME(String),
    OBJECT_ID(String),
    CENTER_NAME(String),
    REF_FRAME(String),
    REF_FRAME_EPOCH(String),
    TIME_SYSTEM(String),
    START_TIME(String),
    USEABLE_START_TIME(String),
    USEABLE_STOP_TIME(String),
    STOP_TIME(String),
    INTERPOLATION(String),
    INTERPOLATION_DEGREE(u32),
    COMMENT(String),
}

impl XMLOEMMetadata {
    fn find_str(&self, variant: &str) -> Option<&str> {
        self.items.iter().find_map(|item| match item {
            XMLOEMMetadataItem::OBJECT_NAME(s) if variant == "OBJECT_NAME" => Some(s.as_str()),
            XMLOEMMetadataItem::OBJECT_ID(s) if variant == "OBJECT_ID" => Some(s.as_str()),
            XMLOEMMetadataItem::CENTER_NAME(s) if variant == "CENTER_NAME" => Some(s.as_str()),
            XMLOEMMetadataItem::REF_FRAME(s) if variant == "REF_FRAME" => Some(s.as_str()),
            XMLOEMMetadataItem::REF_FRAME_EPOCH(s) if variant == "REF_FRAME_EPOCH" => {
                Some(s.as_str())
            }
            XMLOEMMetadataItem::TIME_SYSTEM(s) if variant == "TIME_SYSTEM" => Some(s.as_str()),
            XMLOEMMetadataItem::START_TIME(s) if variant == "START_TIME" => Some(s.as_str()),
            XMLOEMMetadataItem::USEABLE_START_TIME(s) if variant == "USEABLE_START_TIME" => {
                Some(s.as_str())
            }
            XMLOEMMetadataItem::USEABLE_STOP_TIME(s) if variant == "USEABLE_STOP_TIME" => {
                Some(s.as_str())
            }
            XMLOEMMetadataItem::STOP_TIME(s) if variant == "STOP_TIME" => Some(s.as_str()),
            XMLOEMMetadataItem::INTERPOLATION(s) if variant == "INTERPOLATION" => Some(s.as_str()),
            _ => None,
        })
    }

    fn interpolation_degree(&self) -> Option<u32> {
        self.items.iter().find_map(|item| {
            if let XMLOEMMetadataItem::INTERPOLATION_DEGREE(v) = item {
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
                if let XMLOEMMetadataItem::COMMENT(s) = item {
                    Some(s.trim().to_string())
                } else {
                    None
                }
            })
            .collect()
    }
}

/// OEM data block containing state vectors, covariance, and comments.
///
/// Uses `$value` to capture all child elements as a flat sequence, since
/// quick-xml cannot handle multiple `<COMMENT>` elements as a Vec directly.
#[derive(Debug, Deserialize)]
struct XMLOEMData {
    #[serde(rename = "$value", default)]
    items: Vec<XMLOEMDataItem>,
}

/// Individual items within an OEM data block.
#[derive(Debug, Deserialize)]
#[allow(clippy::large_enum_variant)]
enum XMLOEMDataItem {
    #[serde(rename = "COMMENT")]
    Comment(String),
    #[serde(rename = "stateVector")]
    StateVector(XMLStateVector),
    #[serde(rename = "covarianceMatrix")]
    CovarianceMatrix(XMLCovarianceMatrix),
}

impl XMLOEMData {
    fn comments(&self) -> Vec<String> {
        self.items
            .iter()
            .filter_map(|item| {
                if let XMLOEMDataItem::Comment(s) = item {
                    Some(s.trim().to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    fn state_vectors(&self) -> Vec<&XMLStateVector> {
        self.items
            .iter()
            .filter_map(|item| {
                if let XMLOEMDataItem::StateVector(sv) = item {
                    Some(sv)
                } else {
                    None
                }
            })
            .collect()
    }

    fn covariance_matrices(&self) -> Vec<&XMLCovarianceMatrix> {
        self.items
            .iter()
            .filter_map(|item| {
                if let XMLOEMDataItem::CovarianceMatrix(cm) = item {
                    Some(cm)
                } else {
                    None
                }
            })
            .collect()
    }
}

#[derive(Debug, Deserialize)]
struct XMLStateVector {
    #[serde(rename = "EPOCH")]
    epoch: String,
    #[serde(rename = "X")]
    x: XMLValue,
    #[serde(rename = "Y")]
    y: XMLValue,
    #[serde(rename = "Z")]
    z: XMLValue,
    #[serde(rename = "X_DOT")]
    x_dot: XMLValue,
    #[serde(rename = "Y_DOT")]
    y_dot: XMLValue,
    #[serde(rename = "Z_DOT")]
    z_dot: XMLValue,
    #[serde(rename = "X_DDOT")]
    x_ddot: Option<XMLValue>,
    #[serde(rename = "Y_DDOT")]
    y_ddot: Option<XMLValue>,
    #[serde(rename = "Z_DDOT")]
    z_ddot: Option<XMLValue>,
}

/// Wrapper for XML values that may have unit attributes.
#[derive(Debug, Deserialize)]
struct XMLValue {
    #[serde(rename = "@units")]
    _units: Option<String>,
    #[serde(rename = "$text")]
    value: String,
}

impl XMLValue {
    fn parse_f64(&self) -> Result<f64, BraheError> {
        self.value.trim().parse::<f64>().map_err(|_| {
            ccsds_parse_error("XML", &format!("invalid numeric value '{}'", self.value))
        })
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct XMLCovarianceMatrix {
    #[serde(rename = "EPOCH")]
    pub epoch: Option<String>,
    #[serde(rename = "COV_REF_FRAME")]
    pub cov_ref_frame: Option<String>,
    #[serde(rename = "CX_X")]
    cx_x: XMLValue,
    #[serde(rename = "CY_X")]
    cy_x: XMLValue,
    #[serde(rename = "CY_Y")]
    cy_y: XMLValue,
    #[serde(rename = "CZ_X")]
    cz_x: XMLValue,
    #[serde(rename = "CZ_Y")]
    cz_y: XMLValue,
    #[serde(rename = "CZ_Z")]
    cz_z: XMLValue,
    #[serde(rename = "CX_DOT_X")]
    cx_dot_x: XMLValue,
    #[serde(rename = "CX_DOT_Y")]
    cx_dot_y: XMLValue,
    #[serde(rename = "CX_DOT_Z")]
    cx_dot_z: XMLValue,
    #[serde(rename = "CX_DOT_X_DOT")]
    cx_dot_x_dot: XMLValue,
    #[serde(rename = "CY_DOT_X")]
    cy_dot_x: XMLValue,
    #[serde(rename = "CY_DOT_Y")]
    cy_dot_y: XMLValue,
    #[serde(rename = "CY_DOT_Z")]
    cy_dot_z: XMLValue,
    #[serde(rename = "CY_DOT_X_DOT")]
    cy_dot_x_dot: XMLValue,
    #[serde(rename = "CY_DOT_Y_DOT")]
    cy_dot_y_dot: XMLValue,
    #[serde(rename = "CZ_DOT_X")]
    cz_dot_x: XMLValue,
    #[serde(rename = "CZ_DOT_Y")]
    cz_dot_y: XMLValue,
    #[serde(rename = "CZ_DOT_Z")]
    cz_dot_z: XMLValue,
    #[serde(rename = "CZ_DOT_X_DOT")]
    cz_dot_x_dot: XMLValue,
    #[serde(rename = "CZ_DOT_Y_DOT")]
    cz_dot_y_dot: XMLValue,
    #[serde(rename = "CZ_DOT_Z_DOT")]
    cz_dot_z_dot: XMLValue,
    #[serde(rename = "COMMENT", default)]
    comment: Vec<String>,
}

// ============================================================================
// Conversion: XML intermediate → public types
// ============================================================================

pub(crate) fn convert_xml_covariance(
    xml_cov: &XMLCovarianceMatrix,
    time_system: &CCSDSTimeSystem,
) -> Result<CCSDSCovariance, BraheError> {
    let epoch = xml_cov
        .epoch
        .as_ref()
        .map(|s| parse_ccsds_datetime(s, time_system))
        .transpose()?;

    let cov_ref_frame = xml_cov
        .cov_ref_frame
        .as_ref()
        .map(|s| CCSDSRefFrame::parse(s));

    let values: [f64; 21] = [
        xml_cov.cx_x.parse_f64()?,
        xml_cov.cy_x.parse_f64()?,
        xml_cov.cy_y.parse_f64()?,
        xml_cov.cz_x.parse_f64()?,
        xml_cov.cz_y.parse_f64()?,
        xml_cov.cz_z.parse_f64()?,
        xml_cov.cx_dot_x.parse_f64()?,
        xml_cov.cx_dot_y.parse_f64()?,
        xml_cov.cx_dot_z.parse_f64()?,
        xml_cov.cx_dot_x_dot.parse_f64()?,
        xml_cov.cy_dot_x.parse_f64()?,
        xml_cov.cy_dot_y.parse_f64()?,
        xml_cov.cy_dot_z.parse_f64()?,
        xml_cov.cy_dot_x_dot.parse_f64()?,
        xml_cov.cy_dot_y_dot.parse_f64()?,
        xml_cov.cz_dot_x.parse_f64()?,
        xml_cov.cz_dot_y.parse_f64()?,
        xml_cov.cz_dot_z.parse_f64()?,
        xml_cov.cz_dot_x_dot.parse_f64()?,
        xml_cov.cz_dot_y_dot.parse_f64()?,
        xml_cov.cz_dot_z_dot.parse_f64()?,
    ];

    // XML covariance values are in km² units — convert to m²
    let matrix = covariance_from_lower_triangular(&values, 1e6);

    Ok(CCSDSCovariance {
        epoch,
        cov_ref_frame,
        matrix,
        comments: xml_cov
            .comment
            .iter()
            .map(|s| s.trim().to_string())
            .collect(),
    })
}

/// Parse an OEM message from XML format.
pub fn parse_oem_xml(content: &str) -> Result<OEM, BraheError> {
    let xml_oem: XMLOEM = quick_xml::de::from_str(content)
        .map_err(|e| ccsds_parse_error("OEM", &format!("XML parse error: {}", e)))?;

    let format_version = xml_oem
        .version
        .as_ref()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(3.0);

    let creation_date_str = xml_oem
        .header
        .creation_date()
        .ok_or_else(|| ccsds_parse_error("OEM", "missing CREATION_DATE in header"))?;
    let originator = xml_oem
        .header
        .originator()
        .ok_or_else(|| ccsds_parse_error("OEM", "missing ORIGINATOR in header"))?
        .to_string();

    let header = ODMHeader {
        format_version,
        classification: xml_oem.header.classification(),
        creation_date: parse_ccsds_datetime(creation_date_str, &CCSDSTimeSystem::UTC)?,
        originator,
        message_id: xml_oem.header.message_id(),
        comments: xml_oem.header.comments(),
    };

    let mut segments = Vec::new();
    for xml_seg in xml_oem.body.segments {
        let meta = &xml_seg.metadata;
        let time_system_str = meta
            .find_str("TIME_SYSTEM")
            .ok_or_else(|| ccsds_parse_error("OEM", "missing TIME_SYSTEM in metadata"))?;
        let time_system = CCSDSTimeSystem::parse(time_system_str)?;

        let ref_frame_epoch = meta
            .find_str("REF_FRAME_EPOCH")
            .map(|s| parse_ccsds_datetime(s, &time_system))
            .transpose()?;

        let metadata = OEMMetadata {
            object_name: meta
                .find_str("OBJECT_NAME")
                .ok_or_else(|| ccsds_parse_error("OEM", "missing OBJECT_NAME"))?
                .to_string(),
            object_id: meta
                .find_str("OBJECT_ID")
                .ok_or_else(|| ccsds_parse_error("OEM", "missing OBJECT_ID"))?
                .to_string(),
            center_name: meta
                .find_str("CENTER_NAME")
                .ok_or_else(|| ccsds_parse_error("OEM", "missing CENTER_NAME"))?
                .to_string(),
            ref_frame: CCSDSRefFrame::parse(
                meta.find_str("REF_FRAME")
                    .ok_or_else(|| ccsds_parse_error("OEM", "missing REF_FRAME"))?,
            ),
            ref_frame_epoch,
            time_system: time_system.clone(),
            start_time: parse_ccsds_datetime(
                meta.find_str("START_TIME")
                    .ok_or_else(|| ccsds_parse_error("OEM", "missing START_TIME"))?,
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
            stop_time: parse_ccsds_datetime(
                meta.find_str("STOP_TIME")
                    .ok_or_else(|| ccsds_parse_error("OEM", "missing STOP_TIME"))?,
                &time_system,
            )?,
            interpolation: meta.find_str("INTERPOLATION").map(|s| s.to_string()),
            interpolation_degree: meta.interpolation_degree(),
            comments: meta.comments(),
        };

        let mut states = Vec::new();
        for sv in xml_seg.data.state_vectors().iter() {
            let epoch = parse_ccsds_datetime(&sv.epoch, &time_system)?;

            // XML values are in km and km/s — convert to m and m/s
            let position = [
                sv.x.parse_f64()? * 1000.0,
                sv.y.parse_f64()? * 1000.0,
                sv.z.parse_f64()? * 1000.0,
            ];
            let velocity = [
                sv.x_dot.parse_f64()? * 1000.0,
                sv.y_dot.parse_f64()? * 1000.0,
                sv.z_dot.parse_f64()? * 1000.0,
            ];
            let acceleration = match (&sv.x_ddot, &sv.y_ddot, &sv.z_ddot) {
                (Some(ax), Some(ay), Some(az)) => Some([
                    ax.parse_f64()? * 1000.0,
                    ay.parse_f64()? * 1000.0,
                    az.parse_f64()? * 1000.0,
                ]),
                _ => None,
            };

            states.push(OEMStateVector {
                epoch,
                position,
                velocity,
                acceleration,
            });
        }

        let mut covariances = Vec::new();
        for xml_cov in xml_seg.data.covariance_matrices().iter() {
            covariances.push(convert_xml_covariance(xml_cov, &time_system)?);
        }

        segments.push(OEMSegment {
            metadata,
            comments: xml_seg.data.comments(),
            states,
            covariances,
        });
    }

    Ok(OEM { header, segments })
}

// ============================================================================
// Intermediate XML structs for OMM
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(rename = "omm")]
#[allow(clippy::upper_case_acronyms)]
struct XMLOMM {
    #[serde(rename = "@version")]
    version: Option<String>,
    header: XMLHeader,
    body: XMLOMMBody,
}

#[derive(Debug, Deserialize)]
struct XMLOMMBody {
    segment: XMLOMMSegment,
}

#[derive(Debug, Deserialize)]
struct XMLOMMSegment {
    metadata: XMLOMMMetadata,
    data: XMLOMMData,
}

#[derive(Debug, Deserialize)]
struct XMLOMMMetadata {
    #[serde(rename = "$value")]
    items: Vec<XMLOMMMetadataItem>,
}

#[derive(Debug, Deserialize)]
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
enum XMLOMMMetadataItem {
    OBJECT_NAME(String),
    OBJECT_ID(String),
    CENTER_NAME(String),
    REF_FRAME(String),
    REF_FRAME_EPOCH(String),
    TIME_SYSTEM(String),
    MEAN_ELEMENT_THEORY(String),
    COMMENT(String),
}

impl XMLOMMMetadata {
    fn find_str(&self, variant: &str) -> Option<&str> {
        self.items.iter().find_map(|item| match item {
            XMLOMMMetadataItem::OBJECT_NAME(s) if variant == "OBJECT_NAME" => Some(s.as_str()),
            XMLOMMMetadataItem::OBJECT_ID(s) if variant == "OBJECT_ID" => Some(s.as_str()),
            XMLOMMMetadataItem::CENTER_NAME(s) if variant == "CENTER_NAME" => Some(s.as_str()),
            XMLOMMMetadataItem::REF_FRAME(s) if variant == "REF_FRAME" => Some(s.as_str()),
            XMLOMMMetadataItem::REF_FRAME_EPOCH(s) if variant == "REF_FRAME_EPOCH" => {
                Some(s.as_str())
            }
            XMLOMMMetadataItem::TIME_SYSTEM(s) if variant == "TIME_SYSTEM" => Some(s.as_str()),
            XMLOMMMetadataItem::MEAN_ELEMENT_THEORY(s) if variant == "MEAN_ELEMENT_THEORY" => {
                Some(s.as_str())
            }
            _ => None,
        })
    }

    fn comments(&self) -> Vec<String> {
        self.items
            .iter()
            .filter_map(|item| {
                if let XMLOMMMetadataItem::COMMENT(s) = item {
                    Some(s.trim().to_string())
                } else {
                    None
                }
            })
            .collect()
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct XMLOMMData {
    mean_elements: XMLMeanElements,
    #[serde(default)]
    tle_parameters: Option<XMLTleParameters>,
    #[serde(default)]
    spacecraft_parameters: Option<XMLSpacecraftParameters>,
    #[serde(default)]
    covariance_matrix: Option<XMLCovarianceMatrix>,
}

#[derive(Debug, Deserialize)]
struct XMLMeanElements {
    #[serde(rename = "EPOCH")]
    epoch: String,
    #[serde(rename = "MEAN_MOTION")]
    mean_motion: Option<XMLValue>,
    #[serde(rename = "SEMI_MAJOR_AXIS")]
    semi_major_axis: Option<XMLValue>,
    #[serde(rename = "ECCENTRICITY")]
    eccentricity: XMLValue,
    #[serde(rename = "INCLINATION")]
    inclination: XMLValue,
    #[serde(rename = "RA_OF_ASC_NODE")]
    ra_of_asc_node: XMLValue,
    #[serde(rename = "ARG_OF_PERICENTER")]
    arg_of_pericenter: XMLValue,
    #[serde(rename = "MEAN_ANOMALY")]
    mean_anomaly: XMLValue,
    #[serde(rename = "GM")]
    gm: Option<XMLValue>,
    #[serde(rename = "COMMENT", default)]
    comments: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct XMLTleParameters {
    #[serde(rename = "EPHEMERIS_TYPE")]
    ephemeris_type: Option<XMLValue>,
    #[serde(rename = "CLASSIFICATION_TYPE")]
    classification_type: Option<XMLValue>,
    #[serde(rename = "NORAD_CAT_ID")]
    norad_cat_id: Option<XMLValue>,
    #[serde(rename = "ELEMENT_SET_NO")]
    element_set_no: Option<XMLValue>,
    #[serde(rename = "REV_AT_EPOCH")]
    rev_at_epoch: Option<XMLValue>,
    #[serde(rename = "BSTAR")]
    bstar: Option<XMLValue>,
    #[serde(rename = "BTERM")]
    bterm: Option<XMLValue>,
    #[serde(rename = "MEAN_MOTION_DOT")]
    mean_motion_dot: Option<XMLValue>,
    #[serde(rename = "MEAN_MOTION_DDOT")]
    mean_motion_ddot: Option<XMLValue>,
    #[serde(rename = "AGOM")]
    agom: Option<XMLValue>,
    #[serde(rename = "COMMENT", default)]
    comments: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct XMLSpacecraftParameters {
    #[serde(rename = "MASS")]
    mass: Option<XMLValue>,
    #[serde(rename = "SOLAR_RAD_AREA")]
    solar_rad_area: Option<XMLValue>,
    #[serde(rename = "SOLAR_RAD_COEFF")]
    solar_rad_coeff: Option<XMLValue>,
    #[serde(rename = "DRAG_AREA")]
    drag_area: Option<XMLValue>,
    #[serde(rename = "DRAG_COEFF")]
    drag_coeff: Option<XMLValue>,
    #[serde(rename = "COMMENT", default)]
    comments: Vec<String>,
}

fn convert_xml_spacecraft_params(
    xml_sp: &XMLSpacecraftParameters,
) -> Result<crate::ccsds::common::CCSDSSpacecraftParameters, BraheError> {
    Ok(crate::ccsds::common::CCSDSSpacecraftParameters {
        mass: xml_sp.mass.as_ref().map(|v| v.parse_f64()).transpose()?,
        solar_rad_area: xml_sp
            .solar_rad_area
            .as_ref()
            .map(|v| v.parse_f64())
            .transpose()?,
        solar_rad_coeff: xml_sp
            .solar_rad_coeff
            .as_ref()
            .map(|v| v.parse_f64())
            .transpose()?,
        drag_area: xml_sp
            .drag_area
            .as_ref()
            .map(|v| v.parse_f64())
            .transpose()?,
        drag_coeff: xml_sp
            .drag_coeff
            .as_ref()
            .map(|v| v.parse_f64())
            .transpose()?,
        comments: xml_sp
            .comments
            .iter()
            .map(|s| s.trim().to_string())
            .collect(),
    })
}

// ============================================================================
// OMM XML Parser
// ============================================================================

/// Parse an OMM message from XML format.
pub fn parse_omm_xml(content: &str) -> Result<crate::ccsds::omm::OMM, BraheError> {
    use crate::ccsds::omm::*;

    let xml_omm: XMLOMM = quick_xml::de::from_str(content)
        .map_err(|e| ccsds_parse_error("OMM", &format!("XML parse error: {}", e)))?;

    let format_version = xml_omm
        .version
        .as_ref()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(3.0);

    let creation_date_str = xml_omm
        .header
        .creation_date()
        .ok_or_else(|| ccsds_parse_error("OMM", "missing CREATION_DATE in header"))?;
    let originator = xml_omm
        .header
        .originator()
        .ok_or_else(|| ccsds_parse_error("OMM", "missing ORIGINATOR in header"))?
        .to_string();

    let header = ODMHeader {
        format_version,
        classification: xml_omm.header.classification(),
        creation_date: parse_ccsds_datetime(creation_date_str, &CCSDSTimeSystem::UTC)?,
        originator,
        message_id: xml_omm.header.message_id(),
        comments: xml_omm.header.comments(),
    };

    let meta = &xml_omm.body.segment.metadata;
    let time_system_str = meta
        .find_str("TIME_SYSTEM")
        .ok_or_else(|| ccsds_parse_error("OMM", "missing TIME_SYSTEM in metadata"))?;
    let time_system = CCSDSTimeSystem::parse(time_system_str)?;

    let ref_frame_epoch = meta
        .find_str("REF_FRAME_EPOCH")
        .map(|s| parse_ccsds_datetime(s, &time_system))
        .transpose()?;

    let metadata = OMMMetadata {
        object_name: meta
            .find_str("OBJECT_NAME")
            .ok_or_else(|| ccsds_parse_error("OMM", "missing OBJECT_NAME"))?
            .to_string(),
        object_id: meta.find_str("OBJECT_ID").unwrap_or("").to_string(),
        center_name: meta
            .find_str("CENTER_NAME")
            .ok_or_else(|| ccsds_parse_error("OMM", "missing CENTER_NAME"))?
            .to_string(),
        ref_frame: CCSDSRefFrame::parse(
            meta.find_str("REF_FRAME")
                .ok_or_else(|| ccsds_parse_error("OMM", "missing REF_FRAME"))?,
        ),
        ref_frame_epoch,
        time_system: time_system.clone(),
        mean_element_theory: meta
            .find_str("MEAN_ELEMENT_THEORY")
            .ok_or_else(|| ccsds_parse_error("OMM", "missing MEAN_ELEMENT_THEORY"))?
            .to_string(),
        comments: meta.comments(),
    };

    let me = &xml_omm.body.segment.data.mean_elements;
    let epoch = parse_ccsds_datetime(&me.epoch, &time_system)?;

    let mean_elements = OMMeanElements {
        epoch,
        mean_motion: me.mean_motion.as_ref().map(|v| v.parse_f64()).transpose()?,
        semi_major_axis: me
            .semi_major_axis
            .as_ref()
            .map(|v| v.parse_f64())
            .transpose()?,
        eccentricity: me.eccentricity.parse_f64()?,
        inclination: me.inclination.parse_f64()?,
        ra_of_asc_node: me.ra_of_asc_node.parse_f64()?,
        arg_of_pericenter: me.arg_of_pericenter.parse_f64()?,
        mean_anomaly: me.mean_anomaly.parse_f64()?,
        // GM: km³/s² → m³/s²
        gm: me
            .gm
            .as_ref()
            .map(|v| v.parse_f64().map(|g| g * 1e9))
            .transpose()?,
        comments: me.comments.iter().map(|s| s.trim().to_string()).collect(),
    };

    let tle_parameters = xml_omm
        .body
        .segment
        .data
        .tle_parameters
        .as_ref()
        .map(|tle| -> Result<OMMTleParameters, BraheError> {
            Ok(OMMTleParameters {
                ephemeris_type: tle
                    .ephemeris_type
                    .as_ref()
                    .map(|v| v.parse_f64().map(|f| f as u32))
                    .transpose()?,
                classification_type: tle
                    .classification_type
                    .as_ref()
                    .and_then(|v| v.value.trim().chars().next()),
                norad_cat_id: tle
                    .norad_cat_id
                    .as_ref()
                    .map(|v| v.parse_f64().map(|f| f as u32))
                    .transpose()?,
                element_set_no: tle
                    .element_set_no
                    .as_ref()
                    .map(|v| v.parse_f64().map(|f| f as u32))
                    .transpose()?,
                rev_at_epoch: tle
                    .rev_at_epoch
                    .as_ref()
                    .map(|v| v.parse_f64().map(|f| f as u32))
                    .transpose()?,
                bstar: tle.bstar.as_ref().map(|v| v.parse_f64()).transpose()?,
                bterm: tle.bterm.as_ref().map(|v| v.parse_f64()).transpose()?,
                mean_motion_dot: tle
                    .mean_motion_dot
                    .as_ref()
                    .map(|v| v.parse_f64())
                    .transpose()?,
                mean_motion_ddot: tle
                    .mean_motion_ddot
                    .as_ref()
                    .map(|v| v.parse_f64())
                    .transpose()?,
                agom: tle.agom.as_ref().map(|v| v.parse_f64()).transpose()?,
                comments: tle.comments.iter().map(|s| s.trim().to_string()).collect(),
            })
        })
        .transpose()?;

    let spacecraft_parameters = xml_omm
        .body
        .segment
        .data
        .spacecraft_parameters
        .as_ref()
        .map(convert_xml_spacecraft_params)
        .transpose()?;

    let covariance = xml_omm
        .body
        .segment
        .data
        .covariance_matrix
        .as_ref()
        .map(|c| convert_xml_covariance(c, &time_system))
        .transpose()?;

    Ok(OMM {
        header,
        metadata,
        mean_elements,
        tle_parameters,
        spacecraft_parameters,
        covariance,
        user_defined: extract_xml_user_defined(content),
        comments: Vec::new(),
    })
}

// ============================================================================
// Intermediate XML structs for OPM
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(rename = "opm")]
#[allow(clippy::upper_case_acronyms)]
struct XMLOPM {
    #[serde(rename = "@version")]
    version: Option<String>,
    header: XMLHeader,
    body: XMLOPMBody,
}

#[derive(Debug, Deserialize)]
struct XMLOPMBody {
    segment: XMLOPMSegment,
}

#[derive(Debug, Deserialize)]
struct XMLOPMSegment {
    metadata: XMLOPMMetadata,
    #[serde(default)]
    data: Option<XMLOPMData>,
}

#[derive(Debug, Deserialize)]
struct XMLOPMMetadata {
    #[serde(rename = "$value")]
    items: Vec<XMLOPMMetadataItem>,
}

#[derive(Debug, Deserialize)]
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
enum XMLOPMMetadataItem {
    OBJECT_NAME(String),
    OBJECT_ID(String),
    CENTER_NAME(String),
    REF_FRAME(String),
    REF_FRAME_EPOCH(String),
    TIME_SYSTEM(String),
    COMMENT(String),
}

impl XMLOPMMetadata {
    fn find_str(&self, variant: &str) -> Option<&str> {
        self.items.iter().find_map(|item| match item {
            XMLOPMMetadataItem::OBJECT_NAME(s) if variant == "OBJECT_NAME" => Some(s.as_str()),
            XMLOPMMetadataItem::OBJECT_ID(s) if variant == "OBJECT_ID" => Some(s.as_str()),
            XMLOPMMetadataItem::CENTER_NAME(s) if variant == "CENTER_NAME" => Some(s.as_str()),
            XMLOPMMetadataItem::REF_FRAME(s) if variant == "REF_FRAME" => Some(s.as_str()),
            XMLOPMMetadataItem::REF_FRAME_EPOCH(s) if variant == "REF_FRAME_EPOCH" => {
                Some(s.as_str())
            }
            XMLOPMMetadataItem::TIME_SYSTEM(s) if variant == "TIME_SYSTEM" => Some(s.as_str()),
            _ => None,
        })
    }

    fn comments(&self) -> Vec<String> {
        self.items
            .iter()
            .filter_map(|item| {
                if let XMLOPMMetadataItem::COMMENT(s) = item {
                    Some(s.trim().to_string())
                } else {
                    None
                }
            })
            .collect()
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct XMLOPMData {
    state_vector: XMLOPMStateVector,
    #[serde(default)]
    keplerian_elements: Option<XMLKeplerianElements>,
    #[serde(default)]
    spacecraft_parameters: Option<XMLSpacecraftParameters>,
    #[serde(default)]
    covariance_matrix: Option<XMLCovarianceMatrix>,
    #[serde(default, rename = "maneuverParameters")]
    maneuver_parameters: Vec<XMLManeuverParameters>,
}

#[derive(Debug, Deserialize)]
struct XMLOPMStateVector {
    #[serde(rename = "EPOCH")]
    epoch: String,
    #[serde(rename = "X")]
    x: XMLValue,
    #[serde(rename = "Y")]
    y: XMLValue,
    #[serde(rename = "Z")]
    z: XMLValue,
    #[serde(rename = "X_DOT")]
    x_dot: XMLValue,
    #[serde(rename = "Y_DOT")]
    y_dot: XMLValue,
    #[serde(rename = "Z_DOT")]
    z_dot: XMLValue,
    #[serde(rename = "COMMENT", default)]
    comments: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct XMLKeplerianElements {
    #[serde(rename = "SEMI_MAJOR_AXIS")]
    semi_major_axis: XMLValue,
    #[serde(rename = "ECCENTRICITY")]
    eccentricity: XMLValue,
    #[serde(rename = "INCLINATION")]
    inclination: XMLValue,
    #[serde(rename = "RA_OF_ASC_NODE")]
    ra_of_asc_node: XMLValue,
    #[serde(rename = "ARG_OF_PERICENTER")]
    arg_of_pericenter: XMLValue,
    #[serde(rename = "TRUE_ANOMALY")]
    true_anomaly: Option<XMLValue>,
    #[serde(rename = "MEAN_ANOMALY")]
    mean_anomaly: Option<XMLValue>,
    #[serde(rename = "GM")]
    gm: Option<XMLValue>,
    #[serde(rename = "COMMENT", default)]
    comments: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct XMLManeuverParameters {
    #[serde(rename = "MAN_EPOCH_IGNITION")]
    epoch_ignition: String,
    #[serde(rename = "MAN_DURATION")]
    duration: XMLValue,
    #[serde(rename = "MAN_DELTA_MASS")]
    delta_mass: Option<XMLValue>,
    #[serde(rename = "MAN_REF_FRAME")]
    ref_frame: String,
    #[serde(rename = "MAN_DV_1")]
    dv_1: XMLValue,
    #[serde(rename = "MAN_DV_2")]
    dv_2: XMLValue,
    #[serde(rename = "MAN_DV_3")]
    dv_3: XMLValue,
    #[serde(rename = "COMMENT", default)]
    comments: Vec<String>,
}

// ============================================================================
// OPM XML Parser
// ============================================================================

/// Parse an OPM message from XML format.
pub fn parse_opm_xml(content: &str) -> Result<crate::ccsds::opm::OPM, BraheError> {
    use crate::ccsds::opm::*;

    let xml_opm: XMLOPM = quick_xml::de::from_str(content)
        .map_err(|e| ccsds_parse_error("OPM", &format!("XML parse error: {}", e)))?;

    let format_version = xml_opm
        .version
        .as_ref()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(3.0);

    let creation_date_str = xml_opm
        .header
        .creation_date()
        .ok_or_else(|| ccsds_parse_error("OPM", "missing CREATION_DATE in header"))?;
    let originator = xml_opm
        .header
        .originator()
        .ok_or_else(|| ccsds_parse_error("OPM", "missing ORIGINATOR in header"))?
        .to_string();

    let header = ODMHeader {
        format_version,
        classification: xml_opm.header.classification(),
        creation_date: parse_ccsds_datetime(creation_date_str, &CCSDSTimeSystem::UTC)?,
        originator,
        message_id: xml_opm.header.message_id(),
        comments: xml_opm.header.comments(),
    };

    let meta = &xml_opm.body.segment.metadata;

    let time_system_str = meta
        .find_str("TIME_SYSTEM")
        .ok_or_else(|| ccsds_parse_error("OPM", "missing TIME_SYSTEM in metadata"))?;
    let time_system = CCSDSTimeSystem::parse(time_system_str)?;

    let ref_frame_epoch = meta
        .find_str("REF_FRAME_EPOCH")
        .map(|s| parse_ccsds_datetime(s, &time_system))
        .transpose()?;

    let metadata = OPMMetadata {
        object_name: meta
            .find_str("OBJECT_NAME")
            .ok_or_else(|| ccsds_parse_error("OPM", "missing OBJECT_NAME"))?
            .to_string(),
        object_id: meta
            .find_str("OBJECT_ID")
            .ok_or_else(|| ccsds_parse_error("OPM", "missing OBJECT_ID"))?
            .to_string(),
        center_name: meta
            .find_str("CENTER_NAME")
            .ok_or_else(|| ccsds_parse_error("OPM", "missing CENTER_NAME"))?
            .to_string(),
        ref_frame: CCSDSRefFrame::parse(
            meta.find_str("REF_FRAME")
                .ok_or_else(|| ccsds_parse_error("OPM", "missing REF_FRAME"))?,
        ),
        ref_frame_epoch,
        time_system: time_system.clone(),
        comments: meta.comments(),
    };

    // If data block is missing (e.g., spurious-metadata test), return minimal OPM
    let data = match xml_opm.body.segment.data {
        Some(ref d) => d,
        None => {
            return Err(ccsds_parse_error("OPM", "missing data block"));
        }
    };

    let sv = &data.state_vector;
    let epoch = parse_ccsds_datetime(&sv.epoch, &time_system)?;

    // Position: km → m
    let state_vector = OPMStateVector {
        epoch,
        position: [
            sv.x.parse_f64()? * 1e3,
            sv.y.parse_f64()? * 1e3,
            sv.z.parse_f64()? * 1e3,
        ],
        velocity: [
            sv.x_dot.parse_f64()? * 1e3,
            sv.y_dot.parse_f64()? * 1e3,
            sv.z_dot.parse_f64()? * 1e3,
        ],
        comments: sv.comments.iter().map(|s| s.trim().to_string()).collect(),
    };

    // Keplerian elements
    let keplerian_elements = data
        .keplerian_elements
        .as_ref()
        .map(|ke| -> Result<OPMKeplerianElements, BraheError> {
            Ok(OPMKeplerianElements {
                // km → m
                semi_major_axis: ke.semi_major_axis.parse_f64()? * 1e3,
                eccentricity: ke.eccentricity.parse_f64()?,
                inclination: ke.inclination.parse_f64()?,
                ra_of_asc_node: ke.ra_of_asc_node.parse_f64()?,
                arg_of_pericenter: ke.arg_of_pericenter.parse_f64()?,
                true_anomaly: ke
                    .true_anomaly
                    .as_ref()
                    .map(|v| v.parse_f64())
                    .transpose()?,
                mean_anomaly: ke
                    .mean_anomaly
                    .as_ref()
                    .map(|v| v.parse_f64())
                    .transpose()?,
                // km³/s² → m³/s²
                gm: ke
                    .gm
                    .as_ref()
                    .map(|v| v.parse_f64().map(|g| g * 1e9))
                    .transpose()?,
                comments: ke.comments.iter().map(|s| s.trim().to_string()).collect(),
            })
        })
        .transpose()?;

    // Spacecraft parameters
    let spacecraft_parameters = data
        .spacecraft_parameters
        .as_ref()
        .map(convert_xml_spacecraft_params)
        .transpose()?;

    // Covariance
    let covariance = data
        .covariance_matrix
        .as_ref()
        .map(|c| convert_xml_covariance(c, &time_system))
        .transpose()?;

    // Maneuvers
    let mut maneuvers = Vec::new();
    for man in &data.maneuver_parameters {
        let epoch_ignition = parse_ccsds_datetime(&man.epoch_ignition, &time_system)?;
        maneuvers.push(OPMManeuver {
            epoch_ignition,
            duration: man.duration.parse_f64()?,
            delta_mass: man.delta_mass.as_ref().map(|v| v.parse_f64()).transpose()?,
            ref_frame: CCSDSRefFrame::parse(&man.ref_frame),
            // km/s → m/s
            dv: [
                man.dv_1.parse_f64()? * 1e3,
                man.dv_2.parse_f64()? * 1e3,
                man.dv_3.parse_f64()? * 1e3,
            ],
            comments: man.comments.iter().map(|s| s.trim().to_string()).collect(),
        });
    }

    Ok(OPM {
        header,
        metadata,
        state_vector,
        keplerian_elements,
        spacecraft_parameters,
        covariance,
        maneuvers,
        user_defined: extract_xml_user_defined(content),
    })
}

/// Extract user-defined parameters from XML content.
///
/// Scans for `<USER_DEFINED_xxx value="yyy"/>` elements inside
/// `<userDefinedParameters>` blocks and returns them as a `CCSDSUserDefined`.
fn extract_xml_user_defined(content: &str) -> Option<CCSDSUserDefined> {
    use quick_xml::Reader;
    use quick_xml::events::Event;

    let mut reader = Reader::from_str(content);
    let mut in_user_defined = false;
    let mut params: HashMap<String, String> = HashMap::new();

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                if name == "userDefinedParameters" {
                    in_user_defined = true;
                } else if in_user_defined && let Some(key) = name.strip_prefix("USER_DEFINED_") {
                    for attr in e.attributes().flatten() {
                        let attr_name = String::from_utf8_lossy(attr.key.as_ref()).to_string();
                        if attr_name == "value"
                            && let Ok(val) =
                                attr.normalized_value(quick_xml::XmlVersion::Explicit1_0)
                        {
                            params.insert(key.to_string(), val.to_string());
                        }
                    }
                }
            }
            Ok(Event::End(e)) => {
                let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                if name == "userDefinedParameters" {
                    in_user_defined = false;
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }

    if params.is_empty() {
        None
    } else {
        Some(CCSDSUserDefined { parameters: params })
    }
}

/// Emit the KVN line for a CDM XML element that has just closed.
///
/// Structural container elements carry no value of their own, and the text
/// accumulated inside them is only inter-element whitespace, so they are
/// skipped along with any element whose text is empty.
///
/// # Arguments
///
/// * `lines` - The KVN line buffer being built.
/// * `tag` - The name of the element that just closed.
/// * `text` - The character data accumulated inside it.
///
/// # Returns
///
/// Nothing; `lines` is extended in place when the element carries a value.
///
/// # Examples
///
/// ```ignore
/// let mut lines = Vec::new();
/// push_cdm_kvn_line(&mut lines, "ORIGINATOR", " JSPOC ");
/// push_cdm_kvn_line(&mut lines, "COMMENT", "a note");
/// push_cdm_kvn_line(&mut lines, "segment", "\n  ");
/// assert_eq!(lines, ["ORIGINATOR = JSPOC", "COMMENT a note"]);
/// ```
fn push_cdm_kvn_line(lines: &mut Vec<String>, tag: &str, text: &str) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }

    match tag {
        "header"
        | "body"
        | "relativeMetadataData"
        | "segment"
        | "metadata"
        | "data"
        | "odParameters"
        | "additionalParameters"
        | "stateVector"
        | "covarianceMatrix"
        | "relativeStateVector"
        | "additionalCovarianceMetadata"
        | "userDefinedParameters"
        | "cdm" => {}
        "COMMENT" => lines.push(format!("COMMENT {}", text)),
        tag if tag.starts_with("USER_DEFINED_") || tag.starts_with(|c: char| c.is_uppercase()) => {
            lines.push(format!("{} = {}", tag, text))
        }
        // camelCase tags for sub-blocks carry no value.
        _ => {}
    }
}

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
            man = man.with_delta_mass(dm.parse_f64()?);
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

/// Parse a CDM message from XML format.
///
/// Converts XML to KVN-like key=value representation, then delegates to the
/// KVN parser. This ensures full feature parity between KVN and XML parsing
/// without duplicating the field-by-field dispatch logic.
pub fn parse_cdm_xml(content: &str) -> Result<crate::ccsds::cdm::CDM, BraheError> {
    use quick_xml::Reader;
    use quick_xml::events::Event;

    let mut reader = Reader::from_str(content);
    let mut kvn_lines: Vec<String> = Vec::new();
    let mut tag_stack: Vec<String> = Vec::new();
    let mut current_tag = String::new();
    // quick-xml reports an entity reference as its own event, splitting the
    // character data of an element into several `Text` events. Element text is
    // therefore accumulated here and consumed when the element closes.
    let mut text_buf = String::new();

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => {
                let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                tag_stack.push(name.clone());
                current_tag = name.clone();
                text_buf.clear();

                // Handle cdm root element version attribute
                if name == "cdm" {
                    for attr in e.attributes().flatten() {
                        let attr_name = String::from_utf8_lossy(attr.key.as_ref()).to_string();
                        if attr_name == "version" {
                            let val = String::from_utf8_lossy(&attr.value).to_string();
                            kvn_lines.push(format!("CCSDS_CDM_VERS = {}", val));
                        }
                    }
                }
            }
            Ok(Event::End(_e)) => {
                push_cdm_kvn_line(&mut kvn_lines, &current_tag, &text_buf);
                text_buf.clear();
                tag_stack.pop();
                current_tag = tag_stack.last().cloned().unwrap_or_default();
            }
            Ok(Event::Empty(e)) => {
                // Handle self-closing elements like <FIELD nil="true"/>
                let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                // Check for nil="true" — skip these
                let mut is_nil = false;
                for attr in e.attributes().flatten() {
                    let attr_name = String::from_utf8_lossy(attr.key.as_ref()).to_string();
                    if attr_name == "nil" {
                        let val = String::from_utf8_lossy(&attr.value).to_string();
                        if val == "true" {
                            is_nil = true;
                        }
                    }
                }
                if !is_nil {
                    // Check for USER_DEFINED_* elements with value attribute
                    if name.starts_with("USER_DEFINED_") {
                        let mut val = String::new();
                        for attr in e.attributes().flatten() {
                            let attr_name = String::from_utf8_lossy(attr.key.as_ref()).to_string();
                            if attr_name == "value" {
                                val = attr
                                    .normalized_value(quick_xml::XmlVersion::Explicit1_0)
                                    .map_err(|err| {
                                        ccsds_parse_error(
                                            "CDM",
                                            &format!("XML attribute decode error: {}", err),
                                        )
                                    })?
                                    .to_string();
                            }
                        }
                        kvn_lines.push(format!("{} = {}", name, val));
                    } else if name.starts_with(|c: char| c.is_uppercase()) && name != "COMMENT" {
                        // Empty element with no nil attribute - treat as empty value
                        kvn_lines.push(format!("{} = ", name));
                    }
                }
            }
            Ok(Event::Text(e)) => {
                let decoded = e.decode().map_err(|err| {
                    ccsds_parse_error("CDM", &format!("XML text decode error: {}", err))
                })?;
                text_buf.push_str(&decoded);
            }
            Ok(Event::CData(e)) => {
                // A CDATA section is character data too, and carries markup
                // characters without escaping them.
                let decoded = e.decode().map_err(|err| {
                    ccsds_parse_error("CDM", &format!("XML text decode error: {}", err))
                })?;
                text_buf.push_str(&decoded);
            }
            Ok(Event::GeneralRef(e)) => {
                let name = e.decode().map_err(|err| {
                    ccsds_parse_error("CDM", &format!("XML text decode error: {}", err))
                })?;
                let reference = format!("&{};", name);
                let resolved = quick_xml::escape::unescape(&reference).map_err(|err| {
                    ccsds_parse_error("CDM", &format!("XML entity decode error: {}", err))
                })?;
                text_buf.push_str(&resolved);
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(ccsds_parse_error("CDM", &format!("XML parse error: {}", e)));
            }
            _ => {}
        }
    }

    // Now parse the generated KVN representation
    let kvn_content = kvn_lines.join("\n");
    crate::ccsds::kvn::parse_cdm(&kvn_content)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use serial_test::parallel;

    #[test]
    #[parallel]
    fn test_parse_oem_xml_example3() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample3.xml").unwrap();
        let oem = parse_oem_xml(&content).unwrap();

        // Header
        assert!((oem.header.format_version - 3.0).abs() < 1e-10);
        assert_eq!(oem.header.originator, "NASA/JPL");
        assert_eq!(oem.header.message_id.as_deref(), Some("OEM 201113719185"));

        // Header comment
        assert_eq!(oem.header.comments.len(), 1);
        assert!(oem.header.comments[0].contains("OEM WITH OPTIONAL ACCELERATIONS"));

        // 1 segment
        assert_eq!(oem.segments.len(), 1);

        let seg = &oem.segments[0];
        assert_eq!(seg.metadata.object_name, "MARS GLOBAL SURVEYOR");
        assert_eq!(seg.metadata.object_id, "2000-028A");
        assert_eq!(seg.metadata.center_name, "MARS BARYCENTER");
        assert_eq!(seg.metadata.ref_frame, CCSDSRefFrame::J2000);
        assert_eq!(seg.metadata.interpolation.as_deref(), Some("HERMITE"));
        assert_eq!(seg.metadata.interpolation_degree, Some(7));

        // 4 state vectors with accelerations
        assert_eq!(seg.states.len(), 4);

        // First state: X=2789.6 km → 2789600.0 m
        assert!((seg.states[0].position[0] - 2789600.0).abs() < 1.0);
        assert!((seg.states[0].velocity[0] - 4730.0).abs() < 1.0);

        // Accelerations present
        assert!(seg.states[0].acceleration.is_some());
        let acc = seg.states[0].acceleration.unwrap();
        assert!((acc[0] - 8.0).abs() < 0.1); // 0.008 km/s² = 8.0 m/s²

        // Data block comments
        assert_eq!(seg.comments.len(), 2);

        // Covariance
        assert_eq!(seg.covariances.len(), 1);
        let cov = &seg.covariances[0];
        assert!(cov.epoch.is_some());
        assert_eq!(cov.cov_ref_frame.as_ref().unwrap(), &CCSDSRefFrame::ITRF97);
        // CX_X = 0.316 km² = 316000 m²
        assert!((cov.matrix[(0, 0)] - 0.316 * 1e6).abs() < 1.0);
    }

    #[test]
    #[parallel]
    fn test_parse_oem_xml_multiple_comments_per_block() {
        let oem = parse_oem_xml(
            &std::fs::read_to_string("test_assets/ccsds/oem/OEM-multiple-comments.xml").unwrap(),
        )
        .unwrap();

        assert_eq!(
            oem.header.comments,
            vec!["first header comment", "second header comment"]
        );

        let seg = &oem.segments[0];
        assert_eq!(
            seg.metadata.comments,
            vec!["first metadata comment", "second metadata comment"]
        );
        assert_eq!(
            seg.comments,
            vec!["first data comment", "second data comment"]
        );
        assert_eq!(
            seg.covariances[0].comments,
            vec!["first covariance comment", "second covariance comment"]
        );
    }

    #[test]
    #[parallel]
    fn test_parse_omm_xml_multiple_comments_per_block() {
        let omm = parse_omm_xml(
            &std::fs::read_to_string("test_assets/ccsds/omm/OMM-multiple-comments.xml").unwrap(),
        )
        .unwrap();

        assert_eq!(
            omm.header.comments,
            vec!["first header comment", "second header comment"]
        );
        assert_eq!(
            omm.metadata.comments,
            vec!["first metadata comment", "second metadata comment"]
        );
        assert_eq!(
            omm.mean_elements.comments,
            vec!["first mean-element comment", "second mean-element comment"]
        );
        assert_eq!(
            omm.tle_parameters.as_ref().unwrap().comments,
            vec!["first TLE comment", "second TLE comment"]
        );
        assert_eq!(
            omm.spacecraft_parameters.as_ref().unwrap().comments,
            vec!["first spacecraft comment", "second spacecraft comment"]
        );
    }

    #[test]
    #[parallel]
    fn test_parse_opm_xml_multiple_comments_per_block() {
        let opm = parse_opm_xml(
            &std::fs::read_to_string("test_assets/ccsds/opm/OPM-multiple-comments.xml").unwrap(),
        )
        .unwrap();

        assert_eq!(
            opm.header.comments,
            vec!["first header comment", "second header comment"]
        );
        assert_eq!(
            opm.metadata.comments,
            vec!["first metadata comment", "second metadata comment"]
        );
        assert_eq!(
            opm.state_vector.comments,
            vec!["first state-vector comment", "second state-vector comment"]
        );
        assert_eq!(
            opm.keplerian_elements.as_ref().unwrap().comments,
            vec!["first Keplerian comment", "second Keplerian comment"]
        );
        assert_eq!(
            opm.maneuvers[0].comments,
            vec!["first maneuver comment", "second maneuver comment"]
        );
    }

    #[test]
    #[parallel]
    fn test_parse_oem_xml_multiple_segments() {
        let oem = parse_oem_xml(
            &std::fs::read_to_string("test_assets/ccsds/oem/OEM-two-segments.xml").unwrap(),
        )
        .unwrap();

        assert_eq!(oem.segments.len(), 2);
        assert_eq!(oem.segments[0].states.len(), 1);
        assert_eq!(oem.segments[1].states.len(), 1);
        assert!((oem.segments[1].states[0].position[0] + 2432200.0).abs() < 1.0);
    }

    #[test]
    #[parallel]
    fn test_parse_opm_xml_multiple_maneuvers() {
        let opm = parse_opm_xml(
            &std::fs::read_to_string("test_assets/ccsds/opm/OPM-two-maneuvers.xml").unwrap(),
        )
        .unwrap();

        assert_eq!(opm.maneuvers.len(), 2);
        assert!((opm.maneuvers[0].duration - 300.0).abs() < 1e-10);
        assert!((opm.maneuvers[1].duration - 150.0).abs() < 1e-10);
        assert!((opm.maneuvers[1].dv[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    #[parallel]
    fn test_parse_cdm_xml_resolves_entity_references() {
        // quick-xml reports each entity reference as its own event, so element
        // text that contains one arrives in several pieces. A conforming CDM
        // must escape `&` and `<` in free text, so this is the ordinary shape
        // of a third-party file, not an edge case.
        let content = std::fs::read_to_string("test_assets/ccsds/cdm/CDMExample1.xml").unwrap();
        let content = content.replace(
            "<ORIGINATOR>JSPOC</ORIGINATOR>",
            "<ORIGINATOR>R&amp;D &lt;ops&gt;</ORIGINATOR>",
        );
        let content = content.replace(
            "<COMMENT>Sample CDM - XML version</COMMENT>",
            "<COMMENT>Sample CDM &amp; more</COMMENT>",
        );

        let cdm = parse_cdm_xml(&content).unwrap();
        assert_eq!(cdm.header.originator, "R&D <ops>");
        assert_eq!(cdm.header.comments[0], "Sample CDM & more");
    }

    #[test]
    #[parallel]
    fn test_parse_cdm_xml_reads_cdata_sections() {
        // A CDATA section is character data that carries markup characters
        // unescaped; quick-xml reports it as its own event.
        let content = std::fs::read_to_string("test_assets/ccsds/cdm/CDMExample1.xml").unwrap();
        let content = content.replace(
            "<ORIGINATOR>JSPOC</ORIGINATOR>",
            "<ORIGINATOR><![CDATA[R&D <ops>]]></ORIGINATOR>",
        );

        let cdm = parse_cdm_xml(&content).unwrap();
        assert_eq!(cdm.header.originator, "R&D <ops>");
    }

    #[test]
    #[parallel]
    fn test_parse_cdm_xml_unescapes_user_defined_attribute() {
        let content = std::fs::read_to_string("test_assets/ccsds/cdm/CDMExample1.xml").unwrap();
        let content = content.replace(
            "</body>",
            "<userDefinedParameters>\
             <USER_DEFINED_EARTH_MODEL value=\"R&amp;D &quot;x&quot;\"/>\
             </userDefinedParameters></body>",
        );

        let cdm = parse_cdm_xml(&content).unwrap();
        assert_eq!(
            cdm.user_defined.unwrap().parameters["EARTH_MODEL"],
            "R&D \"x\""
        );
    }

    // ------------------------------------------------------------------
    // APM
    // ------------------------------------------------------------------

    use crate::ccsds::xml::write_apm_xml;
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
