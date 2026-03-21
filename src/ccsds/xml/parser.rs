/*!
 * XML parser for CCSDS OEM, OMM, and OPM messages.
 *
 * Uses `quick-xml` with serde to deserialize into intermediate structs,
 * then converts to the public CCSDS types with unit conversion.
 */

use std::collections::HashMap;

use serde::Deserialize;

use crate::ccsds::common::{
    CCSDSCovariance, CCSDSRefFrame, CCSDSTimeSystem, CCSDSUserDefined, ODMHeader,
    covariance_from_lower_triangular, parse_ccsds_datetime,
};
use crate::ccsds::error::ccsds_parse_error;
use crate::ccsds::oem::{OEM, OEMMetadata, OEMSegment, OEMStateVector};
use crate::utils::errors::BraheError;

// ============================================================================
// Serde helpers
// ============================================================================

/// Deserialize a COMMENT field that may be a single string or a Vec<String>.
fn deserialize_comments<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct CommentsVisitor;

    impl<'de> serde::de::Visitor<'de> for CommentsVisitor {
        type Value = Vec<String>;

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a string or sequence of strings")
        }

        fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
            Ok(vec![v.to_string()])
        }

        fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> {
            Ok(vec![v])
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut vec = Vec::new();
            while let Some(item) = seq.next_element::<String>()? {
                vec.push(item);
            }
            Ok(vec)
        }

        fn visit_none<E: serde::de::Error>(self) -> Result<Self::Value, E> {
            Ok(Vec::new())
        }

        fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
            Ok(Vec::new())
        }

        fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
        where
            M: serde::de::MapAccess<'de>,
        {
            // Handle case where quick-xml wraps the text in a map like {"$text": "..."}
            let mut result = Vec::new();
            while let Some((key, value)) = map.next_entry::<String, String>()? {
                if key == "$text" || key == "$value" {
                    result.push(value);
                }
            }
            Ok(result)
        }
    }

    deserializer.deserialize_any(CommentsVisitor)
}

/// Deserialize a field that may be a single struct or a sequence of structs.
fn deserialize_one_or_many<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct OneOrManyVisitor<T>(std::marker::PhantomData<T>);

    impl<'de, T: Deserialize<'de>> serde::de::Visitor<'de> for OneOrManyVisitor<T> {
        type Value = Vec<T>;

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("one or many elements")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut vec = Vec::new();
            while let Some(item) = seq.next_element()? {
                vec.push(item);
            }
            Ok(vec)
        }

        fn visit_map<M>(self, map: M) -> Result<Self::Value, M::Error>
        where
            M: serde::de::MapAccess<'de>,
        {
            let item = T::deserialize(serde::de::value::MapAccessDeserializer::new(map))?;
            Ok(vec![item])
        }
    }

    deserializer.deserialize_any(OneOrManyVisitor(std::marker::PhantomData))
}

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
    #[serde(rename = "segment", deserialize_with = "deserialize_one_or_many")]
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
    #[serde(rename = "COMMENT", default, deserialize_with = "deserialize_comments")]
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
    #[serde(rename = "COMMENT", default, deserialize_with = "deserialize_comments")]
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
    #[serde(rename = "COMMENT", default, deserialize_with = "deserialize_comments")]
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
    #[serde(rename = "COMMENT", default, deserialize_with = "deserialize_comments")]
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
    #[serde(
        default,
        rename = "maneuverParameters",
        deserialize_with = "deserialize_one_or_many_opt"
    )]
    maneuver_parameters: Vec<XMLManeuverParameters>,
}

/// Deserialize an optional field that may be absent, a single struct, or a sequence.
fn deserialize_one_or_many_opt<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    deserialize_one_or_many(deserializer)
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
    #[serde(rename = "COMMENT", default, deserialize_with = "deserialize_comments")]
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
    #[serde(rename = "COMMENT", default, deserialize_with = "deserialize_comments")]
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
    #[serde(rename = "COMMENT", default, deserialize_with = "deserialize_comments")]
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
                        if attr_name == "value" {
                            let val = String::from_utf8_lossy(&attr.value).to_string();
                            params.insert(key.to_string(), val);
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

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => {
                let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                tag_stack.push(name.clone());
                current_tag = name.clone();

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
                                val = String::from_utf8_lossy(&attr.value).to_string();
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
                let text = e.unescape().unwrap_or_default().trim().to_string();
                if text.is_empty() {
                    continue;
                }

                // Map XML tag names to KVN keywords
                let keyword = match current_tag.as_str() {
                    // Skip structural tags that don't map to KVN keywords
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
                    | "cdm" => continue,

                    // COMMENT is handled specially
                    "COMMENT" => {
                        kvn_lines.push(format!("COMMENT {}", text));
                        continue;
                    }

                    // USER_DEFINED elements
                    tag if tag.starts_with("USER_DEFINED_") => tag,

                    // All CCSDS keyword tags map directly (uppercase tags)
                    tag if tag.starts_with(|c: char| c.is_uppercase()) => tag,

                    // camelCase tags for sub-blocks - skip
                    _ => continue,
                };

                kvn_lines.push(format!("{} = {}", keyword, text));
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

    #[test]
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
}
