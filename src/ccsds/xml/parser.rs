/*!
 * XML parser for CCSDS OEM, OMM, and OPM messages.
 *
 * Uses `quick-xml` with serde to deserialize into intermediate structs,
 * then converts to the public CCSDS types with unit conversion.
 */

use serde::Deserialize;

use crate::ccsds::common::{
    CCSDSCovariance, CCSDSRefFrame, CCSDSTimeSystem, ODMHeader, covariance_from_lower_triangular,
    parse_ccsds_datetime,
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
        classification: None,
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

/// Parse an OMM message from XML format.
pub fn parse_omm_xml(_content: &str) -> Result<crate::ccsds::omm::OMM, BraheError> {
    Err(ccsds_parse_error("OMM", "XML parsing not yet implemented"))
}

/// Parse an OPM message from XML format.
pub fn parse_opm_xml(_content: &str) -> Result<crate::ccsds::opm::OPM, BraheError> {
    Err(ccsds_parse_error("OPM", "XML parsing not yet implemented"))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    fn setup_eop() {
        use crate::eop::*;
        let eop = StaticEOPProvider::new();
        set_global_eop_provider(eop);
    }

    #[test]
    fn test_parse_oem_xml_example3() {
        setup_eop();

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
