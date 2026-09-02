/*!
 * XML reader and writer for the Orbit Ephemeris Message (OEM).
 *
 * Reference: CCSDS 502.0-B-3 (Orbit Data Messages), section 5
 */

use serde::Deserialize;

use crate::ccsds::common::{
    CCSDSRefFrame, CCSDSTimeSystem, ODMHeader, format_ccsds_datetime_in, parse_ccsds_datetime,
};
use crate::ccsds::error::ccsds_parse_error;
use crate::ccsds::oem::{OEM, OEMMetadata, OEMSegment, OEMStateVector};
use crate::ccsds::xml::common::{
    XMLCovarianceMatrix, XMLHeader, XMLValue, convert_xml_covariance, escape_xml_text,
    validate_xml_characters, write_xml_covariance, write_xml_header,
};
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
// OEM XML Writer
// ============================================================================

/// Write an OEM message to XML format.
pub fn write_oem_xml(oem: &crate::ccsds::oem::OEM) -> Result<String, BraheError> {
    let mut out = String::new();
    let i1 = "  ";
    let i2 = "    ";
    let i3 = "      ";
    let i4 = "        ";

    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str(&format!(
        "<oem id=\"CCSDS_OEM_VERS\" version=\"{:.1}\">\n",
        oem.header.format_version
    ));

    write_xml_header(&mut out, &oem.header, i1, i2);

    out.push_str(&format!("{}<body>\n", i1));

    for segment in &oem.segments {
        out.push_str(&format!("{}<segment>\n", i2));

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
        out.push_str(&format!(
            "{}<CENTER_NAME>{}</CENTER_NAME>\n",
            i4,
            escape_xml_text(&segment.metadata.center_name)
        ));
        out.push_str(&format!(
            "{}<REF_FRAME>{}</REF_FRAME>\n",
            i4, segment.metadata.ref_frame
        ));
        if let Some(ref e) = segment.metadata.ref_frame_epoch {
            out.push_str(&format!(
                "{}<REF_FRAME_EPOCH>{}</REF_FRAME_EPOCH>\n",
                i4,
                format_ccsds_datetime_in(e, &segment.metadata.time_system)
            ));
        }
        out.push_str(&format!(
            "{}<TIME_SYSTEM>{}</TIME_SYSTEM>\n",
            i4, segment.metadata.time_system
        ));
        out.push_str(&format!(
            "{}<START_TIME>{}</START_TIME>\n",
            i4,
            format_ccsds_datetime_in(&segment.metadata.start_time, &segment.metadata.time_system)
        ));
        if let Some(ref t) = segment.metadata.useable_start_time {
            out.push_str(&format!(
                "{}<USEABLE_START_TIME>{}</USEABLE_START_TIME>\n",
                i4,
                format_ccsds_datetime_in(t, &segment.metadata.time_system)
            ));
        }
        if let Some(ref t) = segment.metadata.useable_stop_time {
            out.push_str(&format!(
                "{}<USEABLE_STOP_TIME>{}</USEABLE_STOP_TIME>\n",
                i4,
                format_ccsds_datetime_in(t, &segment.metadata.time_system)
            ));
        }
        out.push_str(&format!(
            "{}<STOP_TIME>{}</STOP_TIME>\n",
            i4,
            format_ccsds_datetime_in(&segment.metadata.stop_time, &segment.metadata.time_system)
        ));
        if let Some(ref interp) = segment.metadata.interpolation {
            out.push_str(&format!(
                "{}<INTERPOLATION>{}</INTERPOLATION>\n",
                i4,
                escape_xml_text(interp)
            ));
        }
        if let Some(deg) = segment.metadata.interpolation_degree {
            out.push_str(&format!(
                "{}<INTERPOLATION_DEGREE>{}</INTERPOLATION_DEGREE>\n",
                i4, deg
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

        // State vectors
        for sv in &segment.states {
            out.push_str(&format!("{}<stateVector>\n", i4));
            out.push_str(&format!(
                "        <EPOCH>{}</EPOCH>\n",
                format_ccsds_datetime_in(&sv.epoch, &segment.metadata.time_system)
            ));
            // Position: m → km
            out.push_str(&format!("        <X>{}</X>\n", sv.position[0] / 1e3));
            out.push_str(&format!("        <Y>{}</Y>\n", sv.position[1] / 1e3));
            out.push_str(&format!("        <Z>{}</Z>\n", sv.position[2] / 1e3));
            // Velocity: m/s → km/s
            out.push_str(&format!(
                "        <X_DOT>{}</X_DOT>\n",
                sv.velocity[0] / 1e3
            ));
            out.push_str(&format!(
                "        <Y_DOT>{}</Y_DOT>\n",
                sv.velocity[1] / 1e3
            ));
            out.push_str(&format!(
                "        <Z_DOT>{}</Z_DOT>\n",
                sv.velocity[2] / 1e3
            ));
            // Acceleration: m/s² → km/s²
            if let Some(ref acc) = sv.acceleration {
                out.push_str(&format!("        <X_DDOT>{}</X_DDOT>\n", acc[0] / 1e3));
                out.push_str(&format!("        <Y_DDOT>{}</Y_DDOT>\n", acc[1] / 1e3));
                out.push_str(&format!("        <Z_DDOT>{}</Z_DDOT>\n", acc[2] / 1e3));
            }
            out.push_str(&format!("{}</stateVector>\n", i4));
        }

        // Covariance
        for cov in &segment.covariances {
            write_xml_covariance(
                &mut out,
                cov,
                &segment.metadata.time_system,
                true,
                i4,
                "        ",
            );
        }

        out.push_str(&format!("{}</data>\n", i3));
        out.push_str(&format!("{}</segment>\n", i2));
    }

    out.push_str(&format!("{}</body>\n", i1));
    out.push_str("</oem>\n");

    validate_xml_characters("OEM", &out)?;

    Ok(out)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    use crate::ccsds::xml::parse_oem_xml;

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
    #[serial_test::parallel]
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
    #[serial_test::parallel]
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
}
