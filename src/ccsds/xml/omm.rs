/*!
 * XML reader and writer for the Orbit Mean-elements Message (OMM).
 *
 * Reference: CCSDS 502.0-B-3 (Orbit Data Messages), section 4
 */

use serde::Deserialize;

use crate::ccsds::common::{
    CCSDSRefFrame, CCSDSTimeSystem, ODMHeader, format_ccsds_datetime_in, parse_ccsds_datetime,
};
use crate::ccsds::error::ccsds_parse_error;
use crate::ccsds::xml::common::{
    XMLCovarianceMatrix, XMLHeader, XMLSpacecraftParameters, XMLValue, convert_xml_covariance,
    convert_xml_spacecraft_params, escape_xml_text, extract_xml_user_defined,
    validate_xml_characters, write_xml_covariance, write_xml_header, write_xml_spacecraft_params,
    write_xml_user_defined,
};
use crate::utils::errors::BraheError;

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
// OMM XML Writer
// ============================================================================

/// Write an OMM message to XML format.
pub fn write_omm_xml(omm: &crate::ccsds::omm::OMM) -> Result<String, BraheError> {
    let mut out = String::new();
    let i1 = "  ";
    let i2 = "    ";
    let i3 = "      ";
    let i4 = "        ";

    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str(&format!(
        "<omm id=\"CCSDS_OMM_VERS\" version=\"{:.1}\">\n",
        omm.header.format_version
    ));

    write_xml_header(&mut out, &omm.header, i1, i2);

    out.push_str(&format!("{}<body>\n", i1));
    out.push_str(&format!("{}<segment>\n", i2));

    // Metadata
    out.push_str(&format!("{}<metadata>\n", i3));
    for c in &omm.metadata.comments {
        out.push_str(&format!(
            "{}<COMMENT>{}</COMMENT>\n",
            i4,
            escape_xml_text(c)
        ));
    }
    out.push_str(&format!(
        "{}<OBJECT_NAME>{}</OBJECT_NAME>\n",
        i4,
        escape_xml_text(&omm.metadata.object_name)
    ));
    out.push_str(&format!(
        "{}<OBJECT_ID>{}</OBJECT_ID>\n",
        i4,
        escape_xml_text(&omm.metadata.object_id)
    ));
    out.push_str(&format!(
        "{}<CENTER_NAME>{}</CENTER_NAME>\n",
        i4,
        escape_xml_text(&omm.metadata.center_name)
    ));
    out.push_str(&format!(
        "{}<REF_FRAME>{}</REF_FRAME>\n",
        i4, omm.metadata.ref_frame
    ));
    if let Some(ref e) = omm.metadata.ref_frame_epoch {
        out.push_str(&format!(
            "{}<REF_FRAME_EPOCH>{}</REF_FRAME_EPOCH>\n",
            i4,
            format_ccsds_datetime_in(e, &omm.metadata.time_system)
        ));
    }
    out.push_str(&format!(
        "{}<TIME_SYSTEM>{}</TIME_SYSTEM>\n",
        i4, omm.metadata.time_system
    ));
    out.push_str(&format!(
        "{}<MEAN_ELEMENT_THEORY>{}</MEAN_ELEMENT_THEORY>\n",
        i4,
        escape_xml_text(&omm.metadata.mean_element_theory)
    ));
    out.push_str(&format!("{}</metadata>\n", i3));

    // Data
    out.push_str(&format!("{}<data>\n", i3));

    // Mean elements
    out.push_str(&format!("{}<meanElements>\n", i4));
    for c in &omm.mean_elements.comments {
        out.push_str(&format!(" <COMMENT>{}</COMMENT>\n", escape_xml_text(c)));
    }
    out.push_str(&format!(
        "        <EPOCH>{}</EPOCH>\n",
        format_ccsds_datetime_in(&omm.mean_elements.epoch, &omm.metadata.time_system)
    ));
    if let Some(mm) = omm.mean_elements.mean_motion {
        out.push_str(&format!("        <MEAN_MOTION>{}</MEAN_MOTION>\n", mm));
    }
    if let Some(sma) = omm.mean_elements.semi_major_axis {
        out.push_str(&format!(
            "        <SEMI_MAJOR_AXIS>{}</SEMI_MAJOR_AXIS>\n",
            sma
        ));
    }
    out.push_str(&format!(
        "        <ECCENTRICITY>{}</ECCENTRICITY>\n",
        omm.mean_elements.eccentricity
    ));
    out.push_str(&format!(
        "        <INCLINATION>{}</INCLINATION>\n",
        omm.mean_elements.inclination
    ));
    out.push_str(&format!(
        "        <RA_OF_ASC_NODE>{}</RA_OF_ASC_NODE>\n",
        omm.mean_elements.ra_of_asc_node
    ));
    out.push_str(&format!(
        "        <ARG_OF_PERICENTER>{}</ARG_OF_PERICENTER>\n",
        omm.mean_elements.arg_of_pericenter
    ));
    out.push_str(&format!(
        "        <MEAN_ANOMALY>{}</MEAN_ANOMALY>\n",
        omm.mean_elements.mean_anomaly
    ));
    if let Some(gm) = omm.mean_elements.gm {
        // m³/s² → km³/s²
        out.push_str(&format!("        <GM>{}</GM>\n", gm / 1e9));
    }
    out.push_str(&format!("{}</meanElements>\n", i4));

    // TLE parameters
    if let Some(ref tle) = omm.tle_parameters {
        out.push_str(&format!("{}<tleParameters>\n", i4));
        for c in &tle.comments {
            out.push_str(&format!(" <COMMENT>{}</COMMENT>\n", escape_xml_text(c)));
        }
        if let Some(et) = tle.ephemeris_type {
            out.push_str(&format!(
                "        <EPHEMERIS_TYPE>{}</EPHEMERIS_TYPE>\n",
                et
            ));
        }
        if let Some(ct) = tle.classification_type {
            out.push_str(&format!(
                "        <CLASSIFICATION_TYPE>{}</CLASSIFICATION_TYPE>\n",
                ct
            ));
        }
        if let Some(id) = tle.norad_cat_id {
            out.push_str(&format!("        <NORAD_CAT_ID>{}</NORAD_CAT_ID>\n", id));
        }
        if let Some(esn) = tle.element_set_no {
            out.push_str(&format!(
                "        <ELEMENT_SET_NO>{}</ELEMENT_SET_NO>\n",
                esn
            ));
        }
        if let Some(rev) = tle.rev_at_epoch {
            out.push_str(&format!("        <REV_AT_EPOCH>{}</REV_AT_EPOCH>\n", rev));
        }
        if let Some(bs) = tle.bstar {
            out.push_str(&format!("        <BSTAR>{}</BSTAR>\n", bs));
        }
        if let Some(bt) = tle.bterm {
            out.push_str(&format!("        <BTERM>{}</BTERM>\n", bt));
        }
        if let Some(mmd) = tle.mean_motion_dot {
            out.push_str(&format!(
                "        <MEAN_MOTION_DOT>{}</MEAN_MOTION_DOT>\n",
                mmd
            ));
        }
        if let Some(mmdd) = tle.mean_motion_ddot {
            out.push_str(&format!(
                "        <MEAN_MOTION_DDOT>{}</MEAN_MOTION_DDOT>\n",
                mmdd
            ));
        }
        if let Some(ag) = tle.agom {
            out.push_str(&format!("        <AGOM>{}</AGOM>\n", ag));
        }
        out.push_str(&format!("{}</tleParameters>\n", i4));
    }

    // Spacecraft parameters
    if let Some(ref sp) = omm.spacecraft_parameters {
        write_xml_spacecraft_params(&mut out, sp, i4, "        ");
    }

    // Covariance
    if let Some(ref cov) = omm.covariance {
        write_xml_covariance(
            &mut out,
            cov,
            &omm.metadata.time_system,
            false,
            i4,
            "        ",
        );
    }

    out.push_str(&format!("{}</data>\n", i3));
    out.push_str(&format!("{}</segment>\n", i2));

    // User-defined parameters
    if let Some(ref ud) = omm.user_defined {
        write_xml_user_defined(&mut out, ud, i2, i3);
    }

    out.push_str(&format!("{}</body>\n", i1));
    out.push_str("</omm>\n");

    validate_xml_characters("OMM", &out)?;

    Ok(out)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {

    use crate::ccsds::xml::parse_omm_xml;

    #[test]
    #[serial_test::parallel]
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
}
