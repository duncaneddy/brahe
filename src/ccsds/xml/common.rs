/*!
 * Shared XML intermediate structs and writing helpers.
 *
 * The readers deserialize each message into serde intermediate structs; the
 * header, covariance, spacecraft-parameter, and user-defined blocks are
 * identical across messages and are deserialized and written from here.
 */

use std::collections::HashMap;

use serde::Deserialize;

use crate::ccsds::common::{
    CCSDSCovariance, CCSDSRefFrame, CCSDSSpacecraftParameters, CCSDSTimeSystem, CCSDSUserDefined,
    ODMHeader, covariance_from_lower_triangular, covariance_to_lower_triangular,
    format_ccsds_datetime_in, parse_ccsds_datetime, round_ccsds_value,
};
use crate::ccsds::error::ccsds_parse_error;
use crate::utils::errors::BraheError;

#[derive(Debug, Deserialize)]
pub(super) struct XMLHeader {
    #[serde(rename = "$value")]
    pub(super) items: Vec<XMLHeaderItem>,
}

#[derive(Debug, Deserialize)]
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
pub(super) enum XMLHeaderItem {
    CREATION_DATE(String),
    ORIGINATOR(String),
    MESSAGE_ID(String),
    CLASSIFICATION(String),
    COMMENT(String),
}

impl XMLHeader {
    pub(super) fn creation_date(&self) -> Option<&str> {
        self.items.iter().find_map(|item| {
            if let XMLHeaderItem::CREATION_DATE(s) = item {
                Some(s.as_str())
            } else {
                None
            }
        })
    }

    pub(super) fn originator(&self) -> Option<&str> {
        self.items.iter().find_map(|item| {
            if let XMLHeaderItem::ORIGINATOR(s) = item {
                Some(s.as_str())
            } else {
                None
            }
        })
    }

    pub(super) fn message_id(&self) -> Option<String> {
        self.items.iter().find_map(|item| {
            if let XMLHeaderItem::MESSAGE_ID(s) = item {
                Some(s.clone())
            } else {
                None
            }
        })
    }

    pub(super) fn classification(&self) -> Option<String> {
        self.items.iter().find_map(|item| {
            if let XMLHeaderItem::CLASSIFICATION(s) = item {
                Some(s.clone())
            } else {
                None
            }
        })
    }

    pub(super) fn comments(&self) -> Vec<String> {
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

/// Wrapper for XML values that may have unit attributes.
#[derive(Debug, Deserialize)]
pub(super) struct XMLValue {
    #[serde(rename = "@units")]
    pub(super) _units: Option<String>,
    #[serde(rename = "$text")]
    pub(super) value: String,
}

impl XMLValue {
    pub(super) fn parse_f64(&self) -> Result<f64, BraheError> {
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

#[derive(Debug, Deserialize)]
pub(super) struct XMLSpacecraftParameters {
    #[serde(rename = "MASS")]
    pub(super) mass: Option<XMLValue>,
    #[serde(rename = "SOLAR_RAD_AREA")]
    pub(super) solar_rad_area: Option<XMLValue>,
    #[serde(rename = "SOLAR_RAD_COEFF")]
    pub(super) solar_rad_coeff: Option<XMLValue>,
    #[serde(rename = "DRAG_AREA")]
    pub(super) drag_area: Option<XMLValue>,
    #[serde(rename = "DRAG_COEFF")]
    pub(super) drag_coeff: Option<XMLValue>,
    #[serde(rename = "COMMENT", default)]
    pub(super) comments: Vec<String>,
}

pub(super) fn convert_xml_spacecraft_params(
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

/// Extract user-defined parameters from XML content.
///
/// Scans for `<USER_DEFINED_xxx value="yyy"/>` elements inside
/// `<userDefinedParameters>` blocks and returns them as a `CCSDSUserDefined`.
pub(super) fn extract_xml_user_defined(content: &str) -> Option<CCSDSUserDefined> {
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

// ============================================================================
// Shared XML writing helpers
// ============================================================================

/// The ADM `<quaternion>` element, shared by the APM and AEM XML schemas.
#[derive(Debug, Deserialize)]
pub(super) struct XMLQuaternion {
    #[serde(rename = "Q1")]
    pub(super) q1: XMLValue,
    #[serde(rename = "Q2")]
    pub(super) q2: XMLValue,
    #[serde(rename = "Q3")]
    pub(super) q3: XMLValue,
    #[serde(rename = "QC")]
    pub(super) qc: XMLValue,
}

/// The ADM `<quaternionDot>` element, shared by the APM and AEM XML schemas.
#[derive(Debug, Deserialize)]
pub(super) struct XMLQuaternionDot {
    #[serde(rename = "Q1_DOT")]
    pub(super) q1_dot: XMLValue,
    #[serde(rename = "Q2_DOT")]
    pub(super) q2_dot: XMLValue,
    #[serde(rename = "Q3_DOT")]
    pub(super) q3_dot: XMLValue,
    #[serde(rename = "QC_DOT")]
    pub(super) qc_dot: XMLValue,
}


/// Escape the XML markup delimiters in element text content.
///
/// CCSDS 502.0-B-3 subsection 8.2 fixes the XML version declaration at 1.0 and
/// subsection 8.13.5 defines every text value as the XML Schema `string` type,
/// so free-text fields must be well-formed XML 1.0 character data. Only `&` and
/// `<` are forbidden outright in content; `>` is escaped as well because it is
/// forbidden in the `]]>` sequence and escaping it unconditionally is what the
/// XML specification recommends.
///
/// # Arguments
///
/// * `s` - The text to place between an element's tags.
///
/// # Returns
///
/// * `String` - The text with `&`, `<`, and `>` replaced by entity references.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(escape_xml_text("R&D <ops>"), "R&amp;D &lt;ops&gt;");
/// ```
pub(super) fn escape_xml_text(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Escape the XML markup delimiters in a double-quoted attribute value.
///
/// Attribute values additionally forbid the quote character that delimits them.
///
/// # Arguments
///
/// * `s` - The text to place inside a double-quoted attribute value.
///
/// # Returns
///
/// * `String` - The text with `&`, `<`, `>`, and `"` replaced by entity
///   references.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(escape_xml_attribute("a \"b\""), "a &quot;b&quot;");
/// ```
pub(super) fn escape_xml_attribute(s: &str) -> String {
    escape_xml_text(s).replace('"', "&quot;")
}

/// Test a character against the XML 1.0 `Char` production.
///
/// XML 1.0 subsection 2.2 admits `#x9`, `#xA`, `#xD`, and the ranges
/// `[#x20-#xD7FF]`, `[#xE000-#xFFFD]`, and `[#x10000-#x10FFFF]`. The remaining
/// C0 controls, `U+FFFE`, and `U+FFFF` cannot appear in a document at all, and
/// unlike the markup delimiters they cannot be rescued by escaping: the
/// numeric character reference for such a character is equally forbidden.
///
/// # Arguments
///
/// * `c` - The character to test.
///
/// # Returns
///
/// * `bool` - `true` when XML 1.0 permits the character in a document.
///
/// # Examples
///
/// ```ignore
/// assert!(is_xml_char('\t'));
/// assert!(!is_xml_char('\u{1}'));
/// ```
pub(super) fn is_xml_char(c: char) -> bool {
    matches!(c,
        '\u{9}' | '\u{A}' | '\u{D}'
        | '\u{20}'..='\u{D7FF}'
        | '\u{E000}'..='\u{FFFD}'
        | '\u{10000}'..='\u{10FFFF}'
    )
}

/// Reject a document carrying a character XML 1.0 forbids.
///
/// CCSDS 502.0-B-3 subsection 8.2 fixes the XML version at 1.0 and subsection
/// 8.13.5 defines every ODM text value as the XML Schema `string` type, so a
/// value carrying a character outside the `Char` production has no valid
/// representation and the message cannot be written. The check runs over the
/// assembled document rather than each value, which covers every emission
/// site: element content, attribute values, and the element names the
/// user-defined block builds from its keys.
///
/// # Arguments
///
/// * `msg_type` - The CCSDS message type named in the error (e.g. "OPM")
/// * `document` - The assembled XML document.
///
/// # Returns
///
/// * `Result<(), BraheError>` - `Ok` when every character is writable, or an
///   `Error` naming the element and the offending code point.
///
/// # Examples
///
/// ```ignore
/// assert!(validate_xml_characters("OPM", "<ORIGINATOR>GSOC</ORIGINATOR>").is_ok());
/// assert!(validate_xml_characters("OPM", "<ORIGINATOR>GS\u{1}C</ORIGINATOR>").is_err());
/// ```
pub(super) fn validate_xml_characters(msg_type: &str, document: &str) -> Result<(), BraheError> {
    let Some((index, c)) = document.char_indices().find(|(_, c)| !is_xml_char(*c)) else {
        return Ok(());
    };

    // The offending character sits inside the element opened most recently, so
    // the enclosing tag name is the CCSDS keyword to name in the error. A
    // character that interrupts the name rather than following it leaves only
    // a prefix, since the rest of the name cannot be printed.
    let tag = document[..index]
        .rfind('<')
        .map(|open| &document[open + 1..index])
        .unwrap_or_default();
    let element: String = tag
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | ':'))
        .collect();
    let located = if element.len() == tag.len() {
        format!("XML element name beginning '{}'", element)
    } else {
        format!("XML element '{}'", element)
    };

    Err(BraheError::Error(format!(
        "CCSDS {}: {} contains U+{:04X}, which XML 1.0 forbids in any document; \
         the character has no XML representation and must be removed before writing",
        msg_type, located, c as u32
    )))
}

/// XML covariance element names (6x6 lower-triangular, 21 elements).
pub(super) const COV_NAMES: [&str; 21] = [
    "CX_X",
    "CY_X",
    "CY_Y",
    "CZ_X",
    "CZ_Y",
    "CZ_Z",
    "CX_DOT_X",
    "CX_DOT_Y",
    "CX_DOT_Z",
    "CX_DOT_X_DOT",
    "CY_DOT_X",
    "CY_DOT_Y",
    "CY_DOT_Z",
    "CY_DOT_X_DOT",
    "CY_DOT_Y_DOT",
    "CZ_DOT_X",
    "CZ_DOT_Y",
    "CZ_DOT_Z",
    "CZ_DOT_X_DOT",
    "CZ_DOT_Y_DOT",
    "CZ_DOT_Z_DOT",
];

/// Write XML header block (shared across OEM/OMM/OPM).
pub(super) fn write_xml_header(out: &mut String, header: &ODMHeader, i1: &str, i2: &str) {
    out.push_str(&format!("{}<header>\n", i1));
    for c in &header.comments {
        out.push_str(&format!(
            "{}<COMMENT>{}</COMMENT>\n",
            i2,
            escape_xml_text(c)
        ));
    }
    if let Some(ref cl) = header.classification {
        out.push_str(&format!(
            "{}<CLASSIFICATION>{}</CLASSIFICATION>\n",
            i2,
            escape_xml_text(cl)
        ));
    }
    out.push_str(&format!(
        "{}<CREATION_DATE>{}</CREATION_DATE>\n",
        i2,
        format_ccsds_datetime_in(&header.creation_date, &CCSDSTimeSystem::UTC)
    ));
    out.push_str(&format!(
        "{}<ORIGINATOR>{}</ORIGINATOR>\n",
        i2,
        escape_xml_text(&header.originator)
    ));
    if let Some(ref mid) = header.message_id {
        out.push_str(&format!(
            "{}<MESSAGE_ID>{}</MESSAGE_ID>\n",
            i2,
            escape_xml_text(mid)
        ));
    }
    out.push_str(&format!("{}</header>\n", i1));
}

/// Write XML 6x6 covariance block (shared across OEM/OMM/OPM).
/// Write an XML covariance block.
///
/// `with_epoch` selects whether the block carries an `EPOCH` element. CCSDS
/// 502.0-B-3 subsection 5.2.5.3 requires one for the OEM, whose covariance
/// matrices each belong to a navigation solution of their own, while tables
/// 3-3 (OPM) and 4-3 (OMM) list only `COMMENT`, `COV_REF_FRAME`, and the
/// matrix entries, the covariance there applying to the message's single
/// state.
pub(super) fn write_xml_covariance(
    out: &mut String,
    cov: &CCSDSCovariance,
    time_system: &CCSDSTimeSystem,
    with_epoch: bool,
    i_block: &str,
    i_elem: &str,
) {
    out.push_str(&format!("{}<covarianceMatrix>\n", i_block));
    if with_epoch && let Some(ref epoch) = cov.epoch {
        out.push_str(&format!(
            "{}<EPOCH>{}</EPOCH>\n",
            i_elem,
            format_ccsds_datetime_in(epoch, time_system)
        ));
    }
    if let Some(ref frame) = cov.cov_ref_frame {
        out.push_str(&format!(
            "{}<COV_REF_FRAME>{}</COV_REF_FRAME>\n",
            i_elem, frame
        ));
    }
    for c in &cov.comments {
        out.push_str(&format!(
            "{}<COMMENT>{}</COMMENT>\n",
            i_elem,
            escape_xml_text(c)
        ));
    }
    // Convert m² → km²
    let values = covariance_to_lower_triangular(&cov.matrix, 1e-6).map(round_ccsds_value);
    for (i, v) in values.iter().enumerate() {
        out.push_str(&format!(
            "{}<{}>{:E}</{}>\n",
            i_elem, COV_NAMES[i], v, COV_NAMES[i]
        ));
    }
    out.push_str(&format!("{}</covarianceMatrix>\n", i_block));
}

/// Write XML spacecraft parameters block (shared across OMM/OPM).
pub(super) fn write_xml_spacecraft_params(
    out: &mut String,
    sp: &CCSDSSpacecraftParameters,
    i_block: &str,
    i_elem: &str,
) {
    out.push_str(&format!("{}<spacecraftParameters>\n", i_block));
    for c in &sp.comments {
        out.push_str(&format!(
            "{}<COMMENT>{}</COMMENT>\n",
            i_elem,
            escape_xml_text(c)
        ));
    }
    if let Some(v) = sp.mass {
        out.push_str(&format!("{}<MASS>{:.6}</MASS>\n", i_elem, v));
    }
    if let Some(v) = sp.solar_rad_area {
        out.push_str(&format!(
            "{}<SOLAR_RAD_AREA>{:.6}</SOLAR_RAD_AREA>\n",
            i_elem, v
        ));
    }
    if let Some(v) = sp.solar_rad_coeff {
        out.push_str(&format!(
            "{}<SOLAR_RAD_COEFF>{:.6}</SOLAR_RAD_COEFF>\n",
            i_elem, v
        ));
    }
    if let Some(v) = sp.drag_area {
        out.push_str(&format!("{}<DRAG_AREA>{:.6}</DRAG_AREA>\n", i_elem, v));
    }
    if let Some(v) = sp.drag_coeff {
        out.push_str(&format!("{}<DRAG_COEFF>{:.6}</DRAG_COEFF>\n", i_elem, v));
    }
    out.push_str(&format!("{}</spacecraftParameters>\n", i_block));
}

/// Write XML user-defined parameters block (shared across OMM/OPM).
pub(super) fn write_xml_user_defined(
    out: &mut String,
    ud: &CCSDSUserDefined,
    i_block: &str,
    i_elem: &str,
) {
    out.push_str(&format!("{}<userDefinedParameters>\n", i_block));
    for (k, v) in &ud.parameters {
        out.push_str(&format!(
            "{}<USER_DEFINED_{} value=\"{}\"/>\n",
            i_elem,
            k,
            escape_xml_attribute(v)
        ));
    }
    out.push_str(&format!("{}</userDefinedParameters>\n", i_block));
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::ccsds::apm::APM;
    use crate::ccsds::cdm::CDM;
    use crate::ccsds::common::CCSDSFormat;
    use crate::ccsds::oem::OEM;
    use crate::ccsds::omm::OMM;
    use crate::ccsds::opm::OPM;

    /// Free text exercising every character that terminates XML markup.
    const MARKUP: &str = "R&D <ops> \"quoted\"";

    /// The same text once escaped for element content.
    const MARKUP_ESCAPED: &str = "R&amp;D &lt;ops&gt; \"quoted\"";

    /// Free text carrying a character XML 1.0 forbids in any document.
    const FORBIDDEN: &str = "GSOC\u{1}LAB";

    #[test]
    #[serial_test::parallel]
    fn test_escape_xml_text_and_attribute() {
        assert_eq!(escape_xml_text(MARKUP), MARKUP_ESCAPED);
        assert_eq!(
            escape_xml_attribute(MARKUP),
            "R&amp;D &lt;ops&gt; &quot;quoted&quot;"
        );
        assert_eq!(escape_xml_text("plain text"), "plain text");
    }

    #[test]
    #[serial_test::parallel]
    fn test_write_oem_xml_escapes_free_text() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let mut oem = OEM::from_str(&content).unwrap();
        oem.header.originator = MARKUP.to_string();
        oem.header.comments = vec![MARKUP.to_string()];
        oem.segments[0].metadata.object_name = MARKUP.to_string();

        let xml = oem.to_string(CCSDSFormat::XML).unwrap();
        assert!(xml.contains(&format!("<ORIGINATOR>{}</ORIGINATOR>", MARKUP_ESCAPED)));
        assert!(xml.contains(&format!("<COMMENT>{}</COMMENT>", MARKUP_ESCAPED)));
        assert!(xml.contains(&format!("<OBJECT_NAME>{}</OBJECT_NAME>", MARKUP_ESCAPED)));

        let reparsed = OEM::from_str(&xml).unwrap();
        assert_eq!(reparsed.header.originator, MARKUP);
        assert_eq!(reparsed.header.comments, vec![MARKUP.to_string()]);
        assert_eq!(reparsed.segments[0].metadata.object_name, MARKUP);
    }

    #[test]
    #[serial_test::parallel]
    fn test_write_omm_xml_escapes_free_text_and_user_defined() {
        let content = std::fs::read_to_string("test_assets/ccsds/omm/OMMExample2.txt").unwrap();
        let mut omm = OMM::from_str(&content).unwrap();
        omm.header.originator = MARKUP.to_string();
        omm.metadata.object_name = MARKUP.to_string();
        omm.user_defined = Some(crate::ccsds::common::CCSDSUserDefined {
            parameters: std::collections::HashMap::from([(
                "EARTH_MODEL".to_string(),
                MARKUP.to_string(),
            )]),
        });

        let xml = omm.to_string(CCSDSFormat::XML).unwrap();
        assert!(xml.contains(&format!("<ORIGINATOR>{}</ORIGINATOR>", MARKUP_ESCAPED)));
        assert!(xml.contains(&format!("<OBJECT_NAME>{}</OBJECT_NAME>", MARKUP_ESCAPED)));
        assert!(xml.contains(r#"value="R&amp;D &lt;ops&gt; &quot;quoted&quot;""#));

        let reparsed = OMM::from_str(&xml).unwrap();
        assert_eq!(reparsed.header.originator, MARKUP);
        assert_eq!(reparsed.metadata.object_name, MARKUP);
        assert_eq!(
            reparsed.user_defined.unwrap().parameters["EARTH_MODEL"],
            MARKUP
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_write_opm_xml_escapes_free_text() {
        let content = std::fs::read_to_string("test_assets/ccsds/opm/OPMExample1.txt").unwrap();
        let mut opm = OPM::from_str(&content).unwrap();
        opm.header.originator = MARKUP.to_string();
        opm.metadata.object_name = MARKUP.to_string();
        opm.metadata.comments = vec![MARKUP.to_string()];

        let xml = opm.to_string(CCSDSFormat::XML).unwrap();
        assert!(xml.contains(&format!("<ORIGINATOR>{}</ORIGINATOR>", MARKUP_ESCAPED)));
        assert!(xml.contains(&format!("<OBJECT_NAME>{}</OBJECT_NAME>", MARKUP_ESCAPED)));

        let reparsed = OPM::from_str(&xml).unwrap();
        assert_eq!(reparsed.header.originator, MARKUP);
        assert_eq!(reparsed.metadata.object_name, MARKUP);
        assert_eq!(reparsed.metadata.comments, vec![MARKUP.to_string()]);
    }

    #[test]
    #[serial_test::parallel]
    fn test_write_cdm_xml_escapes_free_text() {
        let content = std::fs::read_to_string("test_assets/ccsds/cdm/CDMExample1.txt").unwrap();
        let mut cdm = CDM::from_str(&content).unwrap();
        cdm.header.originator = MARKUP.to_string();
        cdm.object1.metadata.object_name = MARKUP.to_string();
        cdm.object1.metadata.gravity_model = Some(MARKUP.to_string());

        let xml = cdm.to_string(CCSDSFormat::XML).unwrap();
        assert!(xml.contains(&format!("<ORIGINATOR>{}</ORIGINATOR>", MARKUP_ESCAPED)));
        assert!(xml.contains(&format!("<OBJECT_NAME>{}</OBJECT_NAME>", MARKUP_ESCAPED)));
        assert!(xml.contains(&format!(
            "<GRAVITY_MODEL>{}</GRAVITY_MODEL>",
            MARKUP_ESCAPED
        )));

        let reparsed = CDM::from_str(&xml).unwrap();
        assert_eq!(reparsed.header.originator, MARKUP);
        assert_eq!(reparsed.object1.metadata.object_name, MARKUP);
        assert_eq!(
            reparsed.object1.metadata.gravity_model.as_deref(),
            Some(MARKUP)
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_xml_escapes_comments_in_every_block() {
        // Every block that can carry a COMMENT gets one containing markup, so
        // the escaping is exercised on each emission site rather than only the
        // few a stock fixture populates.
        let tagged = |name: &str| format!("{} {}", name, MARKUP);
        let escaped = |name: &str| format!("<COMMENT>{} {}</COMMENT>", name, MARKUP_ESCAPED);

        // OEM: header, metadata, data, and covariance.
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let mut oem = OEM::from_str(&content).unwrap();
        oem.header.comments = vec![tagged("oem header")];
        for segment in &mut oem.segments {
            segment.metadata.comments = vec![tagged("oem metadata")];
            segment.comments = vec![tagged("oem data")];
            for covariance in &mut segment.covariances {
                covariance.comments = vec![tagged("oem covariance")];
            }
        }
        let xml = oem.to_string(CCSDSFormat::XML).unwrap();
        for block in ["oem header", "oem metadata", "oem data", "oem covariance"] {
            assert!(xml.contains(&escaped(block)), "OEM missing {}", block);
        }

        // OMM: header, metadata, mean elements, TLE, and spacecraft parameters.
        let content = std::fs::read_to_string("test_assets/ccsds/omm/OMMExample2.txt").unwrap();
        let mut omm = OMM::from_str(&content).unwrap();
        omm.header.comments = vec![tagged("omm header")];
        omm.metadata.comments = vec![tagged("omm metadata")];
        omm.mean_elements.comments = vec![tagged("omm mean elements")];
        if let Some(ref mut tle) = omm.tle_parameters {
            tle.comments = vec![tagged("omm tle")];
        }
        omm.spacecraft_parameters = Some(crate::ccsds::common::CCSDSSpacecraftParameters {
            mass: Some(300.0),
            solar_rad_area: None,
            solar_rad_coeff: None,
            drag_area: None,
            drag_coeff: None,
            comments: vec![tagged("omm spacecraft")],
        });
        let xml = omm.to_string(CCSDSFormat::XML).unwrap();
        for block in [
            "omm header",
            "omm metadata",
            "omm mean elements",
            "omm tle",
            "omm spacecraft",
        ] {
            assert!(xml.contains(&escaped(block)), "OMM missing {}", block);
        }

        // OPM: state vector and Keplerian elements.
        let content = std::fs::read_to_string("test_assets/ccsds/opm/OPMExample2.txt").unwrap();
        let mut opm = OPM::from_str(&content).unwrap();
        opm.state_vector.comments = vec![tagged("opm state vector")];
        opm.keplerian_elements
            .as_mut()
            .expect("fixture has Keplerian elements")
            .comments = vec![tagged("opm keplerian")];
        let xml = opm.to_string(CCSDSFormat::XML).unwrap();
        for block in ["opm state vector", "opm keplerian"] {
            assert!(xml.contains(&escaped(block)), "OPM missing {}", block);
        }

        // CDM: object metadata and each data sub-block, plus the two optional
        // metadata fields whose escaping no fixture reaches.
        let content = std::fs::read_to_string("test_assets/ccsds/cdm/CDMExample2.txt").unwrap();
        let mut cdm = CDM::from_str(&content).unwrap();
        let object = &mut cdm.object1;
        object.metadata.comments = vec![tagged("cdm metadata")];
        object.metadata.ops_status = Some(MARKUP.to_string());
        object.metadata.orbit_center = Some(MARKUP.to_string());
        object.data.comments = vec![tagged("cdm data")];
        object.data.state_vector.comments = vec![tagged("cdm state vector")];
        object.data.rtn_covariance.comments = vec![tagged("cdm rtn covariance")];
        if let Some(ref mut od) = object.data.od_parameters {
            od.comments = vec![tagged("cdm od")];
        }
        if let Some(ref mut ap) = object.data.additional_parameters {
            ap.comments = vec![tagged("cdm additional")];
        }
        object.data.additional_covariance_metadata =
            Some(crate::ccsds::cdm::CDMAdditionalCovarianceMetadata {
                density_forecast_uncertainty: None,
                cscale_factor_min: None,
                cscale_factor: None,
                cscale_factor_max: None,
                screening_data_source: None,
                dcp_sensitivity_vector_position: None,
                dcp_sensitivity_vector_velocity: None,
                comments: vec![tagged("cdm covariance metadata")],
            });
        let xml = cdm.to_string(CCSDSFormat::XML).unwrap();
        for block in [
            "cdm metadata",
            "cdm data",
            "cdm state vector",
            "cdm rtn covariance",
            "cdm od",
            "cdm additional",
            "cdm covariance metadata",
        ] {
            assert!(xml.contains(&escaped(block)), "CDM missing {}", block);
        }
        assert!(xml.contains(&format!("<OPS_STATUS>{}</OPS_STATUS>", MARKUP_ESCAPED)));
        assert!(xml.contains(&format!("<ORBIT_CENTER>{}</ORBIT_CENTER>", MARKUP_ESCAPED)));

        // Nothing raw leaked into any of them.
        assert!(!xml.contains("R&D <ops>"));
    }

    #[test]
    #[serial_test::parallel]
    fn test_write_oem_xml_rejects_forbidden_characters() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let mut oem = OEM::from_str(&content).unwrap();
        oem.header.originator = FORBIDDEN.to_string();

        let msg = oem.to_string(CCSDSFormat::XML).unwrap_err().to_string();
        assert!(msg.contains("OEM"), "{}", msg);
        assert!(msg.contains("ORIGINATOR"), "{}", msg);
        assert!(msg.contains("U+0001"), "{}", msg);
    }

    #[test]
    #[serial_test::parallel]
    fn test_write_omm_xml_rejects_forbidden_characters() {
        let content = std::fs::read_to_string("test_assets/ccsds/omm/OMMExample2.txt").unwrap();
        let mut omm = OMM::from_str(&content).unwrap();
        omm.metadata.object_name = FORBIDDEN.to_string();

        let msg = omm.to_string(CCSDSFormat::XML).unwrap_err().to_string();
        assert!(msg.contains("OMM"), "{}", msg);
        assert!(msg.contains("OBJECT_NAME"), "{}", msg);
        assert!(msg.contains("U+0001"), "{}", msg);

        // A user-defined parameter reaches the document as an attribute value
        // and as part of an element name, neither of which the element-content
        // escaping touches.
        let mut omm = OMM::from_str(&content).unwrap();
        omm.user_defined = Some(crate::ccsds::common::CCSDSUserDefined {
            parameters: std::collections::HashMap::from([(
                "EARTH_MODEL".to_string(),
                FORBIDDEN.to_string(),
            )]),
        });
        let msg = omm.to_string(CCSDSFormat::XML).unwrap_err().to_string();
        assert!(msg.contains("USER_DEFINED_EARTH_MODEL"), "{}", msg);
        assert!(msg.contains("U+0001"), "{}", msg);

        let mut omm = OMM::from_str(&content).unwrap();
        omm.user_defined = Some(crate::ccsds::common::CCSDSUserDefined {
            parameters: std::collections::HashMap::from([(
                FORBIDDEN.to_string(),
                "value".to_string(),
            )]),
        });
        let msg = omm.to_string(CCSDSFormat::XML).unwrap_err().to_string();
        assert!(msg.contains("USER_DEFINED_GSOC"), "{}", msg);
        assert!(msg.contains("U+0001"), "{}", msg);
    }

    #[test]
    #[serial_test::parallel]
    fn test_write_opm_xml_rejects_forbidden_characters() {
        let content = std::fs::read_to_string("test_assets/ccsds/opm/OPMExample1.txt").unwrap();
        let mut opm = OPM::from_str(&content).unwrap();
        opm.metadata.comments = vec![FORBIDDEN.to_string()];

        let msg = opm.to_string(CCSDSFormat::XML).unwrap_err().to_string();
        assert!(msg.contains("OPM"), "{}", msg);
        assert!(msg.contains("COMMENT"), "{}", msg);
        assert!(msg.contains("U+0001"), "{}", msg);
    }

    #[test]
    #[serial_test::parallel]
    fn test_write_cdm_xml_rejects_forbidden_characters() {
        let content = std::fs::read_to_string("test_assets/ccsds/cdm/CDMExample1.txt").unwrap();
        let mut cdm = CDM::from_str(&content).unwrap();
        cdm.object1.metadata.object_name = FORBIDDEN.to_string();

        let msg = cdm.to_string(CCSDSFormat::XML).unwrap_err().to_string();
        assert!(msg.contains("CDM"), "{}", msg);
        assert!(msg.contains("OBJECT_NAME"), "{}", msg);
        assert!(msg.contains("U+0001"), "{}", msg);
    }

    #[test]
    #[serial_test::parallel]
    fn test_write_apm_xml_rejects_forbidden_characters() {
        let content = std::fs::read_to_string("test_assets/ccsds/apm/APMExampleG1.txt").unwrap();
        let mut apm = APM::from_str(&content).unwrap();
        apm.header.originator = FORBIDDEN.to_string();

        let msg = apm.to_string(CCSDSFormat::XML).unwrap_err().to_string();
        assert!(msg.contains("APM"), "{}", msg);
        assert!(msg.contains("ORIGINATOR"), "{}", msg);
        assert!(msg.contains("U+0001"), "{}", msg);
    }

    #[test]
    #[serial_test::parallel]
    fn test_write_xml_follows_the_xml_char_production() {
        // XML 1.0 subsection 2.2 admits #x9, #xA, #xD, and the ranges
        // [#x20-#xD7FF], [#xE000-#xFFFD], and [#x10000-#x10FFFF]; everything
        // else is forbidden outright.
        let content = std::fs::read_to_string("test_assets/ccsds/opm/OPMExample1.txt").unwrap();
        let write_with_originator = |c: char| {
            let mut opm = OPM::from_str(&content).unwrap();
            opm.header.originator = format!("GSOC{}LAB", c);
            opm.to_string(CCSDSFormat::XML)
        };

        for c in [
            '\t',
            '\n',
            '\r',
            ' ',
            '\u{D7FF}',
            '\u{E000}',
            '\u{FFFD}',
            '\u{10000}',
            '\u{1FFFE}',
            '\u{10FFFF}',
        ] {
            let xml = write_with_originator(c)
                .unwrap_or_else(|e| panic!("U+{:04X} should be written: {}", c as u32, e));
            assert!(
                xml.contains(&format!("<ORIGINATOR>GSOC{}LAB</ORIGINATOR>", c)),
                "U+{:04X} missing from output",
                c as u32
            );
        }

        for c in [
            '\u{0}', '\u{8}', '\u{B}', '\u{C}', '\u{E}', '\u{1F}', '\u{FFFE}', '\u{FFFF}',
        ] {
            let err = write_with_originator(c)
                .expect_err(&format!("U+{:04X} should be rejected", c as u32));
            assert!(
                err.to_string().contains(&format!("U+{:04X}", c as u32)),
                "{}",
                err
            );
        }
    }
}
