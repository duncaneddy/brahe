/*!
 * XML reader and writer for the Orbit Parameter Message (OPM).
 *
 * Reference: CCSDS 502.0-B-3 (Orbit Data Messages), section 3
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

// ============================================================================
// OPM XML Writer
// ============================================================================

/// Write an OPM message to XML format.
pub fn write_opm_xml(opm: &crate::ccsds::opm::OPM) -> Result<String, BraheError> {
    let mut out = String::new();
    let i1 = "  ";
    let i2 = "    ";
    let i3 = "      ";
    let i4 = "        ";

    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str(&format!(
        "<opm id=\"CCSDS_OPM_VERS\" version=\"{:.1}\">\n",
        opm.header.format_version
    ));

    write_xml_header(&mut out, &opm.header, i1, i2);

    out.push_str(&format!("{}<body>\n", i1));
    out.push_str(&format!("{}<segment>\n", i2));

    // Metadata
    out.push_str(&format!("{}<metadata>\n", i3));
    for c in &opm.metadata.comments {
        out.push_str(&format!(
            "{}<COMMENT>{}</COMMENT>\n",
            i4,
            escape_xml_text(c)
        ));
    }
    out.push_str(&format!(
        "{}<OBJECT_NAME>{}</OBJECT_NAME>\n",
        i4,
        escape_xml_text(&opm.metadata.object_name)
    ));
    out.push_str(&format!(
        "{}<OBJECT_ID>{}</OBJECT_ID>\n",
        i4,
        escape_xml_text(&opm.metadata.object_id)
    ));
    out.push_str(&format!(
        "{}<CENTER_NAME>{}</CENTER_NAME>\n",
        i4,
        escape_xml_text(&opm.metadata.center_name)
    ));
    out.push_str(&format!(
        "{}<REF_FRAME>{}</REF_FRAME>\n",
        i4, opm.metadata.ref_frame
    ));
    if let Some(ref e) = opm.metadata.ref_frame_epoch {
        out.push_str(&format!(
            "{}<REF_FRAME_EPOCH>{}</REF_FRAME_EPOCH>\n",
            i4,
            format_ccsds_datetime_in(e, &opm.metadata.time_system)
        ));
    }
    out.push_str(&format!(
        "{}<TIME_SYSTEM>{}</TIME_SYSTEM>\n",
        i4, opm.metadata.time_system
    ));
    out.push_str(&format!("{}</metadata>\n", i3));

    // Data
    out.push_str(&format!("{}<data>\n", i3));

    // State vector
    out.push_str(&format!("{}<stateVector>\n", i4));
    for c in &opm.state_vector.comments {
        out.push_str(&format!(" <COMMENT>{}</COMMENT>\n", escape_xml_text(c)));
    }
    out.push_str(&format!(
        "        <EPOCH>{}</EPOCH>\n",
        format_ccsds_datetime_in(&opm.state_vector.epoch, &opm.metadata.time_system)
    ));
    // Position: m → km
    out.push_str(&format!(
        "        <X>{:.6}</X>\n",
        opm.state_vector.position[0] / 1e3
    ));
    out.push_str(&format!(
        "        <Y>{:.6}</Y>\n",
        opm.state_vector.position[1] / 1e3
    ));
    out.push_str(&format!(
        "        <Z>{:.6}</Z>\n",
        opm.state_vector.position[2] / 1e3
    ));
    // Velocity: m/s → km/s
    out.push_str(&format!(
        "        <X_DOT>{:.6}</X_DOT>\n",
        opm.state_vector.velocity[0] / 1e3
    ));
    out.push_str(&format!(
        "        <Y_DOT>{:.6}</Y_DOT>\n",
        opm.state_vector.velocity[1] / 1e3
    ));
    out.push_str(&format!(
        "        <Z_DOT>{:.6}</Z_DOT>\n",
        opm.state_vector.velocity[2] / 1e3
    ));
    out.push_str(&format!("{}</stateVector>\n", i4));

    // Keplerian elements
    if let Some(ref ke) = opm.keplerian_elements {
        out.push_str(&format!("{}<keplerianElements>\n", i4));
        for c in &ke.comments {
            out.push_str(&format!(" <COMMENT>{}</COMMENT>\n", escape_xml_text(c)));
        }
        // Semi-major axis: m → km
        out.push_str(&format!(
            "        <SEMI_MAJOR_AXIS>{:.6}</SEMI_MAJOR_AXIS>\n",
            ke.semi_major_axis / 1e3
        ));
        out.push_str(&format!(
            "        <ECCENTRICITY>{}</ECCENTRICITY>\n",
            ke.eccentricity
        ));
        out.push_str(&format!(
            "        <INCLINATION>{}</INCLINATION>\n",
            ke.inclination
        ));
        out.push_str(&format!(
            "        <RA_OF_ASC_NODE>{}</RA_OF_ASC_NODE>\n",
            ke.ra_of_asc_node
        ));
        out.push_str(&format!(
            "        <ARG_OF_PERICENTER>{}</ARG_OF_PERICENTER>\n",
            ke.arg_of_pericenter
        ));
        if let Some(ta) = ke.true_anomaly {
            out.push_str(&format!("        <TRUE_ANOMALY>{}</TRUE_ANOMALY>\n", ta));
        }
        if let Some(ma) = ke.mean_anomaly {
            out.push_str(&format!("        <MEAN_ANOMALY>{}</MEAN_ANOMALY>\n", ma));
        }
        if let Some(gm) = ke.gm {
            // m³/s² → km³/s²
            out.push_str(&format!("        <GM>{}</GM>\n", gm / 1e9));
        }
        out.push_str(&format!("{}</keplerianElements>\n", i4));
    }

    // Spacecraft parameters
    if let Some(ref sp) = opm.spacecraft_parameters {
        write_xml_spacecraft_params(&mut out, sp, i4, "        ");
    }

    // Covariance
    if let Some(ref cov) = opm.covariance {
        write_xml_covariance(
            &mut out,
            cov,
            &opm.metadata.time_system,
            false,
            i4,
            "        ",
        );
    }

    // Maneuvers
    for man in &opm.maneuvers {
        out.push_str(&format!("{}<maneuverParameters>\n", i4));
        for c in &man.comments {
            out.push_str(&format!(" <COMMENT>{}</COMMENT>\n", escape_xml_text(c)));
        }
        out.push_str(&format!(
            "        <MAN_EPOCH_IGNITION>{}</MAN_EPOCH_IGNITION>\n",
            format_ccsds_datetime_in(&man.epoch_ignition, &opm.metadata.time_system)
        ));
        out.push_str(&format!(
            "        <MAN_DURATION>{:.2}</MAN_DURATION>\n",
            man.duration
        ));
        if let Some(dm) = man.delta_mass {
            out.push_str(&format!(
                "        <MAN_DELTA_MASS>{:.3}</MAN_DELTA_MASS>\n",
                dm
            ));
        }
        out.push_str(&format!(
            "        <MAN_REF_FRAME>{}</MAN_REF_FRAME>\n",
            man.ref_frame
        ));
        // DV: m/s → km/s
        out.push_str(&format!(
            "        <MAN_DV_1>{:.8}</MAN_DV_1>\n",
            man.dv[0] / 1e3
        ));
        out.push_str(&format!(
            "        <MAN_DV_2>{:.8}</MAN_DV_2>\n",
            man.dv[1] / 1e3
        ));
        out.push_str(&format!(
            "        <MAN_DV_3>{:.8}</MAN_DV_3>\n",
            man.dv[2] / 1e3
        ));
        out.push_str(&format!("{}</maneuverParameters>\n", i4));
    }

    out.push_str(&format!("{}</data>\n", i3));
    out.push_str(&format!("{}</segment>\n", i2));

    // User-defined parameters
    if let Some(ref ud) = opm.user_defined {
        write_xml_user_defined(&mut out, ud, i2, i3);
    }

    out.push_str(&format!("{}</body>\n", i1));
    out.push_str("</opm>\n");

    validate_xml_characters("OPM", &out)?;

    Ok(out)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {

    use crate::ccsds::xml::parse_opm_xml;

    #[test]
    #[serial_test::parallel]
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
    #[serial_test::parallel]
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
}
