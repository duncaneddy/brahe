/*!
 * KVN reader and writer for the Orbit Mean-elements Message (OMM).
 *
 * Reference: CCSDS 502.0-B-3 (Orbit Data Messages), section 4
 */

use std::collections::HashMap;

use crate::ccsds::common::{
    CCSDSCovariance, CCSDSRefFrame, CCSDSSpacecraftParameters, CCSDSTimeSystem, CCSDSUserDefined,
    ODMHeader, covariance_from_lower_triangular, format_ccsds_datetime_in, parse_ccsds_datetime,
    strip_units,
};
use crate::ccsds::error::{ccsds_missing_field, ccsds_parse_error};
use crate::ccsds::kvn::common::{
    KVNToken, tokenize_line, write_kvn_covariance_elements, write_kvn_spacecraft_params,
    write_kvn_user_defined,
};
use crate::ccsds::omm::{OMM, OMMMetadata, OMMTleParameters, OMMeanElements};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

/// The OMM section a keyword belongs to.
#[derive(Clone, Copy, PartialEq)]
enum OMMSection {
    Header,
    Metadata,
    MeanElements,
    TleParameters,
    SpacecraftParameters,
    Covariance,
}

/// The OMM section a keyword introduces, or `None` when the keyword is not one
/// this parser recognizes.
///
/// Keywords are grouped as CCSDS 502.0-B-3 tables 4-1 (header), 4-2 (metadata),
/// and 4-3 (data) list them.
///
/// # Arguments
///
/// * `key` - The KVN keyword, without surrounding whitespace.
///
/// # Returns
///
/// * `Option<OMMSection>` - The owning section, or `None` if unrecognized.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(omm_section("ORIGINATOR"), Some(OMMSection::Header));
/// assert_eq!(omm_section("MEAN_MOTION"), Some(OMMSection::MeanElements));
/// assert_eq!(omm_section("BSTAR"), Some(OMMSection::TleParameters));
/// // EPOCH appears in more than one block, so it names none on its own.
/// assert_eq!(omm_section("EPOCH"), None);
/// ```
fn omm_section(key: &str) -> Option<OMMSection> {
    Some(match key {
        "CCSDS_OMM_VERS" | "CLASSIFICATION" | "CREATION_DATE" | "ORIGINATOR" | "MESSAGE_ID" => {
            OMMSection::Header
        }

        "OBJECT_NAME"
        | "OBJECT_ID"
        | "CENTER_NAME"
        | "REF_FRAME"
        | "REF_FRAME_EPOCH"
        | "TIME_SYSTEM"
        | "MEAN_ELEMENT_THEORY" => OMMSection::Metadata,

        // As for the OPM, EPOCH alone does not name a block.
        "EPOCH" => return None,

        "SEMI_MAJOR_AXIS" | "MEAN_MOTION" | "ECCENTRICITY" | "INCLINATION" | "RA_OF_ASC_NODE"
        | "ARG_OF_PERICENTER" | "MEAN_ANOMALY" | "GM" => OMMSection::MeanElements,

        "EPHEMERIS_TYPE"
        | "CLASSIFICATION_TYPE"
        | "NORAD_CAT_ID"
        | "ELEMENT_SET_NO"
        | "REV_AT_EPOCH"
        | "BSTAR"
        | "BTERM"
        | "MEAN_MOTION_DOT"
        | "MEAN_MOTION_DDOT"
        | "AGOM" => OMMSection::TleParameters,

        "MASS" | "SOLAR_RAD_AREA" | "SOLAR_RAD_COEFF" | "DRAG_AREA" | "DRAG_COEFF" => {
            OMMSection::SpacecraftParameters
        }

        "COV_REF_FRAME" => OMMSection::Covariance,
        k if k.starts_with("CX_") || k.starts_with("CY_") || k.starts_with("CZ_") => {
            OMMSection::Covariance
        }

        _ => return None,
    })
}

/// Parse an OMM message from KVN format.
///
/// OMM KVN is flat — no META_START/META_STOP blocks. All key-value pairs
/// are parsed sequentially into header, metadata, mean elements, TLE params,
/// spacecraft params, covariance, and user-defined sections.
pub fn parse_omm(content: &str) -> Result<OMM, BraheError> {
    // Collect all key-value pairs and comments
    let mut header_comments: Vec<String> = Vec::new();
    let mut metadata_comments: Vec<String> = Vec::new();
    let mut mean_element_comments: Vec<String> = Vec::new();
    let mut tle_comments: Vec<String> = Vec::new();
    let mut spacecraft_comments: Vec<String> = Vec::new();

    // Header
    let mut format_version: Option<f64> = None;
    let mut classification: Option<String> = None;
    let mut creation_date: Option<Epoch> = None;
    let mut originator: Option<String> = None;
    let mut message_id: Option<String> = None;

    // Metadata
    let mut object_name: Option<String> = None;
    let mut object_id: Option<String> = None;
    let mut center_name: Option<String> = None;
    let mut ref_frame: Option<CCSDSRefFrame> = None;
    let mut ref_frame_epoch: Option<String> = None;
    let mut time_system: Option<CCSDSTimeSystem> = None;
    let mut mean_element_theory: Option<String> = None;

    // Mean elements
    let mut epoch: Option<Epoch> = None;
    let mut mean_motion: Option<f64> = None;
    let mut semi_major_axis: Option<f64> = None;
    let mut eccentricity: Option<f64> = None;
    let mut inclination: Option<f64> = None;
    let mut ra_of_asc_node: Option<f64> = None;
    let mut arg_of_pericenter: Option<f64> = None;
    let mut mean_anomaly: Option<f64> = None;
    let mut gm: Option<f64> = None;

    // TLE parameters
    let mut ephemeris_type: Option<u32> = None;
    let mut classification_type: Option<char> = None;
    let mut norad_cat_id: Option<u32> = None;
    let mut element_set_no: Option<u32> = None;
    let mut rev_at_epoch: Option<u32> = None;
    let mut bstar: Option<f64> = None;
    let mut bterm: Option<f64> = None;
    let mut mean_motion_dot: Option<f64> = None;
    let mut mean_motion_ddot: Option<f64> = None;
    let mut agom: Option<f64> = None;

    // Spacecraft parameters
    let mut mass: Option<f64> = None;
    let mut solar_rad_area: Option<f64> = None;
    let mut solar_rad_coeff: Option<f64> = None;
    let mut drag_area: Option<f64> = None;
    let mut drag_coeff: Option<f64> = None;

    // Covariance
    let mut cov_ref_frame: Option<CCSDSRefFrame> = None;
    let mut cov_values: Vec<f64> = Vec::new();
    let mut cov_comments: Vec<String> = Vec::new();

    // User-defined
    let mut user_defined: HashMap<String, String> = HashMap::new();

    // Comments accumulate until a keyword names the section they introduce.
    // CCSDS 502.0-B-3 tables 4-1 through 4-3 open every OMM section with a
    // COMMENT row and subsection 7.4.8 fixes the keyword order to match, so a
    // comment belongs to the section that follows it, not the one before it.
    let mut pending_comments: Vec<String> = Vec::new();
    let mut last_section = OMMSection::Header;

    let active_ts = |ts: &Option<CCSDSTimeSystem>| ts.clone().unwrap_or(CCSDSTimeSystem::UTC);

    macro_rules! file_comments {
        ($section:expr) => {{
            if !pending_comments.is_empty() {
                let sink = match $section {
                    OMMSection::Header => &mut header_comments,
                    OMMSection::Metadata => &mut metadata_comments,
                    OMMSection::MeanElements => &mut mean_element_comments,
                    OMMSection::TleParameters => &mut tle_comments,
                    OMMSection::SpacecraftParameters => &mut spacecraft_comments,
                    OMMSection::Covariance => &mut cov_comments,
                };
                sink.append(&mut pending_comments);
            }
        }};
    }

    for line in content.lines() {
        let token = tokenize_line(line);
        match token {
            KVNToken::KeyValue { key, value } => {
                let val = strip_units(&value);
                if let Some(section) = omm_section(&key) {
                    last_section = section;
                    file_comments!(section);
                }
                match key.as_str() {
                    // Header
                    "CCSDS_OMM_VERS" => {
                        format_version = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OMM", "invalid version"))?,
                        );
                    }
                    "CLASSIFICATION" => {
                        classification = Some(val.to_string());
                    }
                    "CREATION_DATE" => {
                        creation_date = Some(parse_ccsds_datetime(val, &CCSDSTimeSystem::UTC)?);
                    }
                    "ORIGINATOR" => {
                        originator = Some(val.to_string());
                    }
                    "MESSAGE_ID" => {
                        message_id = Some(val.to_string());
                    }

                    // Metadata
                    "OBJECT_NAME" => {
                        object_name = Some(val.to_string());
                    }
                    "OBJECT_ID" => {
                        object_id = Some(val.to_string());
                    }
                    "CENTER_NAME" => {
                        center_name = Some(val.to_string());
                    }
                    "REF_FRAME" => {
                        ref_frame = Some(CCSDSRefFrame::parse(val));
                    }
                    "REF_FRAME_EPOCH" => ref_frame_epoch = Some(val.to_string()),
                    "TIME_SYSTEM" => {
                        time_system = Some(CCSDSTimeSystem::parse(val)?);
                    }
                    "MEAN_ELEMENT_THEORY" => {
                        mean_element_theory = Some(val.to_string());
                    }

                    // Mean elements
                    "EPOCH" => {
                        epoch = Some(parse_ccsds_datetime(val, &active_ts(&time_system))?);
                    }
                    "MEAN_MOTION" => {
                        mean_motion = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OMM", "invalid MEAN_MOTION"))?,
                        );
                    }
                    "SEMI_MAJOR_AXIS" => {
                        semi_major_axis =
                            Some(val.parse().map_err(|_| {
                                ccsds_parse_error("OMM", "invalid SEMI_MAJOR_AXIS")
                            })?);
                    }
                    "ECCENTRICITY" => {
                        eccentricity = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OMM", "invalid ECCENTRICITY"))?,
                        );
                    }
                    "INCLINATION" => {
                        inclination = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OMM", "invalid INCLINATION"))?,
                        );
                    }
                    "RA_OF_ASC_NODE" => {
                        ra_of_asc_node = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OMM", "invalid RA_OF_ASC_NODE"))?,
                        );
                    }
                    "ARG_OF_PERICENTER" => {
                        arg_of_pericenter =
                            Some(val.parse().map_err(|_| {
                                ccsds_parse_error("OMM", "invalid ARG_OF_PERICENTER")
                            })?);
                    }
                    "MEAN_ANOMALY" => {
                        mean_anomaly = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OMM", "invalid MEAN_ANOMALY"))?,
                        );
                    }
                    "GM" => {
                        let gm_val: f64 = val
                            .parse()
                            .map_err(|_| ccsds_parse_error("OMM", "invalid GM"))?;
                        gm = Some(gm_val * 1e9); // km³/s² → m³/s²
                    }

                    // TLE parameters
                    "EPHEMERIS_TYPE" => {
                        ephemeris_type = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OMM", "invalid EPHEMERIS_TYPE"))?,
                        );
                    }
                    "CLASSIFICATION_TYPE" => {
                        classification_type = val.chars().next();
                    }
                    "NORAD_CAT_ID" => {
                        norad_cat_id = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OMM", "invalid NORAD_CAT_ID"))?,
                        );
                    }
                    "ELEMENT_SET_NO" => {
                        element_set_no = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OMM", "invalid ELEMENT_SET_NO"))?,
                        );
                    }
                    "REV_AT_EPOCH" => {
                        rev_at_epoch = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OMM", "invalid REV_AT_EPOCH"))?,
                        );
                    }
                    "BSTAR" => {
                        bstar = Some(parse_scientific_notation(val)?);
                    }
                    "BTERM" => {
                        bterm = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OMM", "invalid BTERM"))?,
                        );
                    }
                    "MEAN_MOTION_DOT" => {
                        mean_motion_dot = Some(parse_scientific_notation(val)?);
                    }
                    "MEAN_MOTION_DDOT" => {
                        mean_motion_ddot = Some(parse_scientific_notation(val)?);
                    }
                    "AGOM" => {
                        agom = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OMM", "invalid AGOM"))?,
                        );
                    }

                    // Spacecraft parameters
                    "MASS" => {
                        mass = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OMM", "invalid MASS"))?,
                        );
                    }
                    "SOLAR_RAD_AREA" => {
                        solar_rad_area = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OMM", "invalid SOLAR_RAD_AREA"))?,
                        );
                    }
                    "SOLAR_RAD_COEFF" => {
                        solar_rad_coeff =
                            Some(val.parse().map_err(|_| {
                                ccsds_parse_error("OMM", "invalid SOLAR_RAD_COEFF")
                            })?);
                    }
                    "DRAG_AREA" => {
                        drag_area = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OMM", "invalid DRAG_AREA"))?,
                        );
                    }
                    "DRAG_COEFF" => {
                        drag_coeff = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OMM", "invalid DRAG_COEFF"))?,
                        );
                    }

                    // Covariance
                    "COV_REF_FRAME" => {
                        cov_ref_frame = Some(CCSDSRefFrame::parse(val));
                    }
                    k if k.starts_with("CX_") || k.starts_with("CY_") || k.starts_with("CZ_") => {
                        let v: f64 = val.parse().map_err(|_| {
                            ccsds_parse_error("OMM", &format!("invalid covariance value '{}'", val))
                        })?;
                        cov_values.push(v);
                    }

                    // User-defined
                    k if k.starts_with("USER_DEFINED_") => {
                        let param_name = k.strip_prefix("USER_DEFINED_").unwrap_or(k);
                        user_defined.insert(param_name.to_string(), val.to_string());
                    }

                    _ => {
                        // Unknown key — skip for robustness
                    }
                }
            }
            KVNToken::Comment(text) => pending_comments.push(text),
            KVNToken::Empty => {}
            KVNToken::DataLine(_) => {}
        }
    }

    // Comments trailing the final keyword introduce no further section.
    file_comments!(last_section);

    let header = ODMHeader {
        format_version: format_version
            .ok_or_else(|| ccsds_missing_field("OMM", "CCSDS_OMM_VERS"))?,
        classification,
        creation_date: creation_date.ok_or_else(|| ccsds_missing_field("OMM", "CREATION_DATE"))?,
        originator: originator.ok_or_else(|| ccsds_missing_field("OMM", "ORIGINATOR"))?,
        message_id,
        comments: header_comments,
    };

    let metadata = OMMMetadata {
        object_name: object_name.ok_or_else(|| ccsds_missing_field("OMM", "OBJECT_NAME"))?,
        object_id: object_id.ok_or_else(|| ccsds_missing_field("OMM", "OBJECT_ID"))?,
        center_name: center_name.ok_or_else(|| ccsds_missing_field("OMM", "CENTER_NAME"))?,
        ref_frame: ref_frame.ok_or_else(|| ccsds_missing_field("OMM", "REF_FRAME"))?,
        ref_frame_epoch: ref_frame_epoch
            .map(|raw| parse_ccsds_datetime(&raw, &active_ts(&time_system)))
            .transpose()?,
        time_system: time_system.ok_or_else(|| ccsds_missing_field("OMM", "TIME_SYSTEM"))?,
        mean_element_theory: mean_element_theory
            .ok_or_else(|| ccsds_missing_field("OMM", "MEAN_ELEMENT_THEORY"))?,
        comments: metadata_comments,
    };

    let mean_elements = OMMeanElements {
        epoch: epoch.ok_or_else(|| ccsds_missing_field("OMM", "EPOCH"))?,
        mean_motion,
        semi_major_axis,
        eccentricity: eccentricity.ok_or_else(|| ccsds_missing_field("OMM", "ECCENTRICITY"))?,
        inclination: inclination.ok_or_else(|| ccsds_missing_field("OMM", "INCLINATION"))?,
        ra_of_asc_node: ra_of_asc_node
            .ok_or_else(|| ccsds_missing_field("OMM", "RA_OF_ASC_NODE"))?,
        arg_of_pericenter: arg_of_pericenter
            .ok_or_else(|| ccsds_missing_field("OMM", "ARG_OF_PERICENTER"))?,
        mean_anomaly: mean_anomaly.ok_or_else(|| ccsds_missing_field("OMM", "MEAN_ANOMALY"))?,
        gm,
        comments: mean_element_comments,
    };

    let tle_parameters = if ephemeris_type.is_some()
        || norad_cat_id.is_some()
        || bstar.is_some()
        || mean_motion_dot.is_some()
    {
        Some(OMMTleParameters {
            ephemeris_type,
            classification_type,
            norad_cat_id,
            element_set_no,
            rev_at_epoch,
            bstar,
            bterm,
            mean_motion_dot,
            mean_motion_ddot,
            agom,
            comments: tle_comments,
        })
    } else {
        None
    };

    let spacecraft_parameters = if mass.is_some() || solar_rad_area.is_some() || drag_area.is_some()
    {
        Some(CCSDSSpacecraftParameters {
            mass,
            solar_rad_area,
            solar_rad_coeff,
            drag_area,
            drag_coeff,
            comments: spacecraft_comments,
        })
    } else {
        None
    };

    let covariance = if cov_values.len() == 21 {
        let mut vals = [0.0_f64; 21];
        vals.copy_from_slice(&cov_values);
        let matrix = covariance_from_lower_triangular(&vals, 1e6);
        Some(CCSDSCovariance {
            epoch: None,
            cov_ref_frame,
            matrix,
            comments: cov_comments,
        })
    } else {
        None
    };

    let user_def = if user_defined.is_empty() {
        None
    } else {
        Some(CCSDSUserDefined {
            parameters: user_defined,
        })
    };

    Ok(OMM {
        header,
        metadata,
        mean_elements,
        tle_parameters,
        spacecraft_parameters,
        covariance,
        user_defined: user_def,
        comments: Vec::new(),
    })
}

/// Parse scientific notation that may use Fortran-style format (e.g., "-.47102E-5").
fn parse_scientific_notation(s: &str) -> Result<f64, BraheError> {
    let s = s.trim();
    // Handle Fortran-style: ".47102E-5" or "-.47102E-5"
    s.parse::<f64>()
        .map_err(|_| ccsds_parse_error("OMM", &format!("invalid numeric value '{}'", s)))
}

/// Write an OMM message to KVN format.
pub fn write_omm(omm: &crate::ccsds::omm::OMM) -> Result<String, BraheError> {
    let mut out = String::new();

    // Header
    out.push_str(&format!(
        "CCSDS_OMM_VERS = {:.1}\n",
        omm.header.format_version
    ));
    for comment in &omm.header.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }
    if let Some(ref class) = omm.header.classification {
        out.push_str(&format!("CLASSIFICATION = {}\n", class));
    }
    out.push_str(&format!(
        "CREATION_DATE = {}\n",
        format_ccsds_datetime_in(&omm.header.creation_date, &CCSDSTimeSystem::UTC)
    ));
    out.push_str(&format!("ORIGINATOR = {}\n", omm.header.originator));
    if let Some(ref msg_id) = omm.header.message_id {
        out.push_str(&format!("MESSAGE_ID = {}\n", msg_id));
    }
    out.push('\n');

    // Metadata comments
    for comment in &omm.metadata.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }
    out.push_str(&format!("OBJECT_NAME = {}\n", omm.metadata.object_name));
    out.push_str(&format!("OBJECT_ID = {}\n", omm.metadata.object_id));
    out.push_str(&format!("CENTER_NAME = {}\n", omm.metadata.center_name));
    out.push_str(&format!("REF_FRAME = {}\n", omm.metadata.ref_frame));
    if let Some(ref epoch) = omm.metadata.ref_frame_epoch {
        out.push_str(&format!(
            "REF_FRAME_EPOCH = {}\n",
            format_ccsds_datetime_in(epoch, &omm.metadata.time_system)
        ));
    }
    out.push_str(&format!("TIME_SYSTEM = {}\n", omm.metadata.time_system));
    out.push_str(&format!(
        "MEAN_ELEMENT_THEORY = {}\n",
        omm.metadata.mean_element_theory
    ));
    out.push('\n');

    // Mean elements
    for comment in &omm.mean_elements.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }
    out.push_str(&format!(
        "EPOCH = {}\n",
        format_ccsds_datetime_in(&omm.mean_elements.epoch, &omm.metadata.time_system)
    ));
    if let Some(mm) = omm.mean_elements.mean_motion {
        out.push_str(&format!("MEAN_MOTION = {}\n", mm));
    }
    if let Some(sma) = omm.mean_elements.semi_major_axis {
        out.push_str(&format!("SEMI_MAJOR_AXIS = {}\n", sma));
    }
    out.push_str(&format!(
        "ECCENTRICITY = {}\n",
        omm.mean_elements.eccentricity
    ));
    out.push_str(&format!(
        "INCLINATION = {}\n",
        omm.mean_elements.inclination
    ));
    out.push_str(&format!(
        "RA_OF_ASC_NODE = {}\n",
        omm.mean_elements.ra_of_asc_node
    ));
    out.push_str(&format!(
        "ARG_OF_PERICENTER = {}\n",
        omm.mean_elements.arg_of_pericenter
    ));
    out.push_str(&format!(
        "MEAN_ANOMALY = {}\n",
        omm.mean_elements.mean_anomaly
    ));
    if let Some(gm) = omm.mean_elements.gm {
        // Internal m³/s² → CCSDS km³/s²
        out.push_str(&format!("GM = {}\n", gm / 1e9));
    }

    // TLE parameters
    if let Some(ref tle) = omm.tle_parameters {
        out.push('\n');
        for comment in &tle.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        if let Some(et) = tle.ephemeris_type {
            out.push_str(&format!("EPHEMERIS_TYPE = {}\n", et));
        }
        if let Some(ct) = tle.classification_type {
            out.push_str(&format!("CLASSIFICATION_TYPE = {}\n", ct));
        }
        if let Some(id) = tle.norad_cat_id {
            out.push_str(&format!("NORAD_CAT_ID = {}\n", id));
        }
        if let Some(esn) = tle.element_set_no {
            out.push_str(&format!("ELEMENT_SET_NO = {}\n", esn));
        }
        if let Some(rev) = tle.rev_at_epoch {
            out.push_str(&format!("REV_AT_EPOCH = {}\n", rev));
        }
        if let Some(bs) = tle.bstar {
            out.push_str(&format!("BSTAR = {}\n", bs));
        }
        if let Some(bt) = tle.bterm {
            out.push_str(&format!("BTERM = {}\n", bt));
        }
        if let Some(mmd) = tle.mean_motion_dot {
            out.push_str(&format!("MEAN_MOTION_DOT = {}\n", mmd));
        }
        if let Some(mmdd) = tle.mean_motion_ddot {
            out.push_str(&format!("MEAN_MOTION_DDOT = {}\n", mmdd));
        }
        if let Some(ag) = tle.agom {
            out.push_str(&format!("AGOM = {}\n", ag));
        }
    }

    // Spacecraft parameters
    write_kvn_spacecraft_params(&mut out, &omm.spacecraft_parameters);

    // Covariance (OMM uses flat key=value pairs, no COVARIANCE_START/STOP)
    if let Some(ref cov) = omm.covariance {
        out.push('\n');
        for comment in &cov.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        if let Some(ref frame) = cov.cov_ref_frame {
            out.push_str(&format!("COV_REF_FRAME = {}\n", frame));
        }
        write_kvn_covariance_elements(&mut out, &cov.matrix);
    }

    // User-defined parameters
    write_kvn_user_defined(&mut out, &omm.user_defined);

    // Comments at message level
    for comment in &omm.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }

    Ok(out)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::ccsds::common::CCSDSTimeSystem;
    use crate::ccsds::kvn::parse_omm;

    use serial_test::parallel;
    // OMM Tests

    #[test]
    #[parallel]
    fn test_parse_omm_example1() {
        let content = std::fs::read_to_string("test_assets/ccsds/omm/OMMExample1.txt").unwrap();
        let omm = parse_omm(&content).unwrap();

        assert!((omm.header.format_version - 3.0).abs() < 1e-10);
        assert_eq!(omm.header.originator, "NOAA/USA");
        assert!(omm.header.message_id.is_none());

        assert_eq!(omm.metadata.object_name, "GOES 9");
        assert_eq!(omm.metadata.object_id, "1995-025A");
        assert_eq!(omm.metadata.center_name, "EARTH");
        assert_eq!(omm.metadata.ref_frame, CCSDSRefFrame::TEME);
        assert_eq!(omm.metadata.time_system, CCSDSTimeSystem::UTC);
        assert_eq!(omm.metadata.mean_element_theory, "SGP/SGP4");

        // Mean elements
        assert!(omm.mean_elements.mean_motion.is_some());
        assert!((omm.mean_elements.mean_motion.unwrap() - 1.00273272).abs() < 1e-10);
        assert!((omm.mean_elements.eccentricity - 0.0005013).abs() < 1e-10);
        assert!((omm.mean_elements.inclination - 3.0539).abs() < 1e-4);
        assert!((omm.mean_elements.ra_of_asc_node - 81.7939).abs() < 1e-4);
        assert!((omm.mean_elements.arg_of_pericenter - 249.2363).abs() < 1e-4);
        assert!((omm.mean_elements.mean_anomaly - 150.1602).abs() < 1e-4);
        // GM: 398600.8 km³/s² → 398600.8e9 m³/s²
        assert!((omm.mean_elements.gm.unwrap() - 398600.8e9).abs() < 1e3);

        // TLE parameters
        let tle = omm.tle_parameters.as_ref().unwrap();
        assert_eq!(tle.ephemeris_type, Some(0));
        assert_eq!(tle.classification_type, Some('U'));
        assert_eq!(tle.norad_cat_id, Some(23581));
        assert_eq!(tle.element_set_no, Some(925));
        assert_eq!(tle.rev_at_epoch, Some(4316));
        assert!((tle.bstar.unwrap() - 0.0001).abs() < 1e-10);
        assert!((tle.mean_motion_dot.unwrap() - (-0.00000113)).abs() < 1e-12);
        assert!((tle.mean_motion_ddot.unwrap() - 0.0).abs() < 1e-15);
    }

    #[test]
    #[parallel]
    fn test_parse_omm_example2_with_covariance() {
        let content = std::fs::read_to_string("test_assets/ccsds/omm/OMMExample2.txt").unwrap();
        let omm = parse_omm(&content).unwrap();

        assert_eq!(omm.metadata.object_name, "GOES 9");
        assert!(omm.covariance.is_some());
        let cov = omm.covariance.as_ref().unwrap();
        assert_eq!(cov.cov_ref_frame.as_ref().unwrap(), &CCSDSRefFrame::TEME);
        // CX_X = 3.331349476038534e-04 km² → * 1e6 m²
        assert!((cov.matrix[(0, 0)] - 3.331349476038534e-04 * 1e6).abs() < 1e-2);
    }

    #[test]
    #[parallel]
    fn test_parse_omm_example3_unsupported_time_system() {
        // OMMExample3.txt uses TIME_SYSTEM = MRT which is not supported for epoch conversion
        let content = std::fs::read_to_string("test_assets/ccsds/omm/OMMExample3.txt").unwrap();
        let result = parse_omm(&content);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("MRT"),
            "Error should mention unsupported time system MRT: {}",
            err_msg
        );
    }

    #[test]
    #[parallel]
    fn test_parse_omm_example4() {
        let content = std::fs::read_to_string("test_assets/ccsds/omm/OMMExample4.txt").unwrap();
        let omm = parse_omm(&content).unwrap();

        assert_eq!(omm.metadata.object_name, "STARLETTE");
        assert_eq!(omm.metadata.object_id, "1975-010A");
        assert!((omm.mean_elements.mean_motion.unwrap() - 13.82309053).abs() < 1e-8);
        assert!((omm.mean_elements.eccentricity - 0.0205751).abs() < 1e-7);

        let tle = omm.tle_parameters.as_ref().unwrap();
        assert_eq!(tle.norad_cat_id, Some(7646));
        // BSTAR: -.47102E-5 = -4.7102e-6
        assert!((tle.bstar.unwrap() - (-4.7102e-6)).abs() < 1e-12);
    }

    #[test]
    #[parallel]
    fn test_parse_omm_example5_sgp4xp() {
        let content = std::fs::read_to_string("test_assets/ccsds/omm/OMMExample5.txt").unwrap();
        let omm = parse_omm(&content).unwrap();

        assert_eq!(omm.metadata.mean_element_theory, "SGP4-XP");
        let tle = omm.tle_parameters.as_ref().unwrap();
        assert_eq!(tle.ephemeris_type, Some(4));
        assert!((tle.bterm.unwrap() - 0.0015).abs() < 1e-10);
        assert!((tle.agom.unwrap() - 0.001).abs() < 1e-10);
    }

    #[test]
    #[serial_test::parallel]
    fn test_parse_omm_attributes_comments_to_the_block_they_introduce() {
        let omm = parse_omm(
            &std::fs::read_to_string("test_assets/ccsds/omm/OMM-section-comments.txt").unwrap(),
        )
        .unwrap();

        assert_eq!(omm.header.comments, vec!["header comment"]);
        assert_eq!(omm.metadata.comments, vec!["metadata comment"]);
        assert_eq!(omm.mean_elements.comments, vec!["mean element comment"]);
        assert_eq!(
            omm.tle_parameters.as_ref().unwrap().comments,
            vec!["tle comment"]
        );
        assert_eq!(
            omm.spacecraft_parameters.as_ref().unwrap().comments,
            vec!["spacecraft comment"]
        );
    }
}
