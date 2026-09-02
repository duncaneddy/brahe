/*!
 * KVN reader and writer for the Orbit Parameter Message (OPM).
 *
 * Reference: CCSDS 502.0-B-3 (Orbit Data Messages), section 3
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
use crate::ccsds::opm::{OPM, OPMKeplerianElements, OPMManeuver, OPMMetadata, OPMStateVector};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

/// The OPM section a keyword belongs to.
#[derive(Clone, Copy, PartialEq)]
enum OPMSection {
    Header,
    Metadata,
    StateVector,
    KeplerianElements,
    SpacecraftParameters,
    Covariance,
    Maneuver,
}

/// The OPM section a keyword introduces, or `None` when the keyword is not one
/// this parser recognizes.
///
/// Keywords are grouped as CCSDS 502.0-B-3 tables 3-1 (header), 3-2 (metadata),
/// and 3-3 (data) list them.
///
/// # Arguments
///
/// * `key` - The KVN keyword, without surrounding whitespace.
///
/// # Returns
///
/// * `Option<OPMSection>` - The owning section, or `None` if unrecognized.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(opm_section("ORIGINATOR"), Some(OPMSection::Header));
/// assert_eq!(opm_section("X_DOT"), Some(OPMSection::StateVector));
/// assert_eq!(opm_section("COV_REF_FRAME"), Some(OPMSection::Covariance));
/// // EPOCH appears in more than one block, so it names none on its own.
/// assert_eq!(opm_section("EPOCH"), None);
/// ```
fn opm_section(key: &str) -> Option<OPMSection> {
    Some(match key {
        "CCSDS_OPM_VERS" | "CLASSIFICATION" | "CREATION_DATE" | "ORIGINATOR" | "MESSAGE_ID" => {
            OPMSection::Header
        }

        "OBJECT_NAME" | "OBJECT_ID" | "CENTER_NAME" | "REF_FRAME" | "REF_FRAME_EPOCH"
        | "TIME_SYSTEM" => OPMSection::Metadata,

        // EPOCH names the state vector in a conforming OPM, but a message may
        // also carry one inside a covariance block, so it cannot pick a section
        // on its own. Leaving it unclassified holds any pending comments until
        // a keyword that names exactly one block arrives.
        "EPOCH" => return None,

        "X" | "Y" | "Z" | "X_DOT" | "Y_DOT" | "Z_DOT" => OPMSection::StateVector,

        "SEMI_MAJOR_AXIS" | "ECCENTRICITY" | "INCLINATION" | "RA_OF_ASC_NODE"
        | "ARG_OF_PERICENTER" | "TRUE_ANOMALY" | "MEAN_ANOMALY" | "GM" => {
            OPMSection::KeplerianElements
        }

        "MASS" | "SOLAR_RAD_AREA" | "SOLAR_RAD_COEFF" | "DRAG_AREA" | "DRAG_COEFF" => {
            OPMSection::SpacecraftParameters
        }

        "COV_REF_FRAME" => OPMSection::Covariance,
        k if k.starts_with("CX_") || k.starts_with("CY_") || k.starts_with("CZ_") => {
            OPMSection::Covariance
        }

        "MAN_EPOCH_IGNITION" | "MAN_DURATION" | "MAN_DELTA_MASS" | "MAN_REF_FRAME" | "MAN_DV_1"
        | "MAN_DV_2" | "MAN_DV_3" => OPMSection::Maneuver,

        _ => return None,
    })
}

/// Parse an OPM message from KVN format.
pub fn parse_opm(content: &str) -> Result<OPM, BraheError> {
    let mut header_comments: Vec<String> = Vec::new();
    let mut metadata_comments: Vec<String> = Vec::new();
    let mut state_comments: Vec<String> = Vec::new();
    let mut kep_comments: Vec<String> = Vec::new();
    let mut spacecraft_comments: Vec<String> = Vec::new();
    let mut cov_comments: Vec<String> = Vec::new();
    let mut maneuver_comments: Vec<String> = Vec::new();

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

    // State vector
    let mut sv_epoch: Option<Epoch> = None;
    let mut sv_x: Option<f64> = None;
    let mut sv_y: Option<f64> = None;
    let mut sv_z: Option<f64> = None;
    let mut sv_vx: Option<f64> = None;
    let mut sv_vy: Option<f64> = None;
    let mut sv_vz: Option<f64> = None;

    // Keplerian elements
    let mut kep_sma: Option<f64> = None;
    let mut kep_ecc: Option<f64> = None;
    let mut kep_inc: Option<f64> = None;
    let mut kep_raan: Option<f64> = None;
    let mut kep_argp: Option<f64> = None;
    let mut kep_ta: Option<f64> = None;
    let mut kep_ma: Option<f64> = None;
    let mut kep_gm: Option<f64> = None;

    // Spacecraft
    let mut mass: Option<f64> = None;
    let mut solar_rad_area: Option<f64> = None;
    let mut solar_rad_coeff: Option<f64> = None;
    let mut drag_area: Option<f64> = None;
    let mut drag_coeff: Option<f64> = None;

    // Covariance
    let mut cov_ref_frame: Option<CCSDSRefFrame> = None;
    let mut cov_values: Vec<f64> = Vec::new();

    // Maneuvers
    let mut maneuvers: Vec<OPMManeuver> = Vec::new();
    let mut man_epoch: Option<Epoch> = None;
    let mut man_duration: Option<f64> = None;
    let mut man_delta_mass: Option<f64> = None;
    let mut man_ref_frame: Option<CCSDSRefFrame> = None;
    let mut man_dv1: Option<f64> = None;
    let mut man_dv2: Option<f64> = None;
    let mut man_dv3: Option<f64> = None;

    // User-defined
    let mut user_defined: HashMap<String, String> = HashMap::new();

    // Comments accumulate until a keyword names the section they introduce.
    // CCSDS 502.0-B-3 tables 3-1 through 3-3 open every OPM section with a
    // COMMENT row and subsection 7.4.8 fixes the keyword order to match, so a
    // comment belongs to the section that follows it, not the one before it.
    let mut pending_comments: Vec<String> = Vec::new();
    let mut last_section = OPMSection::Header;

    let active_ts = |ts: &Option<CCSDSTimeSystem>| ts.clone().unwrap_or(CCSDSTimeSystem::UTC);

    let flush_maneuver = |man_epoch: &mut Option<Epoch>,
                          man_duration: &mut Option<f64>,
                          man_delta_mass: &mut Option<f64>,
                          man_ref_frame: &mut Option<CCSDSRefFrame>,
                          man_dv1: &mut Option<f64>,
                          man_dv2: &mut Option<f64>,
                          man_dv3: &mut Option<f64>,
                          maneuvers: &mut Vec<OPMManeuver>,
                          comments: &mut Vec<String>| {
        if let (Some(epoch), Some(dur), Some(frame), Some(dv1), Some(dv2), Some(dv3)) = (
            man_epoch.take(),
            man_duration.take(),
            man_ref_frame.take(),
            man_dv1.take(),
            man_dv2.take(),
            man_dv3.take(),
        ) {
            maneuvers.push(OPMManeuver {
                epoch_ignition: epoch,
                duration: dur,
                delta_mass: man_delta_mass.take(),
                ref_frame: frame,
                dv: [dv1 * 1000.0, dv2 * 1000.0, dv3 * 1000.0], // km/s → m/s
                comments: std::mem::take(comments),
            });
        }
    };

    macro_rules! file_comments {
        ($section:expr) => {{
            if !pending_comments.is_empty() {
                let sink = match $section {
                    OPMSection::Header => &mut header_comments,
                    OPMSection::Metadata => &mut metadata_comments,
                    OPMSection::StateVector => &mut state_comments,
                    OPMSection::KeplerianElements => &mut kep_comments,
                    OPMSection::SpacecraftParameters => &mut spacecraft_comments,
                    OPMSection::Covariance => &mut cov_comments,
                    OPMSection::Maneuver => &mut maneuver_comments,
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
                // MAN_EPOCH_IGNITION closes the preceding maneuver and drains
                // its comments, so the comments introducing the new one are
                // filed after that keyword has been dispatched, not before.
                let section = opm_section(&key);
                if let Some(section) = section
                    && key != "MAN_EPOCH_IGNITION"
                {
                    last_section = section;
                    file_comments!(section);
                }
                match key.as_str() {
                    "CCSDS_OPM_VERS" => {
                        format_version = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid version"))?,
                        );
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
                    "CLASSIFICATION" => {
                        classification = Some(val.to_string());
                    }

                    "OBJECT_NAME" => {
                        object_name = Some(val.to_string());
                    }
                    "OBJECT_ID" => {
                        object_id = Some(val.to_string());
                    }
                    "CENTER_NAME" => {
                        center_name = Some(val.to_string());
                    }
                    "REF_FRAME" if ref_frame.is_none() => {
                        ref_frame = Some(CCSDSRefFrame::parse(val));
                    }
                    "REF_FRAME_EPOCH" => ref_frame_epoch = Some(val.to_string()),
                    "TIME_SYSTEM" => {
                        time_system = Some(CCSDSTimeSystem::parse(val)?);
                    }

                    "EPOCH" => {
                        sv_epoch = Some(parse_ccsds_datetime(val, &active_ts(&time_system))?);
                    }
                    "X" => {
                        sv_x = Some(
                            val.parse::<f64>()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid X"))?
                                * 1000.0,
                        );
                    }
                    "Y" => {
                        sv_y = Some(
                            val.parse::<f64>()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid Y"))?
                                * 1000.0,
                        );
                    }
                    "Z" => {
                        sv_z = Some(
                            val.parse::<f64>()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid Z"))?
                                * 1000.0,
                        );
                    }
                    "X_DOT" => {
                        sv_vx = Some(
                            val.parse::<f64>()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid X_DOT"))?
                                * 1000.0,
                        );
                    }
                    "Y_DOT" => {
                        sv_vy = Some(
                            val.parse::<f64>()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid Y_DOT"))?
                                * 1000.0,
                        );
                    }
                    "Z_DOT" => {
                        sv_vz = Some(
                            val.parse::<f64>()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid Z_DOT"))?
                                * 1000.0,
                        );
                    }

                    "SEMI_MAJOR_AXIS" => {
                        kep_sma = Some(
                            val.parse::<f64>()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid SMA"))?
                                * 1000.0,
                        );
                    } // km → m
                    "ECCENTRICITY" if kep_sma.is_some() || kep_ecc.is_none() => {
                        kep_ecc = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid ECC"))?,
                        );
                    }
                    "INCLINATION" if kep_sma.is_some() => {
                        kep_inc = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid INC"))?,
                        );
                    }
                    "RA_OF_ASC_NODE" if kep_sma.is_some() => {
                        kep_raan = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid RAAN"))?,
                        );
                    }
                    "ARG_OF_PERICENTER" if kep_sma.is_some() => {
                        kep_argp = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid ARGP"))?,
                        );
                    }
                    "TRUE_ANOMALY" => {
                        kep_ta = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid TA"))?,
                        );
                    }
                    "MEAN_ANOMALY" => {
                        kep_ma = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid MA"))?,
                        );
                    }
                    "GM" => {
                        kep_gm = Some(
                            val.parse::<f64>()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid GM"))?
                                * 1e9,
                        );
                    } // km³/s² → m³/s²

                    "MASS" => {
                        mass = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid MASS"))?,
                        );
                    }
                    "SOLAR_RAD_AREA" => {
                        solar_rad_area = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid SOLAR_RAD_AREA"))?,
                        );
                    }
                    "SOLAR_RAD_COEFF" => {
                        solar_rad_coeff =
                            Some(val.parse().map_err(|_| {
                                ccsds_parse_error("OPM", "invalid SOLAR_RAD_COEFF")
                            })?);
                    }
                    "DRAG_AREA" => {
                        drag_area = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid DRAG_AREA"))?,
                        );
                    }
                    "DRAG_COEFF" => {
                        drag_coeff = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid DRAG_COEFF"))?,
                        );
                    }

                    "COV_REF_FRAME" => {
                        cov_ref_frame = Some(CCSDSRefFrame::parse(val));
                    }
                    k if k.starts_with("CX_") || k.starts_with("CY_") || k.starts_with("CZ_") => {
                        let v: f64 = val.parse().map_err(|_| {
                            ccsds_parse_error("OPM", &format!("invalid cov value '{}'", val))
                        })?;
                        cov_values.push(v);
                    }

                    "MAN_EPOCH_IGNITION" => {
                        // Flush previous maneuver
                        flush_maneuver(
                            &mut man_epoch,
                            &mut man_duration,
                            &mut man_delta_mass,
                            &mut man_ref_frame,
                            &mut man_dv1,
                            &mut man_dv2,
                            &mut man_dv3,
                            &mut maneuvers,
                            &mut maneuver_comments,
                        );
                        man_epoch = Some(parse_ccsds_datetime(val, &active_ts(&time_system))?);
                    }
                    "MAN_DURATION" => {
                        man_duration = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid MAN_DURATION"))?,
                        );
                    }
                    "MAN_DELTA_MASS" => {
                        man_delta_mass = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid MAN_DELTA_MASS"))?,
                        );
                    }
                    "MAN_REF_FRAME" => {
                        man_ref_frame = Some(CCSDSRefFrame::parse(val));
                    }
                    "MAN_DV_1" => {
                        man_dv1 = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid MAN_DV_1"))?,
                        );
                    }
                    "MAN_DV_2" => {
                        man_dv2 = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid MAN_DV_2"))?,
                        );
                    }
                    "MAN_DV_3" => {
                        man_dv3 = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid MAN_DV_3"))?,
                        );
                    }

                    k if k.starts_with("USER_DEFINED_") => {
                        let param_name = k.strip_prefix("USER_DEFINED_").unwrap_or(k);
                        user_defined.insert(param_name.to_string(), val.to_string());
                    }

                    _ => {}
                }
                if let Some(section) = section
                    && key == "MAN_EPOCH_IGNITION"
                {
                    last_section = section;
                    file_comments!(section);
                }
            }
            KVNToken::Comment(text) => pending_comments.push(text),
            KVNToken::Empty | KVNToken::DataLine(_) => {}
        }
    }

    // Comments trailing the final keyword introduce no further section.
    file_comments!(last_section);

    // Flush last maneuver
    flush_maneuver(
        &mut man_epoch,
        &mut man_duration,
        &mut man_delta_mass,
        &mut man_ref_frame,
        &mut man_dv1,
        &mut man_dv2,
        &mut man_dv3,
        &mut maneuvers,
        &mut maneuver_comments,
    );

    let header = ODMHeader {
        format_version: format_version
            .ok_or_else(|| ccsds_missing_field("OPM", "CCSDS_OPM_VERS"))?,
        classification,
        creation_date: creation_date.ok_or_else(|| ccsds_missing_field("OPM", "CREATION_DATE"))?,
        originator: originator.ok_or_else(|| ccsds_missing_field("OPM", "ORIGINATOR"))?,
        message_id,
        comments: header_comments,
    };

    let state_vector = OPMStateVector {
        epoch: sv_epoch.ok_or_else(|| ccsds_missing_field("OPM", "EPOCH"))?,
        position: [
            sv_x.ok_or_else(|| ccsds_missing_field("OPM", "X"))?,
            sv_y.ok_or_else(|| ccsds_missing_field("OPM", "Y"))?,
            sv_z.ok_or_else(|| ccsds_missing_field("OPM", "Z"))?,
        ],
        velocity: [
            sv_vx.ok_or_else(|| ccsds_missing_field("OPM", "X_DOT"))?,
            sv_vy.ok_or_else(|| ccsds_missing_field("OPM", "Y_DOT"))?,
            sv_vz.ok_or_else(|| ccsds_missing_field("OPM", "Z_DOT"))?,
        ],
        comments: state_comments,
    };

    let keplerian_elements = if let Some(sma) = kep_sma {
        Some(OPMKeplerianElements {
            semi_major_axis: sma,
            eccentricity: kep_ecc.ok_or_else(|| ccsds_missing_field("OPM", "ECCENTRICITY"))?,
            inclination: kep_inc.ok_or_else(|| ccsds_missing_field("OPM", "INCLINATION"))?,
            ra_of_asc_node: kep_raan.ok_or_else(|| ccsds_missing_field("OPM", "RA_OF_ASC_NODE"))?,
            arg_of_pericenter: kep_argp
                .ok_or_else(|| ccsds_missing_field("OPM", "ARG_OF_PERICENTER"))?,
            true_anomaly: kep_ta,
            mean_anomaly: kep_ma,
            gm: kep_gm,
            comments: kep_comments,
        })
    } else {
        None
    };

    let spacecraft_parameters = if mass.is_some() || solar_rad_area.is_some() {
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

    Ok(OPM {
        header,
        metadata: OPMMetadata {
            object_name: object_name.ok_or_else(|| ccsds_missing_field("OPM", "OBJECT_NAME"))?,
            object_id: object_id.ok_or_else(|| ccsds_missing_field("OPM", "OBJECT_ID"))?,
            center_name: center_name.ok_or_else(|| ccsds_missing_field("OPM", "CENTER_NAME"))?,
            ref_frame: ref_frame.ok_or_else(|| ccsds_missing_field("OPM", "REF_FRAME"))?,
            ref_frame_epoch: ref_frame_epoch
                .map(|raw| parse_ccsds_datetime(&raw, &active_ts(&time_system)))
                .transpose()?,
            time_system: time_system.ok_or_else(|| ccsds_missing_field("OPM", "TIME_SYSTEM"))?,
            comments: metadata_comments,
        },
        state_vector,
        keplerian_elements,
        spacecraft_parameters,
        covariance,
        maneuvers,
        user_defined: user_def,
    })
}

/// Write an OPM message to KVN format.
pub fn write_opm(opm: &crate::ccsds::opm::OPM) -> Result<String, BraheError> {
    let mut out = String::new();

    // Header
    out.push_str(&format!(
        "CCSDS_OPM_VERS = {:.1}\n",
        opm.header.format_version
    ));
    for comment in &opm.header.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }
    if let Some(ref class) = opm.header.classification {
        out.push_str(&format!("CLASSIFICATION = {}\n", class));
    }
    out.push_str(&format!(
        "CREATION_DATE = {}\n",
        format_ccsds_datetime_in(&opm.header.creation_date, &CCSDSTimeSystem::UTC)
    ));
    out.push_str(&format!("ORIGINATOR = {}\n", opm.header.originator));
    if let Some(ref msg_id) = opm.header.message_id {
        out.push_str(&format!("MESSAGE_ID = {}\n", msg_id));
    }
    out.push('\n');

    // Metadata comments
    for comment in &opm.metadata.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }
    out.push_str(&format!("OBJECT_NAME = {}\n", opm.metadata.object_name));
    out.push_str(&format!("OBJECT_ID = {}\n", opm.metadata.object_id));
    out.push_str(&format!("CENTER_NAME = {}\n", opm.metadata.center_name));
    out.push_str(&format!("REF_FRAME = {}\n", opm.metadata.ref_frame));
    if let Some(ref epoch) = opm.metadata.ref_frame_epoch {
        out.push_str(&format!(
            "REF_FRAME_EPOCH = {}\n",
            format_ccsds_datetime_in(epoch, &opm.metadata.time_system)
        ));
    }
    out.push_str(&format!("TIME_SYSTEM = {}\n", opm.metadata.time_system));

    // State vector
    for comment in &opm.state_vector.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }
    out.push_str(&format!(
        "EPOCH = {}\n",
        format_ccsds_datetime_in(&opm.state_vector.epoch, &opm.metadata.time_system)
    ));
    // Position: m → km
    out.push_str(&format!("X = {:.6}\n", opm.state_vector.position[0] / 1e3));
    out.push_str(&format!("Y = {:.6}\n", opm.state_vector.position[1] / 1e3));
    out.push_str(&format!("Z = {:.6}\n", opm.state_vector.position[2] / 1e3));
    // Velocity: m/s → km/s
    out.push_str(&format!(
        "X_DOT = {:.6}\n",
        opm.state_vector.velocity[0] / 1e3
    ));
    out.push_str(&format!(
        "Y_DOT = {:.6}\n",
        opm.state_vector.velocity[1] / 1e3
    ));
    out.push_str(&format!(
        "Z_DOT = {:.6}\n",
        opm.state_vector.velocity[2] / 1e3
    ));

    // Keplerian elements
    if let Some(ref ke) = opm.keplerian_elements {
        for comment in &ke.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        // Semi-major axis: m → km
        out.push_str(&format!(
            "SEMI_MAJOR_AXIS = {:.6}\n",
            ke.semi_major_axis / 1e3
        ));
        out.push_str(&format!("ECCENTRICITY = {}\n", ke.eccentricity));
        out.push_str(&format!("INCLINATION = {}\n", ke.inclination));
        out.push_str(&format!("RA_OF_ASC_NODE = {}\n", ke.ra_of_asc_node));
        out.push_str(&format!("ARG_OF_PERICENTER = {}\n", ke.arg_of_pericenter));
        if let Some(ta) = ke.true_anomaly {
            out.push_str(&format!("TRUE_ANOMALY = {}\n", ta));
        }
        if let Some(ma) = ke.mean_anomaly {
            out.push_str(&format!("MEAN_ANOMALY = {}\n", ma));
        }
        if let Some(gm) = ke.gm {
            // m³/s² → km³/s²
            out.push_str(&format!("GM = {}\n", gm / 1e9));
        }
    }

    // Spacecraft parameters
    write_kvn_spacecraft_params(&mut out, &opm.spacecraft_parameters);

    // Covariance (OPM uses flat CX_*/CY_*/CZ_* key=value pairs)
    if let Some(ref cov) = opm.covariance {
        out.push('\n');
        for comment in &cov.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        if let Some(ref frame) = cov.cov_ref_frame {
            out.push_str(&format!("COV_REF_FRAME = {}\n", frame));
        }
        write_kvn_covariance_elements(&mut out, &cov.matrix);
    }

    // Maneuvers
    for man in &opm.maneuvers {
        out.push('\n');
        for comment in &man.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        out.push_str(&format!(
            "MAN_EPOCH_IGNITION = {}\n",
            format_ccsds_datetime_in(&man.epoch_ignition, &opm.metadata.time_system)
        ));
        out.push_str(&format!("MAN_DURATION = {:.2}\n", man.duration));
        if let Some(dm) = man.delta_mass {
            out.push_str(&format!("MAN_DELTA_MASS = {:.3}\n", dm));
        }
        out.push_str(&format!("MAN_REF_FRAME = {}\n", man.ref_frame));
        // DV: m/s → km/s
        out.push_str(&format!("MAN_DV_1 = {:.8}\n", man.dv[0] / 1e3));
        out.push_str(&format!("MAN_DV_2 = {:.8}\n", man.dv[1] / 1e3));
        out.push_str(&format!("MAN_DV_3 = {:.8}\n", man.dv[2] / 1e3));
    }

    // User-defined parameters
    write_kvn_user_defined(&mut out, &opm.user_defined);

    Ok(out)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::ccsds::common::CCSDSTimeSystem;
    use crate::ccsds::kvn::parse_opm;

    use serial_test::parallel;
    // OPM Tests

    #[test]
    #[parallel]
    fn test_parse_opm_example1() {
        let content = std::fs::read_to_string("test_assets/ccsds/opm/OPMExample1.txt").unwrap();
        let opm = parse_opm(&content).unwrap();

        assert!((opm.header.format_version - 3.0).abs() < 1e-10);
        assert_eq!(opm.header.originator, "JAXA");
        assert_eq!(opm.metadata.object_name, "GODZILLA 5");
        assert_eq!(opm.metadata.object_id, "1998-999A");
        assert_eq!(opm.metadata.ref_frame, CCSDSRefFrame::ITRF2000);

        // State vector (km → m)
        assert!((opm.state_vector.position[0] - 6503514.0).abs() < 1.0);
        assert!((opm.state_vector.position[1] - 1239647.0).abs() < 1.0);
        assert!((opm.state_vector.position[2] - (-717490.0)).abs() < 1.0);
        assert!((opm.state_vector.velocity[0] - (-873.160)).abs() < 0.001);
        assert!((opm.state_vector.velocity[1] - 8740.420).abs() < 0.001);
        assert!((opm.state_vector.velocity[2] - (-4191.076)).abs() < 0.001);

        // Spacecraft parameters
        let sc = opm.spacecraft_parameters.as_ref().unwrap();
        assert!((sc.mass.unwrap() - 3000.0).abs() < 1e-3);
        assert!((sc.drag_coeff.unwrap() - 2.5).abs() < 1e-3);

        // No Keplerian, no maneuvers, no covariance
        assert!(opm.keplerian_elements.is_none());
        assert!(opm.maneuvers.is_empty());
        assert!(opm.covariance.is_none());
    }

    #[test]
    #[parallel]
    fn test_parse_opm_example2_with_keplerian_and_maneuvers() {
        let content = std::fs::read_to_string("test_assets/ccsds/opm/OPMExample2.txt").unwrap();
        let opm = parse_opm(&content).unwrap();

        assert_eq!(opm.metadata.object_name, "EUTELSAT W4");
        assert_eq!(opm.metadata.ref_frame, CCSDSRefFrame::TOD);

        // State vector
        assert!((opm.state_vector.position[0] - 6655994.2).abs() < 1.0);

        // Keplerian elements
        let kep = opm.keplerian_elements.as_ref().unwrap();
        assert!((kep.semi_major_axis - 41399512.3).abs() < 1.0); // 41399.5123 km → m
        assert!((kep.eccentricity - 0.020842611).abs() < 1e-9);
        assert!((kep.inclination - 0.117746).abs() < 1e-6);
        assert!(kep.true_anomaly.is_some());
        assert!((kep.true_anomaly.unwrap() - 41.922339).abs() < 1e-6);
        assert!((kep.gm.unwrap() - 398600.4415e9).abs() < 1e3);

        // 2 maneuvers
        assert_eq!(opm.maneuvers.len(), 2);
        let m1 = &opm.maneuvers[0];
        assert!((m1.duration - 132.60).abs() < 0.01);
        assert!((m1.delta_mass.unwrap() - (-18.418)).abs() < 0.001);
        assert_eq!(m1.ref_frame, CCSDSRefFrame::J2000);
        assert!((m1.dv[0] - (-23.257)).abs() < 0.001); // -0.02325700 km/s → -23.257 m/s

        let m2 = &opm.maneuvers[1];
        assert!((m2.duration - 0.0).abs() < 1e-10);
        assert_eq!(m2.ref_frame, CCSDSRefFrame::RTN);
    }

    #[test]
    #[parallel]
    fn test_parse_opm_example4_with_covariance_and_user_defined() {
        let content = std::fs::read_to_string("test_assets/ccsds/opm/OPMExample4.txt").unwrap();
        let opm = parse_opm(&content).unwrap();

        // Covariance
        let cov = opm.covariance.as_ref().unwrap();
        assert_eq!(cov.cov_ref_frame.as_ref().unwrap(), &CCSDSRefFrame::RTN);
        assert!((cov.matrix[(0, 0)] - 3.331349476038534e-04 * 1e6).abs() < 1e-2);

        // User-defined
        let ud = opm.user_defined.as_ref().unwrap();
        assert_eq!(
            ud.parameters.get("OBJ1_TIME_LASTOB_START").unwrap(),
            "2020-01-29T13:30:00"
        );
    }

    #[test]
    #[parallel]
    fn test_parse_opm_example5_with_three_maneuvers() {
        let content = std::fs::read_to_string("test_assets/ccsds/opm/OPMExample5.txt").unwrap();
        let opm = parse_opm(&content).unwrap();

        assert_eq!(opm.metadata.ref_frame, CCSDSRefFrame::GCRF);
        assert_eq!(opm.metadata.time_system, CCSDSTimeSystem::GPS);
        assert_eq!(opm.maneuvers.len(), 3);
    }

    #[test]
    #[serial_test::parallel]
    fn test_parse_opm_attributes_comments_to_the_block_they_introduce() {
        let opm = parse_opm(
            &std::fs::read_to_string("test_assets/ccsds/opm/OPM-section-comments.txt").unwrap(),
        )
        .unwrap();

        assert_eq!(opm.header.comments, vec!["header comment"]);
        assert_eq!(opm.metadata.comments, vec!["metadata comment"]);
        assert_eq!(opm.state_vector.comments, vec!["state vector comment"]);
        assert_eq!(
            opm.keplerian_elements.as_ref().unwrap().comments,
            vec!["keplerian comment"]
        );
        assert_eq!(
            opm.spacecraft_parameters.as_ref().unwrap().comments,
            vec!["spacecraft comment"]
        );
        assert_eq!(opm.maneuvers.len(), 2);
        assert_eq!(opm.maneuvers[0].comments, vec!["first maneuver comment"]);
        assert_eq!(opm.maneuvers[1].comments, vec!["second maneuver comment"]);
    }

    #[test]
    #[serial_test::parallel]
    fn test_opm_covariance_block_omits_epoch() {
        use crate::ccsds::opm::OPM;

        // CCSDS 502.0-B-3 table 3-3 gives the OPM covariance block only
        // COMMENT, COV_REF_FRAME, and the matrix entries. EPOCH belongs to the
        // OEM block alone, and the KVN parser matches it positionally, so a
        // second assignment would land on the state vector.
        let content = std::fs::read_to_string("test_assets/ccsds/opm/OPMExample3.txt").unwrap();
        let written = write_opm(&OPM::from_str(&content).unwrap()).unwrap();

        let epoch_lines = written
            .lines()
            .filter(|line| line.trim_start().starts_with("EPOCH "))
            .count();
        assert_eq!(
            epoch_lines, 1,
            "the OPM state vector's EPOCH is the only one"
        );
    }
}
