/*!
 * KVN tokenizer and parser for CCSDS OEM, OMM, and OPM messages.
 *
 * The KVN parser uses a line-by-line state machine approach. It cannot use
 * serde because KVN is positional/contextual — the meaning of data lines
 * depends on the current parser state (ephemeris block vs. covariance block).
 */

use std::collections::HashMap;

use crate::ccsds::common::{
    CCSDSCovariance, CCSDSRefFrame, CCSDSSpacecraftParameters, CCSDSTimeSystem, CCSDSUserDefined,
    ODMHeader, covariance_from_lower_triangular, parse_ccsds_datetime, strip_units,
};
use crate::ccsds::error::{ccsds_missing_field, ccsds_parse_error};
use crate::ccsds::oem::{OEM, OEMMetadata, OEMSegment, OEMStateVector};
use crate::ccsds::omm::{OMM, OMMMetadata, OMMTleParameters, OMMeanElements};
use crate::ccsds::opm::{OPM, OPMKeplerianElements, OPMManeuver, OPMMetadata, OPMStateVector};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

/// A parsed KVN line token.
#[derive(Debug)]
enum KVNToken {
    /// A key=value pair
    KeyValue { key: String, value: String },
    /// A comment line
    Comment(String),
    /// A data line (space-separated values)
    DataLine(Vec<String>),
    /// An empty or blank line
    Empty,
}

/// Section markers that appear as standalone keywords (no `=`).
const SECTION_MARKERS: &[&str] = &[
    "META_START",
    "META_STOP",
    "COVARIANCE_START",
    "COVARIANCE_STOP",
];

/// Tokenize a single KVN line.
fn tokenize_line(line: &str) -> KVNToken {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return KVNToken::Empty;
    }

    // Check for COMMENT
    if let Some(rest) = trimmed.strip_prefix("COMMENT") {
        let text = rest.trim().to_string();
        return KVNToken::Comment(text);
    }

    // Check for standalone section markers (no '=' sign)
    if SECTION_MARKERS.contains(&trimmed) {
        return KVNToken::KeyValue {
            key: trimmed.to_string(),
            value: String::new(),
        };
    }

    // Check for key=value
    if let Some(eq_pos) = trimmed.find('=') {
        let key = trimmed[..eq_pos].trim().to_string();
        let value = trimmed[eq_pos + 1..].trim().to_string();
        return KVNToken::KeyValue { key, value };
    }

    // Otherwise it's a data line (space-separated tokens)
    let parts: Vec<String> = trimmed.split_whitespace().map(|s| s.to_string()).collect();
    if parts.is_empty() {
        KVNToken::Empty
    } else {
        KVNToken::DataLine(parts)
    }
}

/// Parser state for OEM KVN parsing.
#[derive(Debug, PartialEq)]
enum OEMState {
    Header,
    Metadata,
    EphemerisBlock,
    CovarianceBlock,
}

/// Parse an OEM message from KVN format.
pub fn parse_oem(content: &str) -> Result<OEM, BraheError> {
    let mut state = OEMState::Header;

    // Header fields
    let mut format_version: Option<f64> = None;
    let mut classification: Option<String> = None;
    let mut creation_date: Option<Epoch> = None;
    let mut originator: Option<String> = None;
    let mut message_id: Option<String> = None;
    let mut header_comments: Vec<String> = Vec::new();

    // Current segment
    let mut segments: Vec<OEMSegment> = Vec::new();
    let mut current_metadata: Option<OEMMetadata> = None;
    let mut current_states: Vec<OEMStateVector> = Vec::new();
    let mut current_covariances: Vec<CCSDSCovariance> = Vec::new();
    let mut current_data_comments: Vec<String> = Vec::new();
    let mut current_metadata_comments: Vec<String> = Vec::new();

    // Metadata fields being built
    let mut meta_object_name: Option<String> = None;
    let mut meta_object_id: Option<String> = None;
    let mut meta_center_name: Option<String> = None;
    let mut meta_ref_frame: Option<CCSDSRefFrame> = None;
    let mut meta_ref_frame_epoch: Option<Epoch> = None;
    let mut meta_time_system: Option<CCSDSTimeSystem> = None;
    let mut meta_start_time: Option<Epoch> = None;
    let mut meta_useable_start_time: Option<Epoch> = None;
    let mut meta_useable_stop_time: Option<Epoch> = None;
    let mut meta_stop_time: Option<Epoch> = None;
    let mut meta_interpolation: Option<String> = None;
    let mut meta_interpolation_degree: Option<u32> = None;

    // Covariance state
    let mut cov_epoch: Option<Epoch> = None;
    let mut cov_ref_frame: Option<CCSDSRefFrame> = None;
    let mut cov_values: Vec<f64> = Vec::new();
    let mut cov_comments: Vec<String> = Vec::new();

    // We need the time system from current metadata to parse dates in data blocks
    let mut active_time_system = CCSDSTimeSystem::UTC;

    for line in content.lines() {
        let token = tokenize_line(line);

        match (&state, token) {
            // === HEADER STATE ===
            (OEMState::Header, KVNToken::KeyValue { key, value }) => {
                match key.as_str() {
                    "CCSDS_OEM_VERS" => {
                        format_version = Some(value.parse::<f64>().map_err(|_| {
                            ccsds_parse_error("OEM", &format!("invalid version '{}'", value))
                        })?);
                    }
                    "CLASSIFICATION" => classification = Some(value),
                    "CREATION_DATE" => {
                        creation_date = Some(parse_ccsds_datetime(&value, &CCSDSTimeSystem::UTC)?);
                    }
                    "ORIGINATOR" => originator = Some(value),
                    "MESSAGE_ID" => message_id = Some(value),
                    "META_START" => {
                        // Transition to metadata
                        state = OEMState::Metadata;
                    }
                    _ => {
                        return Err(ccsds_parse_error(
                            "OEM",
                            &format!("unexpected header keyword '{}'", key),
                        ));
                    }
                }
            }
            (OEMState::Header, KVNToken::Comment(text)) => {
                header_comments.push(text);
            }
            (OEMState::Header, KVNToken::Empty) => {}

            // === METADATA STATE ===
            (OEMState::Metadata, KVNToken::KeyValue { key, value }) => {
                match key.as_str() {
                    "META_START" => {
                        // Already in metadata (re-entry handled by state machine)
                    }
                    "OBJECT_NAME" => meta_object_name = Some(value),
                    "OBJECT_ID" => meta_object_id = Some(value),
                    "CENTER_NAME" => meta_center_name = Some(value),
                    "REF_FRAME" => meta_ref_frame = Some(CCSDSRefFrame::parse(&value)),
                    "REF_FRAME_EPOCH" => {
                        let ts = meta_time_system.as_ref().unwrap_or(&CCSDSTimeSystem::UTC);
                        meta_ref_frame_epoch = Some(parse_ccsds_datetime(&value, ts)?);
                    }
                    "TIME_SYSTEM" => {
                        let ts = CCSDSTimeSystem::parse(&value)?;
                        active_time_system = ts.clone();
                        meta_time_system = Some(ts);
                    }
                    "START_TIME" => {
                        let ts = meta_time_system.as_ref().unwrap_or(&CCSDSTimeSystem::UTC);
                        meta_start_time = Some(parse_ccsds_datetime(&value, ts)?);
                    }
                    "USEABLE_START_TIME" => {
                        let ts = meta_time_system.as_ref().unwrap_or(&CCSDSTimeSystem::UTC);
                        meta_useable_start_time = Some(parse_ccsds_datetime(&value, ts)?);
                    }
                    "USEABLE_STOP_TIME" => {
                        let ts = meta_time_system.as_ref().unwrap_or(&CCSDSTimeSystem::UTC);
                        meta_useable_stop_time = Some(parse_ccsds_datetime(&value, ts)?);
                    }
                    "STOP_TIME" => {
                        let ts = meta_time_system.as_ref().unwrap_or(&CCSDSTimeSystem::UTC);
                        meta_stop_time = Some(parse_ccsds_datetime(&value, ts)?);
                    }
                    "INTERPOLATION" => meta_interpolation = Some(value),
                    "INTERPOLATION_DEGREE" => {
                        meta_interpolation_degree = Some(value.parse::<u32>().map_err(|_| {
                            ccsds_parse_error(
                                "OEM",
                                &format!("invalid interpolation degree '{}'", value),
                            )
                        })?);
                    }
                    "META_STOP" => {
                        // Finalize metadata and transition to ephemeris block
                        let metadata = OEMMetadata {
                            object_name: meta_object_name
                                .take()
                                .ok_or_else(|| ccsds_missing_field("OEM", "OBJECT_NAME"))?,
                            object_id: meta_object_id
                                .take()
                                .ok_or_else(|| ccsds_missing_field("OEM", "OBJECT_ID"))?,
                            center_name: meta_center_name
                                .take()
                                .ok_or_else(|| ccsds_missing_field("OEM", "CENTER_NAME"))?,
                            ref_frame: meta_ref_frame
                                .take()
                                .ok_or_else(|| ccsds_missing_field("OEM", "REF_FRAME"))?,
                            ref_frame_epoch: meta_ref_frame_epoch.take(),
                            time_system: meta_time_system
                                .take()
                                .ok_or_else(|| ccsds_missing_field("OEM", "TIME_SYSTEM"))?,
                            start_time: meta_start_time
                                .take()
                                .ok_or_else(|| ccsds_missing_field("OEM", "START_TIME"))?,
                            useable_start_time: meta_useable_start_time.take(),
                            useable_stop_time: meta_useable_stop_time.take(),
                            stop_time: meta_stop_time
                                .take()
                                .ok_or_else(|| ccsds_missing_field("OEM", "STOP_TIME"))?,
                            interpolation: meta_interpolation.take(),
                            interpolation_degree: meta_interpolation_degree.take(),
                            comments: std::mem::take(&mut current_metadata_comments),
                        };

                        // Save the previous segment if there was one
                        if let Some(prev_meta) = current_metadata.take() {
                            segments.push(OEMSegment {
                                metadata: prev_meta,
                                comments: std::mem::take(&mut current_data_comments),
                                states: std::mem::take(&mut current_states),
                                covariances: std::mem::take(&mut current_covariances),
                            });
                        }

                        current_metadata = Some(metadata);
                        state = OEMState::EphemerisBlock;
                    }
                    _ => {
                        return Err(ccsds_parse_error(
                            "OEM",
                            &format!("unexpected metadata keyword '{}'", key),
                        ));
                    }
                }
            }
            (OEMState::Metadata, KVNToken::Comment(text)) => {
                current_metadata_comments.push(text);
            }
            (OEMState::Metadata, KVNToken::Empty) => {}

            // === EPHEMERIS BLOCK STATE ===
            (OEMState::EphemerisBlock, KVNToken::DataLine(parts)) => {
                // Data line: epoch x y z vx vy vz [ax ay az]
                if parts.len() < 7 {
                    return Err(ccsds_parse_error(
                        "OEM",
                        &format!(
                            "ephemeris data line has {} columns, expected at least 7",
                            parts.len()
                        ),
                    ));
                }

                let epoch = parse_ccsds_datetime(&parts[0], &active_time_system)?;

                // Parse position (km → m) and velocity (km/s → m/s)
                let x: f64 = parts[1].parse().map_err(|_| {
                    ccsds_parse_error("OEM", &format!("invalid X value '{}'", parts[1]))
                })?;
                let y: f64 = parts[2].parse().map_err(|_| {
                    ccsds_parse_error("OEM", &format!("invalid Y value '{}'", parts[2]))
                })?;
                let z: f64 = parts[3].parse().map_err(|_| {
                    ccsds_parse_error("OEM", &format!("invalid Z value '{}'", parts[3]))
                })?;
                let vx: f64 = parts[4].parse().map_err(|_| {
                    ccsds_parse_error("OEM", &format!("invalid VX value '{}'", parts[4]))
                })?;
                let vy: f64 = parts[5].parse().map_err(|_| {
                    ccsds_parse_error("OEM", &format!("invalid VY value '{}'", parts[5]))
                })?;
                let vz: f64 = parts[6].parse().map_err(|_| {
                    ccsds_parse_error("OEM", &format!("invalid VZ value '{}'", parts[6]))
                })?;

                // Convert km to m, km/s to m/s
                let position = [x * 1000.0, y * 1000.0, z * 1000.0];
                let velocity = [vx * 1000.0, vy * 1000.0, vz * 1000.0];

                let acceleration = if parts.len() >= 10 {
                    let ax: f64 = parts[7].parse().map_err(|_| {
                        ccsds_parse_error("OEM", &format!("invalid AX value '{}'", parts[7]))
                    })?;
                    let ay: f64 = parts[8].parse().map_err(|_| {
                        ccsds_parse_error("OEM", &format!("invalid AY value '{}'", parts[8]))
                    })?;
                    let az: f64 = parts[9].parse().map_err(|_| {
                        ccsds_parse_error("OEM", &format!("invalid AZ value '{}'", parts[9]))
                    })?;
                    // km/s² → m/s²
                    Some([ax * 1000.0, ay * 1000.0, az * 1000.0])
                } else {
                    None
                };

                current_states.push(OEMStateVector {
                    epoch,
                    position,
                    velocity,
                    acceleration,
                });
            }
            (OEMState::EphemerisBlock, KVNToken::KeyValue { key, value: _ }) => {
                match key.as_str() {
                    "META_START" => {
                        // New segment starting
                        state = OEMState::Metadata;
                    }
                    "COVARIANCE_START" => {
                        cov_epoch = None;
                        cov_ref_frame = None;
                        cov_values.clear();
                        cov_comments.clear();
                        state = OEMState::CovarianceBlock;
                    }
                    _ => {
                        return Err(ccsds_parse_error(
                            "OEM",
                            &format!("unexpected keyword '{}' in ephemeris block", key),
                        ));
                    }
                }
            }
            (OEMState::EphemerisBlock, KVNToken::Comment(text)) => {
                current_data_comments.push(text);
            }
            (OEMState::EphemerisBlock, KVNToken::Empty) => {}

            // === COVARIANCE BLOCK STATE ===
            (OEMState::CovarianceBlock, KVNToken::KeyValue { key, value }) => {
                match key.as_str() {
                    "EPOCH" => {
                        // If we have accumulated values for a previous covariance, save it
                        if cov_values.len() == 21 {
                            let mut vals = [0.0_f64; 21];
                            vals.copy_from_slice(&cov_values);
                            // Convert km² → m² (factor of 1e6 for pos-pos, 1e3 for pos-vel, 1.0 for vel-vel)
                            // Use uniform km² scaling since CCSDS covariance uses km and km/s
                            let matrix = covariance_from_lower_triangular(&vals, 1e6);
                            current_covariances.push(CCSDSCovariance {
                                epoch: cov_epoch.take(),
                                cov_ref_frame: cov_ref_frame.take(),
                                matrix,
                                comments: std::mem::take(&mut cov_comments),
                            });
                            cov_values.clear();
                        }
                        cov_epoch = Some(parse_ccsds_datetime(&value, &active_time_system)?);
                    }
                    "COV_REF_FRAME" => {
                        cov_ref_frame = Some(CCSDSRefFrame::parse(&value));
                    }
                    "COVARIANCE_STOP" => {
                        // Save accumulated covariance
                        if cov_values.len() == 21 {
                            let mut vals = [0.0_f64; 21];
                            vals.copy_from_slice(&cov_values);
                            let matrix = covariance_from_lower_triangular(&vals, 1e6);
                            current_covariances.push(CCSDSCovariance {
                                epoch: cov_epoch.take(),
                                cov_ref_frame: cov_ref_frame.take(),
                                matrix,
                                comments: std::mem::take(&mut cov_comments),
                            });
                            cov_values.clear();
                        }
                        state = OEMState::EphemerisBlock;
                    }
                    _ => {
                        return Err(ccsds_parse_error(
                            "OEM",
                            &format!("unexpected keyword '{}' in covariance block", key),
                        ));
                    }
                }
            }
            (OEMState::CovarianceBlock, KVNToken::DataLine(parts)) => {
                // Covariance data lines: 1 to 6 values per line (lower triangular)
                for part in &parts {
                    let val: f64 = part.parse().map_err(|_| {
                        ccsds_parse_error("OEM", &format!("invalid covariance value '{}'", part))
                    })?;
                    cov_values.push(val);
                }
            }
            (OEMState::CovarianceBlock, KVNToken::Comment(text)) => {
                cov_comments.push(text);
            }
            (OEMState::CovarianceBlock, KVNToken::Empty) => {}

            // Catch unexpected tokens
            (st, token) => {
                return Err(ccsds_parse_error(
                    "OEM",
                    &format!("unexpected token {:?} in state {:?}", token, st),
                ));
            }
        }
    }

    // Save the last segment
    if let Some(meta) = current_metadata.take() {
        segments.push(OEMSegment {
            metadata: meta,
            comments: current_data_comments,
            states: current_states,
            covariances: current_covariances,
        });
    }

    // Build header
    let header = ODMHeader {
        format_version: format_version
            .ok_or_else(|| ccsds_missing_field("OEM", "CCSDS_OEM_VERS"))?,
        classification,
        creation_date: creation_date.ok_or_else(|| ccsds_missing_field("OEM", "CREATION_DATE"))?,
        originator: originator.ok_or_else(|| ccsds_missing_field("OEM", "ORIGINATOR"))?,
        message_id,
        comments: header_comments,
    };

    Ok(OEM { header, segments })
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
    let mut ref_frame_epoch: Option<Epoch> = None;
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

    // Track which section we're in for comments
    let mut in_header = true;
    let mut in_metadata = false;
    let mut in_mean_elements = false;
    let mut in_tle = false;
    let mut in_spacecraft = false;

    let active_ts = |ts: &Option<CCSDSTimeSystem>| ts.clone().unwrap_or(CCSDSTimeSystem::UTC);

    for line in content.lines() {
        let token = tokenize_line(line);
        match token {
            KVNToken::KeyValue { key, value } => {
                let val = strip_units(&value);
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
                        in_header = false;
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
                        in_metadata = true;
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
                    "REF_FRAME_EPOCH" => {
                        ref_frame_epoch =
                            Some(parse_ccsds_datetime(val, &active_ts(&time_system))?);
                    }
                    "TIME_SYSTEM" => {
                        time_system = Some(CCSDSTimeSystem::parse(val)?);
                    }
                    "MEAN_ELEMENT_THEORY" => {
                        mean_element_theory = Some(val.to_string());
                        in_metadata = false;
                    }

                    // Mean elements
                    "EPOCH" => {
                        epoch = Some(parse_ccsds_datetime(val, &active_ts(&time_system))?);
                        in_mean_elements = true;
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
                        in_mean_elements = false;
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
                        in_tle = true;
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
                        in_tle = false;
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
                        in_spacecraft = true;
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
                        in_spacecraft = false;
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
            KVNToken::Comment(text) => {
                if in_header {
                    header_comments.push(text);
                } else if in_metadata {
                    metadata_comments.push(text);
                } else if in_mean_elements {
                    mean_element_comments.push(text);
                } else if in_tle {
                    tle_comments.push(text);
                } else if in_spacecraft {
                    spacecraft_comments.push(text);
                } else if !cov_values.is_empty() {
                    cov_comments.push(text);
                } else {
                    metadata_comments.push(text);
                }
            }
            KVNToken::Empty => {}
            KVNToken::DataLine(_) => {}
        }
    }

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
        ref_frame_epoch,
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

/// Parse an OPM message from KVN format.
pub fn parse_opm(content: &str) -> Result<OPM, BraheError> {
    let mut header_comments: Vec<String> = Vec::new();
    let mut metadata_comments: Vec<String> = Vec::new();
    let mut state_comments: Vec<String> = Vec::new();
    let mut kep_comments: Vec<String> = Vec::new();
    let spacecraft_comments: Vec<String> = Vec::new();
    let mut maneuver_comments: Vec<String> = Vec::new();

    // Header
    let mut format_version: Option<f64> = None;
    let mut creation_date: Option<Epoch> = None;
    let mut originator: Option<String> = None;
    let mut message_id: Option<String> = None;

    // Metadata
    let mut object_name: Option<String> = None;
    let mut object_id: Option<String> = None;
    let mut center_name: Option<String> = None;
    let mut ref_frame: Option<CCSDSRefFrame> = None;
    let mut ref_frame_epoch: Option<Epoch> = None;
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

    let mut in_header = true;

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

    for line in content.lines() {
        let token = tokenize_line(line);
        match token {
            KVNToken::KeyValue { key, value } => {
                let val = strip_units(&value);
                match key.as_str() {
                    "CCSDS_OPM_VERS" => {
                        format_version = Some(
                            val.parse()
                                .map_err(|_| ccsds_parse_error("OPM", "invalid version"))?,
                        );
                    }
                    "CREATION_DATE" => {
                        creation_date = Some(parse_ccsds_datetime(val, &CCSDSTimeSystem::UTC)?);
                        in_header = false;
                    }
                    "ORIGINATOR" => {
                        originator = Some(val.to_string());
                    }
                    "MESSAGE_ID" => {
                        message_id = Some(val.to_string());
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
                    "REF_FRAME" => {
                        if ref_frame.is_none() {
                            ref_frame = Some(CCSDSRefFrame::parse(val));
                        }
                    }
                    "REF_FRAME_EPOCH" => {
                        ref_frame_epoch =
                            Some(parse_ccsds_datetime(val, &active_ts(&time_system))?);
                    }
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
                    "ECCENTRICITY" => {
                        if kep_sma.is_some() || kep_ecc.is_none() {
                            kep_ecc = Some(
                                val.parse()
                                    .map_err(|_| ccsds_parse_error("OPM", "invalid ECC"))?,
                            );
                        }
                    }
                    "INCLINATION" => {
                        if kep_sma.is_some() {
                            kep_inc = Some(
                                val.parse()
                                    .map_err(|_| ccsds_parse_error("OPM", "invalid INC"))?,
                            );
                        }
                    }
                    "RA_OF_ASC_NODE" => {
                        if kep_sma.is_some() {
                            kep_raan = Some(
                                val.parse()
                                    .map_err(|_| ccsds_parse_error("OPM", "invalid RAAN"))?,
                            );
                        }
                    }
                    "ARG_OF_PERICENTER" => {
                        if kep_sma.is_some() {
                            kep_argp = Some(
                                val.parse()
                                    .map_err(|_| ccsds_parse_error("OPM", "invalid ARGP"))?,
                            );
                        }
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
            }
            KVNToken::Comment(text) => {
                if in_header {
                    header_comments.push(text);
                } else if man_epoch.is_some() {
                    maneuver_comments.push(text);
                } else if kep_sma.is_some() {
                    kep_comments.push(text);
                } else if sv_epoch.is_some() && sv_vz.is_none() {
                    state_comments.push(text);
                } else {
                    metadata_comments.push(text);
                }
            }
            KVNToken::Empty | KVNToken::DataLine(_) => {}
        }
    }

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
        classification: None,
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
            comments: Vec::new(),
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
            ref_frame_epoch,
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

/// Parse a CDM message from KVN format.
///
/// Uses a flat key-match approach with object context tracking. The `OBJECT`
/// keyword triggers transitions between Object1 and Object2. Within each
/// object, field names implicitly determine the subsection.
pub fn parse_cdm(content: &str) -> Result<crate::ccsds::cdm::CDM, BraheError> {
    use crate::ccsds::cdm::*;
    use crate::ccsds::common::covariance9x9_from_lower_triangular;

    // Track which object we're currently parsing
    #[derive(PartialEq)]
    enum CurrentObject {
        None,
        Object1,
        Object2,
    }
    let mut current_object = CurrentObject::None;

    // Header fields
    let mut format_version: Option<f64> = None;
    let mut classification: Option<String> = None;
    let mut creation_date: Option<Epoch> = None;
    let mut originator: Option<String> = None;
    let mut message_for: Option<String> = None;
    let mut message_id: Option<String> = None;
    let mut header_comments: Vec<String> = Vec::new();

    // Relative metadata fields
    let mut conjunction_id: Option<String> = None;
    let mut tca: Option<Epoch> = None;
    let mut miss_distance: Option<f64> = None;
    let mut mahalanobis_distance: Option<f64> = None;
    let mut relative_speed: Option<f64> = None;
    let mut rel_pos_r: Option<f64> = None;
    let mut rel_pos_t: Option<f64> = None;
    let mut rel_pos_n: Option<f64> = None;
    let mut rel_vel_r: Option<f64> = None;
    let mut rel_vel_t: Option<f64> = None;
    let mut rel_vel_n: Option<f64> = None;
    let mut approach_angle: Option<f64> = None;
    let mut start_screen_period: Option<Epoch> = None;
    let mut stop_screen_period: Option<Epoch> = None;
    let mut screen_type: Option<String> = None;
    let mut screen_volume_frame: Option<CCSDSRefFrame> = None;
    let mut screen_volume_shape: Option<String> = None;
    let mut screen_volume_radius: Option<f64> = None;
    let mut screen_volume_x: Option<f64> = None;
    let mut screen_volume_y: Option<f64> = None;
    let mut screen_volume_z: Option<f64> = None;
    let mut screen_entry_time: Option<Epoch> = None;
    let mut screen_exit_time: Option<Epoch> = None;
    let mut screen_pc_threshold: Option<f64> = None;
    let mut collision_percentile: Option<Vec<u32>> = None;
    let mut collision_probability: Option<f64> = None;
    let mut collision_probability_method: Option<String> = None;
    let mut collision_max_probability: Option<f64> = None;
    let mut collision_max_pc_method: Option<String> = None;
    let mut sefi_collision_probability: Option<f64> = None;
    let mut sefi_collision_probability_method: Option<String> = None;
    let mut sefi_fragmentation_model: Option<String> = None;
    let mut previous_message_id: Option<String> = None;
    let mut previous_message_epoch: Option<Epoch> = None;
    let mut next_message_epoch: Option<Epoch> = None;
    let mut rel_comments: Vec<String> = Vec::new();

    // Per-object data (index 0 = object1, 1 = object2)
    struct ObjectBuilder {
        // Metadata
        object: Option<String>,
        object_designator: Option<String>,
        catalog_name: Option<String>,
        object_name: Option<String>,
        international_designator: Option<String>,
        object_type: Option<String>,
        ops_status: Option<String>,
        operator_contact_position: Option<String>,
        operator_organization: Option<String>,
        operator_phone: Option<String>,
        operator_email: Option<String>,
        ephemeris_name: Option<String>,
        odm_msg_link: Option<String>,
        adm_msg_link: Option<String>,
        obs_before_next_message: Option<String>,
        covariance_method: Option<String>,
        covariance_source: Option<String>,
        maneuverable: Option<String>,
        orbit_center: Option<String>,
        ref_frame: Option<CCSDSRefFrame>,
        alt_cov_type: Option<String>,
        alt_cov_ref_frame: Option<CCSDSRefFrame>,
        gravity_model: Option<String>,
        atmospheric_model: Option<String>,
        n_body_perturbations: Option<String>,
        solar_rad_pressure: Option<String>,
        earth_tides: Option<String>,
        intrack_thrust: Option<String>,
        metadata_comments: Vec<String>,

        // OD parameters
        time_lastob_start: Option<Epoch>,
        time_lastob_end: Option<Epoch>,
        recommended_od_span: Option<f64>,
        actual_od_span: Option<f64>,
        obs_available: Option<u32>,
        obs_used: Option<u32>,
        tracks_available: Option<u32>,
        tracks_used: Option<u32>,
        residuals_accepted: Option<f64>,
        weighted_rms: Option<f64>,
        od_epoch: Option<Epoch>,
        od_comments: Vec<String>,
        has_od_params: bool,

        // Additional parameters
        area_pc: Option<f64>,
        area_pc_min: Option<f64>,
        area_pc_max: Option<f64>,
        area_drg: Option<f64>,
        area_srp: Option<f64>,
        oeb_parent_frame: Option<String>,
        oeb_parent_frame_epoch: Option<Epoch>,
        oeb_q1: Option<f64>,
        oeb_q2: Option<f64>,
        oeb_q3: Option<f64>,
        oeb_qc: Option<f64>,
        oeb_max: Option<f64>,
        oeb_int: Option<f64>,
        oeb_min: Option<f64>,
        area_along_oeb_max: Option<f64>,
        area_along_oeb_int: Option<f64>,
        area_along_oeb_min: Option<f64>,
        rcs: Option<f64>,
        rcs_min: Option<f64>,
        rcs_max: Option<f64>,
        vm_absolute: Option<f64>,
        vm_apparent_min: Option<f64>,
        vm_apparent: Option<f64>,
        vm_apparent_max: Option<f64>,
        reflectance: Option<f64>,
        mass: Option<f64>,
        hbr: Option<f64>,
        cd_area_over_mass: Option<f64>,
        cr_area_over_mass: Option<f64>,
        thrust_acceleration: Option<f64>,
        sedr: Option<f64>,
        min_dv: Option<[f64; 3]>,
        max_dv: Option<[f64; 3]>,
        lead_time_reqd_before_tca: Option<f64>,
        apoapsis_altitude: Option<f64>,
        periapsis_altitude: Option<f64>,
        inclination: Option<f64>,
        cov_confidence: Option<f64>,
        cov_confidence_method: Option<String>,
        add_comments: Vec<String>,
        has_add_params: bool,

        // State vector
        x: Option<f64>,
        y: Option<f64>,
        z: Option<f64>,
        x_dot: Option<f64>,
        y_dot: Option<f64>,
        z_dot: Option<f64>,
        sv_comments: Vec<String>,

        // RTN covariance (store as lower-triangular values)
        rtn_cov_values: Vec<f64>,
        rtn_cov_comments: Vec<String>,

        // XYZ covariance (store as lower-triangular values)
        xyz_cov_values: Vec<f64>,
        xyz_cov_comments: Vec<String>,

        // CSIG3EIGVEC3
        csig3eigvec3: Option<String>,

        // Additional covariance metadata
        density_forecast_uncertainty: Option<f64>,
        cscale_factor_min: Option<f64>,
        cscale_factor: Option<f64>,
        cscale_factor_max: Option<f64>,
        screening_data_source: Option<String>,
        dcp_sensitivity_vector_position: Option<[f64; 3]>,
        dcp_sensitivity_vector_velocity: Option<[f64; 3]>,
        acm_comments: Vec<String>,
        has_acm: bool,

        data_comments: Vec<String>,
        in_xyz_cov: bool,
    }

    impl ObjectBuilder {
        fn new() -> Self {
            Self {
                object: None,
                object_designator: None,
                catalog_name: None,
                object_name: None,
                international_designator: None,
                object_type: None,
                ops_status: None,
                operator_contact_position: None,
                operator_organization: None,
                operator_phone: None,
                operator_email: None,
                ephemeris_name: None,
                odm_msg_link: None,
                adm_msg_link: None,
                obs_before_next_message: None,
                covariance_method: None,
                covariance_source: None,
                maneuverable: None,
                orbit_center: None,
                ref_frame: None,
                alt_cov_type: None,
                alt_cov_ref_frame: None,
                gravity_model: None,
                atmospheric_model: None,
                n_body_perturbations: None,
                solar_rad_pressure: None,
                earth_tides: None,
                intrack_thrust: None,
                metadata_comments: Vec::new(),

                time_lastob_start: None,
                time_lastob_end: None,
                recommended_od_span: None,
                actual_od_span: None,
                obs_available: None,
                obs_used: None,
                tracks_available: None,
                tracks_used: None,
                residuals_accepted: None,
                weighted_rms: None,
                od_epoch: None,
                od_comments: Vec::new(),
                has_od_params: false,

                area_pc: None,
                area_pc_min: None,
                area_pc_max: None,
                area_drg: None,
                area_srp: None,
                oeb_parent_frame: None,
                oeb_parent_frame_epoch: None,
                oeb_q1: None,
                oeb_q2: None,
                oeb_q3: None,
                oeb_qc: None,
                oeb_max: None,
                oeb_int: None,
                oeb_min: None,
                area_along_oeb_max: None,
                area_along_oeb_int: None,
                area_along_oeb_min: None,
                rcs: None,
                rcs_min: None,
                rcs_max: None,
                vm_absolute: None,
                vm_apparent_min: None,
                vm_apparent: None,
                vm_apparent_max: None,
                reflectance: None,
                mass: None,
                hbr: None,
                cd_area_over_mass: None,
                cr_area_over_mass: None,
                thrust_acceleration: None,
                sedr: None,
                min_dv: None,
                max_dv: None,
                lead_time_reqd_before_tca: None,
                apoapsis_altitude: None,
                periapsis_altitude: None,
                inclination: None,
                cov_confidence: None,
                cov_confidence_method: None,
                add_comments: Vec::new(),
                has_add_params: false,

                x: None,
                y: None,
                z: None,
                x_dot: None,
                y_dot: None,
                z_dot: None,
                sv_comments: Vec::new(),

                rtn_cov_values: Vec::new(),
                rtn_cov_comments: Vec::new(),
                xyz_cov_values: Vec::new(),
                xyz_cov_comments: Vec::new(),
                csig3eigvec3: None,

                density_forecast_uncertainty: None,
                cscale_factor_min: None,
                cscale_factor: None,
                cscale_factor_max: None,
                screening_data_source: None,
                dcp_sensitivity_vector_position: None,
                dcp_sensitivity_vector_velocity: None,
                acm_comments: Vec::new(),
                has_acm: false,

                data_comments: Vec::new(),
                in_xyz_cov: false,
            }
        }
    }

    let mut obj1 = ObjectBuilder::new();
    let mut obj2 = ObjectBuilder::new();
    let mut user_defined: HashMap<String, String> = HashMap::new();

    let utc = CCSDSTimeSystem::UTC;

    // Helper: parse float from value with unit stripping
    let parse_f64 = |val: &str| -> Result<f64, BraheError> {
        strip_units(val)
            .parse()
            .map_err(|_| ccsds_parse_error("CDM", &format!("invalid numeric value '{}'", val)))
    };
    let parse_u32 = |val: &str| -> Result<u32, BraheError> {
        strip_units(val)
            .parse()
            .map_err(|_| ccsds_parse_error("CDM", &format!("invalid integer value '{}'", val)))
    };

    // RTN covariance field name → (row, col) mapping
    fn rtn_cov_index(key: &str) -> Option<(usize, usize)> {
        match key {
            "CR_R" => Some((0, 0)),
            "CT_R" => Some((1, 0)),
            "CT_T" => Some((1, 1)),
            "CN_R" => Some((2, 0)),
            "CN_T" => Some((2, 1)),
            "CN_N" => Some((2, 2)),
            "CRDOT_R" => Some((3, 0)),
            "CRDOT_T" => Some((3, 1)),
            "CRDOT_N" => Some((3, 2)),
            "CRDOT_RDOT" => Some((3, 3)),
            "CTDOT_R" => Some((4, 0)),
            "CTDOT_T" => Some((4, 1)),
            "CTDOT_N" => Some((4, 2)),
            "CTDOT_RDOT" => Some((4, 3)),
            "CTDOT_TDOT" => Some((4, 4)),
            "CNDOT_R" => Some((5, 0)),
            "CNDOT_T" => Some((5, 1)),
            "CNDOT_N" => Some((5, 2)),
            "CNDOT_RDOT" => Some((5, 3)),
            "CNDOT_TDOT" => Some((5, 4)),
            "CNDOT_NDOT" => Some((5, 5)),
            "CDRG_R" => Some((6, 0)),
            "CDRG_T" => Some((6, 1)),
            "CDRG_N" => Some((6, 2)),
            "CDRG_RDOT" => Some((6, 3)),
            "CDRG_TDOT" => Some((6, 4)),
            "CDRG_NDOT" => Some((6, 5)),
            "CDRG_DRG" => Some((6, 6)),
            "CSRP_R" => Some((7, 0)),
            "CSRP_T" => Some((7, 1)),
            "CSRP_N" => Some((7, 2)),
            "CSRP_RDOT" => Some((7, 3)),
            "CSRP_TDOT" => Some((7, 4)),
            "CSRP_NDOT" => Some((7, 5)),
            "CSRP_DRG" => Some((7, 6)),
            "CSRP_SRP" => Some((7, 7)),
            "CTHR_R" => Some((8, 0)),
            "CTHR_T" => Some((8, 1)),
            "CTHR_N" => Some((8, 2)),
            "CTHR_RDOT" => Some((8, 3)),
            "CTHR_TDOT" => Some((8, 4)),
            "CTHR_NDOT" => Some((8, 5)),
            "CTHR_DRG" => Some((8, 6)),
            "CTHR_SRP" => Some((8, 7)),
            "CTHR_THR" => Some((8, 8)),
            _ => None,
        }
    }

    // XYZ covariance field name → (row, col) mapping
    fn xyz_cov_index(key: &str) -> Option<(usize, usize)> {
        match key {
            "CX_X" => Some((0, 0)),
            "CY_X" => Some((1, 0)),
            "CY_Y" => Some((1, 1)),
            "CZ_X" => Some((2, 0)),
            "CZ_Y" => Some((2, 1)),
            "CZ_Z" => Some((2, 2)),
            "CXDOT_X" => Some((3, 0)),
            "CXDOT_Y" => Some((3, 1)),
            "CXDOT_Z" => Some((3, 2)),
            "CXDOT_XDOT" => Some((3, 3)),
            "CYDOT_X" => Some((4, 0)),
            "CYDOT_Y" => Some((4, 1)),
            "CYDOT_Z" => Some((4, 2)),
            "CYDOT_XDOT" => Some((4, 3)),
            "CYDOT_YDOT" => Some((4, 4)),
            "CZDOT_X" => Some((5, 0)),
            "CZDOT_Y" => Some((5, 1)),
            "CZDOT_Z" => Some((5, 2)),
            "CZDOT_XDOT" => Some((5, 3)),
            "CZDOT_YDOT" => Some((5, 4)),
            "CZDOT_ZDOT" => Some((5, 5)),
            "CDRG_X" => Some((6, 0)),
            "CDRG_Y" => Some((6, 1)),
            "CDRG_Z" => Some((6, 2)),
            "CDRG_XDOT" => Some((6, 3)),
            "CDRG_YDOT" => Some((6, 4)),
            "CDRG_ZDOT" => Some((6, 5)),
            // CDRG_DRG is shared between RTN and XYZ contexts
            "CSRP_X" => Some((7, 0)),
            "CSRP_Y" => Some((7, 1)),
            "CSRP_Z" => Some((7, 2)),
            "CSRP_XDOT" => Some((7, 3)),
            "CSRP_YDOT" => Some((7, 4)),
            "CSRP_ZDOT" => Some((7, 5)),
            // CSRP_DRG and CSRP_SRP shared
            "CTHR_X" => Some((8, 0)),
            "CTHR_Y" => Some((8, 1)),
            "CTHR_Z" => Some((8, 2)),
            "CTHR_XDOT" => Some((8, 3)),
            "CTHR_YDOT" => Some((8, 4)),
            "CTHR_ZDOT" => Some((8, 5)),
            // CTHR_DRG, CTHR_SRP, CTHR_THR shared
            _ => None,
        }
    }

    // Parse line-by-line
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Parse COMMENT lines
        if let Some(comment_text) = line.strip_prefix("COMMENT") {
            let comment = comment_text.trim().to_string();
            match current_object {
                CurrentObject::None => {
                    if tca.is_some() {
                        rel_comments.push(comment);
                    } else {
                        header_comments.push(comment);
                    }
                }
                CurrentObject::Object1 => obj1.data_comments.push(comment),
                CurrentObject::Object2 => obj2.data_comments.push(comment),
            }
            continue;
        }

        // Parse key=value
        let eq_pos = match line.find('=') {
            Some(pos) => pos,
            None => continue,
        };
        let key = line[..eq_pos].trim();
        let raw_val = line[eq_pos + 1..].trim();
        let val = strip_units(raw_val);

        // Get mutable reference to current object builder
        let obj = match current_object {
            CurrentObject::Object1 => &mut obj1,
            CurrentObject::Object2 => &mut obj2,
            CurrentObject::None => {
                // Header + relative metadata keys
                match key {
                    "CCSDS_CDM_VERS" => {
                        format_version = Some(parse_f64(val)?);
                    }
                    "CLASSIFICATION" => {
                        classification = Some(val.trim_matches('"').to_string());
                    }
                    "CREATION_DATE" => {
                        creation_date = Some(parse_ccsds_datetime(val, &utc)?);
                    }
                    "ORIGINATOR" => {
                        originator = Some(val.to_string());
                    }
                    "MESSAGE_FOR" => {
                        message_for = Some(val.to_string());
                    }
                    "MESSAGE_ID" => {
                        message_id = Some(val.to_string());
                    }
                    "CONJUNCTION_ID" => {
                        conjunction_id = Some(val.to_string());
                    }
                    "TCA" => {
                        tca = Some(parse_ccsds_datetime(val, &utc)?);
                    }
                    "MISS_DISTANCE" => {
                        miss_distance = Some(parse_f64(val)?);
                    }
                    "MAHALANOBIS_DISTANCE" => {
                        mahalanobis_distance = Some(parse_f64(val)?);
                    }
                    "RELATIVE_SPEED" => {
                        relative_speed = Some(parse_f64(val)?);
                    }
                    "RELATIVE_POSITION_R" => {
                        rel_pos_r = Some(parse_f64(val)?);
                    }
                    "RELATIVE_POSITION_T" => {
                        rel_pos_t = Some(parse_f64(val)?);
                    }
                    "RELATIVE_POSITION_N" => {
                        rel_pos_n = Some(parse_f64(val)?);
                    }
                    "RELATIVE_VELOCITY_R" => {
                        rel_vel_r = Some(parse_f64(val)?);
                    }
                    "RELATIVE_VELOCITY_T" => {
                        rel_vel_t = Some(parse_f64(val)?);
                    }
                    "RELATIVE_VELOCITY_N" => {
                        rel_vel_n = Some(parse_f64(val)?);
                    }
                    "APPROACH_ANGLE" => {
                        approach_angle = Some(parse_f64(val)?);
                    }
                    "START_SCREEN_PERIOD" => {
                        start_screen_period = Some(parse_ccsds_datetime(val, &utc)?);
                    }
                    "STOP_SCREEN_PERIOD" => {
                        stop_screen_period = Some(parse_ccsds_datetime(val, &utc)?);
                    }
                    "SCREEN_TYPE" => {
                        screen_type = Some(val.to_string());
                    }
                    "SCREEN_VOLUME_FRAME" => {
                        screen_volume_frame = Some(CCSDSRefFrame::parse(val));
                    }
                    "SCREEN_VOLUME_SHAPE" => {
                        screen_volume_shape = Some(val.to_string());
                    }
                    "SCREEN_VOLUME_RADIUS" => {
                        screen_volume_radius = Some(parse_f64(val)?);
                    }
                    "SCREEN_VOLUME_X" => {
                        screen_volume_x = Some(parse_f64(val)?);
                    }
                    "SCREEN_VOLUME_Y" => {
                        screen_volume_y = Some(parse_f64(val)?);
                    }
                    "SCREEN_VOLUME_Z" => {
                        screen_volume_z = Some(parse_f64(val)?);
                    }
                    "SCREEN_ENTRY_TIME" => {
                        screen_entry_time = Some(parse_ccsds_datetime(val, &utc)?);
                    }
                    "SCREEN_EXIT_TIME" => {
                        screen_exit_time = Some(parse_ccsds_datetime(val, &utc)?);
                    }
                    "SCREEN_PC_THRESHOLD" => {
                        screen_pc_threshold = Some(parse_f64(val)?);
                    }
                    "COLLISION_PERCENTILE" => {
                        let parts: Result<Vec<u32>, _> =
                            val.split_whitespace().map(|s| s.parse::<u32>()).collect();
                        collision_percentile = Some(parts.map_err(|_| {
                            ccsds_parse_error("CDM", "invalid COLLISION_PERCENTILE")
                        })?);
                    }
                    "COLLISION_PROBABILITY" => {
                        collision_probability = Some(parse_f64(val)?);
                    }
                    "COLLISION_PROBABILITY_METHOD" => {
                        collision_probability_method = Some(val.to_string());
                    }
                    "COLLISION_MAX_PROBABILITY" => {
                        collision_max_probability = Some(parse_f64(val)?);
                    }
                    "COLLISION_MAX_PC_METHOD" => {
                        collision_max_pc_method = Some(val.to_string());
                    }
                    "SEFI_COLLISION_PROBABILITY" => {
                        sefi_collision_probability = Some(parse_f64(val)?);
                    }
                    "SEFI_COLLISION_PROBABILITY_METHOD" => {
                        sefi_collision_probability_method = Some(val.to_string());
                    }
                    "SEFI_FRAGMENTATION_MODEL" => {
                        sefi_fragmentation_model = Some(val.to_string());
                    }
                    "PREVIOUS_MESSAGE_ID" => {
                        previous_message_id = Some(val.to_string());
                    }
                    "PREVIOUS_MESSAGE_EPOCH" => {
                        previous_message_epoch = Some(parse_ccsds_datetime(val, &utc)?);
                    }
                    "NEXT_MESSAGE_EPOCH" => {
                        next_message_epoch = Some(parse_ccsds_datetime(val, &utc)?);
                    }
                    "OBJECT" => match val {
                        "OBJECT1" => {
                            current_object = CurrentObject::Object1;
                            obj1.object = Some("OBJECT1".to_string());
                        }
                        "OBJECT2" => {
                            current_object = CurrentObject::Object2;
                            obj2.object = Some("OBJECT2".to_string());
                        }
                        _ => {
                            return Err(ccsds_parse_error(
                                "CDM",
                                &format!("unexpected OBJECT value '{}'", val),
                            ));
                        }
                    },
                    k if k.starts_with("USER_DEFINED_") => {
                        let ud_key = k.strip_prefix("USER_DEFINED_").unwrap_or(k);
                        user_defined.insert(ud_key.to_string(), val.to_string());
                    }
                    _ => {} // Ignore unknown keys in header/relative metadata
                }
                continue;
            }
        };

        // Object-level keyword dispatch
        match key {
            "OBJECT" => match val {
                "OBJECT1" => {
                    current_object = CurrentObject::Object1;
                    obj1.object = Some("OBJECT1".to_string());
                }
                "OBJECT2" => {
                    current_object = CurrentObject::Object2;
                    obj2.object = Some("OBJECT2".to_string());
                }
                _ => {
                    return Err(ccsds_parse_error(
                        "CDM",
                        &format!("unexpected OBJECT value '{}'", val),
                    ));
                }
            },

            // Metadata fields
            "OBJECT_DESIGNATOR" => {
                obj.object_designator = Some(val.to_string());
            }
            "CATALOG_NAME" => {
                obj.catalog_name = Some(val.to_string());
            }
            "OBJECT_NAME" => {
                obj.object_name = Some(val.to_string());
            }
            "INTERNATIONAL_DESIGNATOR" => {
                obj.international_designator = Some(val.to_string());
            }
            "OBJECT_TYPE" => {
                obj.object_type = Some(val.to_string());
            }
            "OPS_STATUS" => {
                obj.ops_status = Some(val.to_string());
            }
            "OPERATOR_CONTACT_POSITION" => {
                obj.operator_contact_position = Some(val.to_string());
            }
            "OPERATOR_ORGANIZATION" => {
                obj.operator_organization = Some(val.to_string());
            }
            "OPERATOR_PHONE" => {
                obj.operator_phone = Some(val.to_string());
            }
            "OPERATOR_EMAIL" => {
                obj.operator_email = Some(val.to_string());
            }
            "EPHEMERIS_NAME" => {
                obj.ephemeris_name = Some(val.to_string());
            }
            "ODM_MSG_LINK" => {
                obj.odm_msg_link = Some(val.to_string());
            }
            "ADM_MSG_LINK" => {
                obj.adm_msg_link = Some(val.to_string());
            }
            "OBS_BEFORE_NEXT_MESSAGE" => {
                obj.obs_before_next_message = Some(val.to_string());
            }
            "COVARIANCE_METHOD" => {
                obj.covariance_method = Some(val.to_string());
            }
            "COVARIANCE_SOURCE" => {
                obj.covariance_source = Some(val.to_string());
            }
            "MANEUVERABLE" => {
                obj.maneuverable = Some(val.to_string());
            }
            "ORBIT_CENTER" => {
                obj.orbit_center = Some(val.to_string());
            }
            "REF_FRAME" => {
                obj.ref_frame = Some(CCSDSRefFrame::parse(val));
            }
            "ALT_COV_TYPE" => {
                obj.alt_cov_type = Some(val.to_string());
            }
            "ALT_COV_REF_FRAME" => {
                obj.alt_cov_ref_frame = Some(CCSDSRefFrame::parse(val));
            }
            "GRAVITY_MODEL" => {
                obj.gravity_model = Some(val.to_string());
            }
            "ATMOSPHERIC_MODEL" => {
                obj.atmospheric_model = Some(val.to_string());
            }
            "N_BODY_PERTURBATIONS" => {
                obj.n_body_perturbations = Some(val.to_string());
            }
            "SOLAR_RAD_PRESSURE" => {
                obj.solar_rad_pressure = Some(val.to_string());
            }
            "EARTH_TIDES" => {
                obj.earth_tides = Some(val.to_string());
            }
            "INTRACK_THRUST" => {
                obj.intrack_thrust = Some(val.to_string());
            }

            // OD parameters
            "TIME_LASTOB_START" => {
                obj.time_lastob_start = Some(parse_ccsds_datetime(val, &utc)?);
                obj.has_od_params = true;
            }
            "TIME_LASTOB_END" => {
                obj.time_lastob_end = Some(parse_ccsds_datetime(val, &utc)?);
                obj.has_od_params = true;
            }
            "RECOMMENDED_OD_SPAN" => {
                obj.recommended_od_span = Some(parse_f64(val)?);
                obj.has_od_params = true;
            }
            "ACTUAL_OD_SPAN" => {
                obj.actual_od_span = Some(parse_f64(val)?);
                obj.has_od_params = true;
            }
            "OBS_AVAILABLE" => {
                obj.obs_available = Some(parse_u32(val)?);
                obj.has_od_params = true;
            }
            "OBS_USED" => {
                obj.obs_used = Some(parse_u32(val)?);
                obj.has_od_params = true;
            }
            "TRACKS_AVAILABLE" => {
                obj.tracks_available = Some(parse_u32(val)?);
                obj.has_od_params = true;
            }
            "TRACKS_USED" => {
                obj.tracks_used = Some(parse_u32(val)?);
                obj.has_od_params = true;
            }
            "RESIDUALS_ACCEPTED" => {
                obj.residuals_accepted = Some(parse_f64(val)?);
                obj.has_od_params = true;
            }
            "WEIGHTED_RMS" => {
                obj.weighted_rms = Some(parse_f64(val)?);
                obj.has_od_params = true;
            }
            "OD_EPOCH" => {
                obj.od_epoch = Some(parse_ccsds_datetime(val, &utc)?);
                obj.has_od_params = true;
            }

            // Additional parameters
            "AREA_PC" => {
                obj.area_pc = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "AREA_PC_MIN" => {
                obj.area_pc_min = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "AREA_PC_MAX" => {
                obj.area_pc_max = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "AREA_DRG" => {
                obj.area_drg = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "AREA_SRP" => {
                obj.area_srp = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "OEB_PARENT_FRAME" => {
                obj.oeb_parent_frame = Some(val.to_string());
                obj.has_add_params = true;
            }
            "OEB_PARENT_FRAME_EPOCH" => {
                obj.oeb_parent_frame_epoch = Some(parse_ccsds_datetime(val, &utc)?);
                obj.has_add_params = true;
            }
            "OEB_Q1" => {
                obj.oeb_q1 = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "OEB_Q2" => {
                obj.oeb_q2 = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "OEB_Q3" => {
                obj.oeb_q3 = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "OEB_QC" => {
                obj.oeb_qc = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "OEB_MAX" => {
                obj.oeb_max = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "OEB_INT" => {
                obj.oeb_int = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "OEB_MIN" => {
                obj.oeb_min = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "AREA_ALONG_OEB_MAX" => {
                obj.area_along_oeb_max = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "AREA_ALONG_OEB_INT" => {
                obj.area_along_oeb_int = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "AREA_ALONG_OEB_MIN" => {
                obj.area_along_oeb_min = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "RCS" => {
                obj.rcs = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "RCS_MIN" => {
                obj.rcs_min = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "RCS_MAX" => {
                obj.rcs_max = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "VM_ABSOLUTE" => {
                obj.vm_absolute = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "VM_APPARENT_MIN" => {
                obj.vm_apparent_min = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "VM_APPARENT" => {
                obj.vm_apparent = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "VM_APPARENT_MAX" => {
                obj.vm_apparent_max = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "REFLECTANCE" => {
                obj.reflectance = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "MASS" => {
                obj.mass = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "HBR" => {
                obj.hbr = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "CD_AREA_OVER_MASS" => {
                obj.cd_area_over_mass = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "CR_AREA_OVER_MASS" => {
                obj.cr_area_over_mass = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "THRUST_ACCELERATION" => {
                obj.thrust_acceleration = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "SEDR" => {
                obj.sedr = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "MIN_DV" => {
                let parts: Vec<f64> = val
                    .split_whitespace()
                    .map(|s| s.parse().unwrap_or(0.0))
                    .collect();
                if parts.len() == 3 {
                    obj.min_dv = Some([parts[0], parts[1], parts[2]]);
                }
                obj.has_add_params = true;
            }
            "MAX_DV" => {
                let parts: Vec<f64> = val
                    .split_whitespace()
                    .map(|s| s.parse().unwrap_or(0.0))
                    .collect();
                if parts.len() == 3 {
                    obj.max_dv = Some([parts[0], parts[1], parts[2]]);
                }
                obj.has_add_params = true;
            }
            "LEAD_TIME_REQD_BEFORE_TCA" => {
                obj.lead_time_reqd_before_tca = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "APOAPSIS_ALTITUDE" => {
                obj.apoapsis_altitude = Some(parse_f64(val)? * 1e3);
                obj.has_add_params = true;
            } // km → m
            "PERIAPSIS_ALTITUDE" => {
                obj.periapsis_altitude = Some(parse_f64(val)? * 1e3);
                obj.has_add_params = true;
            } // km → m
            "INCLINATION" => {
                obj.inclination = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "COV_CONFIDENCE" => {
                obj.cov_confidence = Some(parse_f64(val)?);
                obj.has_add_params = true;
            }
            "COV_CONFIDENCE_METHOD" => {
                obj.cov_confidence_method = Some(val.to_string());
                obj.has_add_params = true;
            }

            // State vector (km → m, km/s → m/s)
            "X" => {
                obj.x = Some(parse_f64(val)? * 1e3);
            }
            "Y" => {
                obj.y = Some(parse_f64(val)? * 1e3);
            }
            "Z" => {
                obj.z = Some(parse_f64(val)? * 1e3);
            }
            "X_DOT" => {
                obj.x_dot = Some(parse_f64(val)? * 1e3);
            }
            "Y_DOT" => {
                obj.y_dot = Some(parse_f64(val)? * 1e3);
            }
            "Z_DOT" => {
                obj.z_dot = Some(parse_f64(val)? * 1e3);
            }

            // Additional covariance metadata
            "DENSITY_FORECAST_UNCERTAINTY" => {
                obj.density_forecast_uncertainty = Some(parse_f64(val)?);
                obj.has_acm = true;
            }
            "CSCALE_FACTOR_MIN" => {
                obj.cscale_factor_min = Some(parse_f64(val)?);
                obj.has_acm = true;
            }
            "CSCALE_FACTOR" => {
                obj.cscale_factor = Some(parse_f64(val)?);
                obj.has_acm = true;
            }
            "CSCALE_FACTOR_MAX" => {
                obj.cscale_factor_max = Some(parse_f64(val)?);
                obj.has_acm = true;
            }
            "SCREENING_DATA_SOURCE" => {
                obj.screening_data_source = Some(val.to_string());
                obj.has_acm = true;
            }
            "DCP_SENSITIVITY_VECTOR_POSITION" => {
                let parts: Vec<f64> = val
                    .split_whitespace()
                    .map(|s| s.parse().unwrap_or(0.0))
                    .collect();
                if parts.len() == 3 {
                    obj.dcp_sensitivity_vector_position = Some([parts[0], parts[1], parts[2]]);
                }
                obj.has_acm = true;
            }
            "DCP_SENSITIVITY_VECTOR_VELOCITY" => {
                let parts: Vec<f64> = val
                    .split_whitespace()
                    .map(|s| s.parse().unwrap_or(0.0))
                    .collect();
                if parts.len() == 3 {
                    obj.dcp_sensitivity_vector_velocity = Some([parts[0], parts[1], parts[2]]);
                }
                obj.has_acm = true;
            }

            // CSIG3EIGVEC3 (stored as raw string)
            "CSIG3EIGVEC3" => {
                obj.csig3eigvec3 = Some(val.to_string());
            }

            // User-defined
            k if k.starts_with("USER_DEFINED_") => {
                let ud_key = k.strip_prefix("USER_DEFINED_").unwrap_or(k);
                user_defined.insert(ud_key.to_string(), val.to_string());
            }

            // Covariance fields — route to RTN or XYZ based on context
            k => {
                // Check if this is an XYZ-specific key (CX_X, CY_X, etc.)
                if let Some((_row, _col)) = xyz_cov_index(k) {
                    let v = parse_f64(val)?;
                    obj.xyz_cov_values.push(v);
                    obj.in_xyz_cov = true; // Switch to XYZ context
                } else if let Some((_row, _col)) = rtn_cov_index(k) {
                    let v = parse_f64(val)?;
                    // If this is a core RTN-only key (CR_R, CT_R, etc.), reset XYZ context
                    if k == "CR_R" {
                        obj.in_xyz_cov = false;
                    }
                    // Shared keys (CDRG_DRG, CSRP_DRG, CSRP_SRP, CTHR_DRG, CTHR_SRP, CTHR_THR)
                    // are routed based on which covariance context we're currently in
                    if obj.in_xyz_cov {
                        obj.xyz_cov_values.push(v);
                    } else {
                        obj.rtn_cov_values.push(v);
                    }
                }
                // Unknown keys are silently ignored
            }
        }
    }

    // Build the CDM struct from collected fields
    let build_object = |obj: ObjectBuilder, label: &str| -> Result<CDMObject, BraheError> {
        let obj_label = obj.object.clone().unwrap_or_else(|| label.to_string());

        let state_vector = CDMStateVector {
            position: [
                obj.x
                    .ok_or_else(|| ccsds_missing_field("CDM", &format!("{} X", obj_label)))?,
                obj.y
                    .ok_or_else(|| ccsds_missing_field("CDM", &format!("{} Y", obj_label)))?,
                obj.z
                    .ok_or_else(|| ccsds_missing_field("CDM", &format!("{} Z", obj_label)))?,
            ],
            velocity: [
                obj.x_dot
                    .ok_or_else(|| ccsds_missing_field("CDM", &format!("{} X_DOT", obj_label)))?,
                obj.y_dot
                    .ok_or_else(|| ccsds_missing_field("CDM", &format!("{} Y_DOT", obj_label)))?,
                obj.z_dot
                    .ok_or_else(|| ccsds_missing_field("CDM", &format!("{} Z_DOT", obj_label)))?,
            ],
            comments: obj.sv_comments,
        };

        // Build RTN covariance
        let rtn_covariance = if obj.rtn_cov_values.is_empty() {
            return Err(ccsds_missing_field(
                "CDM",
                &format!("{} RTN covariance", obj_label),
            ));
        } else {
            let (matrix, dim) = covariance9x9_from_lower_triangular(&obj.rtn_cov_values)?;
            CDMRTNCovariance {
                matrix,
                dimension: dim,
                comments: obj.rtn_cov_comments,
            }
        };

        // Build XYZ covariance (optional)
        let xyz_covariance = if obj.xyz_cov_values.is_empty() {
            None
        } else {
            let (matrix, dim) = covariance9x9_from_lower_triangular(&obj.xyz_cov_values)?;
            Some(CDMXYZCovariance {
                matrix,
                dimension: dim,
                comments: obj.xyz_cov_comments,
            })
        };

        let od_parameters = if obj.has_od_params {
            Some(CDMODParameters {
                time_lastob_start: obj.time_lastob_start,
                time_lastob_end: obj.time_lastob_end,
                recommended_od_span: obj.recommended_od_span,
                actual_od_span: obj.actual_od_span,
                obs_available: obj.obs_available,
                obs_used: obj.obs_used,
                tracks_available: obj.tracks_available,
                tracks_used: obj.tracks_used,
                residuals_accepted: obj.residuals_accepted,
                weighted_rms: obj.weighted_rms,
                od_epoch: obj.od_epoch,
                comments: obj.od_comments,
            })
        } else {
            None
        };

        let additional_parameters = if obj.has_add_params {
            Some(CDMAdditionalParameters {
                area_pc: obj.area_pc,
                area_pc_min: obj.area_pc_min,
                area_pc_max: obj.area_pc_max,
                area_drg: obj.area_drg,
                area_srp: obj.area_srp,
                oeb_parent_frame: obj.oeb_parent_frame,
                oeb_parent_frame_epoch: obj.oeb_parent_frame_epoch,
                oeb_q1: obj.oeb_q1,
                oeb_q2: obj.oeb_q2,
                oeb_q3: obj.oeb_q3,
                oeb_qc: obj.oeb_qc,
                oeb_max: obj.oeb_max,
                oeb_int: obj.oeb_int,
                oeb_min: obj.oeb_min,
                area_along_oeb_max: obj.area_along_oeb_max,
                area_along_oeb_int: obj.area_along_oeb_int,
                area_along_oeb_min: obj.area_along_oeb_min,
                rcs: obj.rcs,
                rcs_min: obj.rcs_min,
                rcs_max: obj.rcs_max,
                vm_absolute: obj.vm_absolute,
                vm_apparent_min: obj.vm_apparent_min,
                vm_apparent: obj.vm_apparent,
                vm_apparent_max: obj.vm_apparent_max,
                reflectance: obj.reflectance,
                mass: obj.mass,
                hbr: obj.hbr,
                cd_area_over_mass: obj.cd_area_over_mass,
                cr_area_over_mass: obj.cr_area_over_mass,
                thrust_acceleration: obj.thrust_acceleration,
                sedr: obj.sedr,
                min_dv: obj.min_dv,
                max_dv: obj.max_dv,
                lead_time_reqd_before_tca: obj.lead_time_reqd_before_tca,
                apoapsis_altitude: obj.apoapsis_altitude,
                periapsis_altitude: obj.periapsis_altitude,
                inclination: obj.inclination,
                cov_confidence: obj.cov_confidence,
                cov_confidence_method: obj.cov_confidence_method,
                comments: obj.add_comments,
            })
        } else {
            None
        };

        let additional_covariance_metadata = if obj.has_acm {
            Some(CDMAdditionalCovarianceMetadata {
                density_forecast_uncertainty: obj.density_forecast_uncertainty,
                cscale_factor_min: obj.cscale_factor_min,
                cscale_factor: obj.cscale_factor,
                cscale_factor_max: obj.cscale_factor_max,
                screening_data_source: obj.screening_data_source,
                dcp_sensitivity_vector_position: obj.dcp_sensitivity_vector_position,
                dcp_sensitivity_vector_velocity: obj.dcp_sensitivity_vector_velocity,
                comments: obj.acm_comments,
            })
        } else {
            None
        };

        let metadata = CDMObjectMetadata {
            object: obj.object.unwrap_or_else(|| label.to_string()),
            object_designator: obj.object_designator.ok_or_else(|| {
                ccsds_missing_field("CDM", &format!("{} OBJECT_DESIGNATOR", obj_label))
            })?,
            catalog_name: obj.catalog_name.ok_or_else(|| {
                ccsds_missing_field("CDM", &format!("{} CATALOG_NAME", obj_label))
            })?,
            object_name: obj
                .object_name
                .ok_or_else(|| ccsds_missing_field("CDM", &format!("{} OBJECT_NAME", obj_label)))?,
            international_designator: obj.international_designator.ok_or_else(|| {
                ccsds_missing_field("CDM", &format!("{} INTERNATIONAL_DESIGNATOR", obj_label))
            })?,
            object_type: obj.object_type,
            ops_status: obj.ops_status,
            operator_contact_position: obj.operator_contact_position,
            operator_organization: obj.operator_organization,
            operator_phone: obj.operator_phone,
            operator_email: obj.operator_email,
            ephemeris_name: obj.ephemeris_name.ok_or_else(|| {
                ccsds_missing_field("CDM", &format!("{} EPHEMERIS_NAME", obj_label))
            })?,
            odm_msg_link: obj.odm_msg_link,
            adm_msg_link: obj.adm_msg_link,
            obs_before_next_message: obj.obs_before_next_message,
            covariance_method: obj.covariance_method.ok_or_else(|| {
                ccsds_missing_field("CDM", &format!("{} COVARIANCE_METHOD", obj_label))
            })?,
            covariance_source: obj.covariance_source,
            maneuverable: obj.maneuverable.ok_or_else(|| {
                ccsds_missing_field("CDM", &format!("{} MANEUVERABLE", obj_label))
            })?,
            orbit_center: obj.orbit_center,
            ref_frame: obj
                .ref_frame
                .ok_or_else(|| ccsds_missing_field("CDM", &format!("{} REF_FRAME", obj_label)))?,
            alt_cov_type: obj.alt_cov_type,
            alt_cov_ref_frame: obj.alt_cov_ref_frame,
            gravity_model: obj.gravity_model,
            atmospheric_model: obj.atmospheric_model,
            n_body_perturbations: obj.n_body_perturbations,
            solar_rad_pressure: obj.solar_rad_pressure,
            earth_tides: obj.earth_tides,
            intrack_thrust: obj.intrack_thrust,
            comments: obj.metadata_comments,
        };

        Ok(CDMObject {
            metadata,
            data: CDMObjectData {
                od_parameters,
                additional_parameters,
                state_vector,
                rtn_covariance,
                xyz_covariance,
                additional_covariance_metadata,
                csig3eigvec3: obj.csig3eigvec3,
                comments: obj.data_comments,
            },
        })
    };

    // Validate mandatory header/relative metadata fields
    let format_version =
        format_version.ok_or_else(|| ccsds_missing_field("CDM", "CCSDS_CDM_VERS"))?;
    let creation_date = creation_date.ok_or_else(|| ccsds_missing_field("CDM", "CREATION_DATE"))?;
    let originator = originator.ok_or_else(|| ccsds_missing_field("CDM", "ORIGINATOR"))?;
    let message_id = message_id.unwrap_or_default();
    let tca = tca.ok_or_else(|| ccsds_missing_field("CDM", "TCA"))?;
    let miss_distance = miss_distance.ok_or_else(|| ccsds_missing_field("CDM", "MISS_DISTANCE"))?;

    let object1 = build_object(obj1, "OBJECT1")?;
    let object2 = build_object(obj2, "OBJECT2")?;

    let user_defined = if user_defined.is_empty() {
        None
    } else {
        Some(CCSDSUserDefined {
            parameters: user_defined,
        })
    };

    Ok(CDM {
        header: CDMHeader {
            format_version,
            classification,
            creation_date,
            originator,
            message_for,
            message_id,
            comments: header_comments,
        },
        relative_metadata: CDMRelativeMetadata {
            conjunction_id,
            tca,
            miss_distance,
            mahalanobis_distance,
            relative_speed,
            relative_position_r: rel_pos_r,
            relative_position_t: rel_pos_t,
            relative_position_n: rel_pos_n,
            relative_velocity_r: rel_vel_r,
            relative_velocity_t: rel_vel_t,
            relative_velocity_n: rel_vel_n,
            approach_angle,
            start_screen_period,
            stop_screen_period,
            screen_type,
            screen_volume_frame,
            screen_volume_shape,
            screen_volume_radius,
            screen_volume_x,
            screen_volume_y,
            screen_volume_z,
            screen_entry_time,
            screen_exit_time,
            screen_pc_threshold,
            collision_percentile,
            collision_probability,
            collision_probability_method,
            collision_max_probability,
            collision_max_pc_method,
            sefi_collision_probability,
            sefi_collision_probability_method,
            sefi_fragmentation_model,
            previous_message_id,
            previous_message_epoch,
            next_message_epoch,
            comments: rel_comments,
        },
        object1,
        object2,
        user_defined,
    })
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_empty() {
        match tokenize_line("") {
            KVNToken::Empty => {}
            _ => panic!("Expected Empty"),
        }
    }

    #[test]
    fn test_tokenize_comment() {
        match tokenize_line("COMMENT This is a test") {
            KVNToken::Comment(text) => assert_eq!(text, "This is a test"),
            _ => panic!("Expected Comment"),
        }
    }

    #[test]
    fn test_tokenize_key_value() {
        match tokenize_line("OBJECT_NAME = ISS") {
            KVNToken::KeyValue { key, value } => {
                assert_eq!(key, "OBJECT_NAME");
                assert_eq!(value, "ISS");
            }
            _ => panic!("Expected KeyValue"),
        }
    }

    #[test]
    fn test_tokenize_data_line() {
        match tokenize_line(
            "2017-04-11T22:31:43.121856 2906.275 4076.358 4561.364 -6.879 1.450 3.081",
        ) {
            KVNToken::DataLine(parts) => {
                assert_eq!(parts.len(), 7);
                assert_eq!(parts[0], "2017-04-11T22:31:43.121856");
            }
            _ => panic!("Expected DataLine"),
        }
    }

    #[test]
    fn test_parse_oem_example1() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let oem = parse_oem(&content).unwrap();

        // Header
        assert!((oem.header.format_version - 3.0).abs() < 1e-10);
        assert_eq!(
            oem.header.classification.as_deref(),
            Some("public, test-data")
        );
        assert_eq!(oem.header.originator, "NASA/JPL");

        // 3 segments
        assert_eq!(oem.segments.len(), 3);

        // Segment 0 metadata
        let seg0 = &oem.segments[0];
        assert_eq!(seg0.metadata.object_name, "MARS GLOBAL SURVEYOR");
        assert_eq!(seg0.metadata.object_id, "1996-062A");
        assert_eq!(seg0.metadata.center_name, "MARS BARYCENTER");
        assert_eq!(seg0.metadata.ref_frame, CCSDSRefFrame::J2000);
        assert_eq!(seg0.metadata.time_system, CCSDSTimeSystem::UTC);
        assert_eq!(seg0.metadata.interpolation.as_deref(), Some("HERMITE"));
        assert_eq!(seg0.metadata.interpolation_degree, Some(7));

        // Segment 0 states
        assert_eq!(seg0.states.len(), 4);
        // First state: position in km converted to meters
        assert!((seg0.states[0].position[0] - 2789.619 * 1000.0).abs() < 1e-3);
        assert!((seg0.states[0].position[1] - (-280.045) * 1000.0).abs() < 1e-3);
        assert!((seg0.states[0].position[2] - (-1746.755) * 1000.0).abs() < 1e-3);
        assert!((seg0.states[0].velocity[0] - 4.73372 * 1000.0).abs() < 1e-3);
        assert!((seg0.states[0].velocity[1] - (-2.49586) * 1000.0).abs() < 1e-3);
        assert!((seg0.states[0].velocity[2] - (-1.04195) * 1000.0).abs() < 1e-3);

        // Segment 0 has no covariance
        assert!(seg0.covariances.is_empty());

        // Segment 1 has data + covariance
        let seg1 = &oem.segments[1];
        assert_eq!(seg1.states.len(), 4);
        assert_eq!(seg1.covariances.len(), 1);

        // Validate covariance matrix (values are in km² in file, stored as m² in struct)
        let cov = &seg1.covariances[0];
        assert_eq!(cov.cov_ref_frame.as_ref().unwrap(), &CCSDSRefFrame::EME2000);
        // CX_X = 3.3313494e-04 km² = 3.3313494e-04 * 1e6 m² = 333.13494 m²
        assert!((cov.matrix[(0, 0)] - 3.3313494e-04 * 1e6).abs() < 1e-4);
        // CY_X = 4.6189273e-04 km²
        assert!((cov.matrix[(1, 0)] - 4.6189273e-04 * 1e6).abs() < 1e-4);
        // Symmetry
        assert_eq!(cov.matrix[(0, 1)], cov.matrix[(1, 0)]);

        // Segment 2 has 2 covariance blocks with different frames
        let seg2 = &oem.segments[2];
        assert_eq!(seg2.covariances.len(), 2);
        assert_eq!(
            seg2.covariances[0].cov_ref_frame.as_ref().unwrap(),
            &CCSDSRefFrame::RTN
        );
        assert_eq!(
            seg2.covariances[1].cov_ref_frame.as_ref().unwrap(),
            &CCSDSRefFrame::EME2000
        );

        // Comments
        assert_eq!(seg0.comments.len(), 2);
        assert!(seg0.comments[0].contains("M.R. Somebody"));
    }

    #[test]
    fn test_parse_oem_example2_doy_format() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample2.txt").unwrap();
        let oem = parse_oem(&content).unwrap();

        assert!((oem.header.format_version - 2.0).abs() < 1e-10);
        assert_eq!(oem.segments.len(), 2);

        // Check DOY dates parsed correctly
        let seg0 = &oem.segments[0];
        assert_eq!(seg0.metadata.ref_frame, CCSDSRefFrame::TOD);
        assert_eq!(seg0.metadata.time_system, CCSDSTimeSystem::MRT);
        assert_eq!(seg0.states.len(), 4);

        // Segment has ref_frame_epoch
        assert!(seg0.metadata.ref_frame_epoch.is_some());

        // Check header comment
        assert_eq!(oem.header.comments.len(), 1);
        assert_eq!(oem.header.comments[0], "comment");

        // Check metadata comments
        assert_eq!(seg0.metadata.comments.len(), 2);
        assert_eq!(seg0.metadata.comments[0], "comment 1");
        assert_eq!(seg0.metadata.comments[1], "comment 2");
    }

    #[test]
    fn test_parse_oem_example4() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample4.txt").unwrap();
        let oem = parse_oem(&content).unwrap();

        assert!((oem.header.format_version - 2.0).abs() < 1e-10);
        assert_eq!(oem.segments.len(), 1);
        assert_eq!(oem.segments[0].metadata.object_name, "MARS GLOBAL SURVEYOR");
        assert_eq!(oem.segments[0].metadata.center_name, "MARS");
        assert_eq!(oem.segments[0].metadata.ref_frame, CCSDSRefFrame::EME2000);
        assert_eq!(oem.segments[0].states.len(), 3);
    }

    #[test]
    fn test_parse_oem_example5_gcrf() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample5.txt").unwrap();
        let oem = parse_oem(&content).unwrap();

        assert_eq!(oem.segments.len(), 1);
        assert_eq!(oem.segments[0].metadata.ref_frame, CCSDSRefFrame::GCRF);
        assert_eq!(oem.segments[0].metadata.object_name, "ISS");
        assert_eq!(oem.segments[0].metadata.object_id, "1998-067A");
        assert_eq!(oem.segments[0].states.len(), 49);
    }

    #[test]
    fn test_parse_oem_with_header_comment() {
        let content =
            std::fs::read_to_string("test_assets/ccsds/oem/OEMExampleWithHeaderComment.txt")
                .unwrap();
        let oem = parse_oem(&content).unwrap();

        assert!(!oem.header.comments.is_empty());
    }

    #[test]
    fn test_parse_oem_iss_truncated() {
        let content =
            std::fs::read_to_string("test_assets/ccsds/oem/ISS.resampled.truncated.txt").unwrap();
        let oem = parse_oem(&content).unwrap();

        assert_eq!(oem.segments.len(), 1);
        assert!(!oem.segments[0].states.is_empty());
    }

    #[test]
    fn test_parse_oem_lowercase_value() {
        let content =
            std::fs::read_to_string("test_assets/ccsds/oem/oemLowerCaseValue.oem").unwrap();
        // This should either parse successfully or give a meaningful error
        let result = parse_oem(&content);
        // For now just verify it doesn't panic
        let _ = result;
    }

    // OMM Tests

    #[test]
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
    fn test_parse_omm_example3_with_spacecraft_and_user_defined() {
        let content = std::fs::read_to_string("test_assets/ccsds/omm/OMMExample3.txt").unwrap();
        let omm = parse_omm(&content).unwrap();

        assert_eq!(omm.metadata.ref_frame, CCSDSRefFrame::TOD);
        assert_eq!(omm.metadata.time_system, CCSDSTimeSystem::MRT);

        // Spacecraft parameters
        let sc = omm.spacecraft_parameters.as_ref().unwrap();
        assert!((sc.mass.unwrap() - 300.0).abs() < 1e-3);
        assert!((sc.solar_rad_area.unwrap() - 5.0).abs() < 1e-3);

        // Covariance with TNW frame
        let cov = omm.covariance.as_ref().unwrap();
        assert_eq!(cov.cov_ref_frame.as_ref().unwrap(), &CCSDSRefFrame::TNW);

        // User-defined
        let ud = omm.user_defined.as_ref().unwrap();
        assert_eq!(ud.parameters.get("EARTH_MODEL").unwrap(), "WGS-84");
    }

    #[test]
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
    fn test_parse_omm_example5_sgp4xp() {
        let content = std::fs::read_to_string("test_assets/ccsds/omm/OMMExample5.txt").unwrap();
        let omm = parse_omm(&content).unwrap();

        assert_eq!(omm.metadata.mean_element_theory, "SGP4-XP");
        let tle = omm.tle_parameters.as_ref().unwrap();
        assert_eq!(tle.ephemeris_type, Some(4));
        assert!((tle.bterm.unwrap() - 0.0015).abs() < 1e-10);
        assert!((tle.agom.unwrap() - 0.001).abs() < 1e-10);
    }

    // OPM Tests

    #[test]
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
    fn test_parse_opm_example5_with_three_maneuvers() {
        let content = std::fs::read_to_string("test_assets/ccsds/opm/OPMExample5.txt").unwrap();
        let opm = parse_opm(&content).unwrap();

        assert_eq!(opm.metadata.ref_frame, CCSDSRefFrame::GCRF);
        assert_eq!(opm.metadata.time_system, CCSDSTimeSystem::GPS);
        assert_eq!(opm.maneuvers.len(), 3);
    }
}
