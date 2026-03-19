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
        setup_eop();

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
        setup_eop();

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
        setup_eop();

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
        setup_eop();

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
        setup_eop();

        let content =
            std::fs::read_to_string("test_assets/ccsds/oem/OEMExampleWithHeaderComment.txt")
                .unwrap();
        let oem = parse_oem(&content).unwrap();

        assert!(!oem.header.comments.is_empty());
    }

    #[test]
    fn test_parse_oem_iss_truncated() {
        setup_eop();

        let content =
            std::fs::read_to_string("test_assets/ccsds/oem/ISS.resampled.truncated.txt").unwrap();
        let oem = parse_oem(&content).unwrap();

        assert_eq!(oem.segments.len(), 1);
        assert!(!oem.segments[0].states.is_empty());
    }

    #[test]
    fn test_parse_oem_lowercase_value() {
        setup_eop();

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
        setup_eop();

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
        setup_eop();

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
        setup_eop();

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
        setup_eop();

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
        setup_eop();

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
        setup_eop();

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
        setup_eop();

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
        setup_eop();

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
        setup_eop();

        let content = std::fs::read_to_string("test_assets/ccsds/opm/OPMExample5.txt").unwrap();
        let opm = parse_opm(&content).unwrap();

        assert_eq!(opm.metadata.ref_frame, CCSDSRefFrame::GCRF);
        assert_eq!(opm.metadata.time_system, CCSDSTimeSystem::GPS);
        assert_eq!(opm.maneuvers.len(), 3);
    }
}
