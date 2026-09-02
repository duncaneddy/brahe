/*!
 * KVN reader and writer for the Orbit Ephemeris Message (OEM).
 *
 * Reference: CCSDS 502.0-B-3 (Orbit Data Messages), section 5
 */

use crate::ccsds::common::{
    CCSDSCovariance, CCSDSRefFrame, CCSDSTimeSystem, ODMHeader, covariance_from_lower_triangular,
    covariance_to_lower_triangular, format_ccsds_datetime_in, parse_ccsds_datetime,
    round_ccsds_value,
};
use crate::ccsds::error::{ccsds_missing_field, ccsds_parse_error};
use crate::ccsds::kvn::common::{KVNToken, tokenize_line};
use crate::ccsds::oem::{OEM, OEMMetadata, OEMSegment, OEMStateVector};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

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
    let mut meta_ref_frame_epoch: Option<String> = None;
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
                    "REF_FRAME_EPOCH" => meta_ref_frame_epoch = Some(value),
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
                            ref_frame_epoch: meta_ref_frame_epoch
                                .take()
                                .map(|raw| {
                                    parse_ccsds_datetime(
                                        &raw,
                                        meta_time_system.as_ref().unwrap_or(&CCSDSTimeSystem::UTC),
                                    )
                                })
                                .transpose()?,
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

/// Write an OEM message to KVN format.
pub fn write_oem(oem: &OEM) -> Result<String, BraheError> {
    let mut out = String::new();

    // Header
    out.push_str(&format!(
        "CCSDS_OEM_VERS = {:.1}\n",
        oem.header.format_version
    ));
    for comment in &oem.header.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }
    if let Some(ref class) = oem.header.classification {
        out.push_str(&format!("CLASSIFICATION = {}\n", class));
    }
    out.push_str(&format!(
        "CREATION_DATE = {}\n",
        format_ccsds_datetime_in(&oem.header.creation_date, &CCSDSTimeSystem::UTC)
    ));
    out.push_str(&format!("ORIGINATOR = {}\n", oem.header.originator));
    if let Some(ref msg_id) = oem.header.message_id {
        out.push_str(&format!("MESSAGE_ID = {}\n", msg_id));
    }

    // Segments
    for segment in &oem.segments {
        out.push('\n');

        // Metadata
        out.push_str("META_START\n");
        for comment in &segment.metadata.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }
        out.push_str(&format!("OBJECT_NAME = {}\n", segment.metadata.object_name));
        out.push_str(&format!("OBJECT_ID = {}\n", segment.metadata.object_id));
        out.push_str(&format!("CENTER_NAME = {}\n", segment.metadata.center_name));
        out.push_str(&format!("REF_FRAME = {}\n", segment.metadata.ref_frame));
        if let Some(ref epoch) = segment.metadata.ref_frame_epoch {
            out.push_str(&format!(
                "REF_FRAME_EPOCH = {}\n",
                format_ccsds_datetime_in(epoch, &segment.metadata.time_system)
            ));
        }
        out.push_str(&format!("TIME_SYSTEM = {}\n", segment.metadata.time_system));
        out.push_str(&format!(
            "START_TIME = {}\n",
            format_ccsds_datetime_in(&segment.metadata.start_time, &segment.metadata.time_system)
        ));
        if let Some(ref t) = segment.metadata.useable_start_time {
            out.push_str(&format!(
                "USEABLE_START_TIME = {}\n",
                format_ccsds_datetime_in(t, &segment.metadata.time_system)
            ));
        }
        if let Some(ref t) = segment.metadata.useable_stop_time {
            out.push_str(&format!(
                "USEABLE_STOP_TIME = {}\n",
                format_ccsds_datetime_in(t, &segment.metadata.time_system)
            ));
        }
        out.push_str(&format!(
            "STOP_TIME = {}\n",
            format_ccsds_datetime_in(&segment.metadata.stop_time, &segment.metadata.time_system)
        ));
        if let Some(ref interp) = segment.metadata.interpolation {
            out.push_str(&format!("INTERPOLATION = {}\n", interp));
        }
        if let Some(deg) = segment.metadata.interpolation_degree {
            out.push_str(&format!("INTERPOLATION_DEGREE = {}\n", deg));
        }
        out.push_str("META_STOP\n");

        // Data comments
        for comment in &segment.comments {
            out.push_str(&format!("COMMENT {}\n", comment));
        }

        // Ephemeris data lines
        for sv in &segment.states {
            let epoch_str = format_ccsds_datetime_in(&sv.epoch, &segment.metadata.time_system);
            // Convert m → km, m/s → km/s
            let x = sv.position[0] / 1000.0;
            let y = sv.position[1] / 1000.0;
            let z = sv.position[2] / 1000.0;
            let vx = sv.velocity[0] / 1000.0;
            let vy = sv.velocity[1] / 1000.0;
            let vz = sv.velocity[2] / 1000.0;

            if let Some(ref acc) = sv.acceleration {
                let ax = acc[0] / 1000.0;
                let ay = acc[1] / 1000.0;
                let az = acc[2] / 1000.0;
                out.push_str(&format!(
                    "{} {:15.6} {:15.6} {:15.6} {:15.9} {:15.9} {:15.9} {:15.9} {:15.9} {:15.9}\n",
                    epoch_str, x, y, z, vx, vy, vz, ax, ay, az
                ));
            } else {
                out.push_str(&format!(
                    "{} {:15.6} {:15.6} {:15.6} {:15.9} {:15.9} {:15.9}\n",
                    epoch_str, x, y, z, vx, vy, vz
                ));
            }
        }

        // Covariance blocks
        if !segment.covariances.is_empty() {
            out.push_str("\nCOVARIANCE_START\n");
            for cov in &segment.covariances {
                if let Some(ref epoch) = cov.epoch {
                    out.push_str(&format!(
                        "EPOCH = {}\n",
                        format_ccsds_datetime_in(epoch, &segment.metadata.time_system)
                    ));
                }
                if let Some(ref frame) = cov.cov_ref_frame {
                    out.push_str(&format!("COV_REF_FRAME = {}\n", frame));
                }
                for comment in &cov.comments {
                    out.push_str(&format!("COMMENT {}\n", comment));
                }

                // Convert m² → km² (factor 1e-6)
                let values =
                    covariance_to_lower_triangular(&cov.matrix, 1e-6).map(round_ccsds_value);
                let mut idx = 0;
                for row in 0..6 {
                    let line: Vec<String> = (0..=row)
                        .map(|_| {
                            let v = values[idx];
                            idx += 1;
                            format!("{:.10e}", v)
                        })
                        .collect();
                    out.push_str(&line.join(" "));
                    out.push('\n');
                }
            }
            out.push_str("COVARIANCE_STOP\n");
        }
    }

    Ok(out)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::ccsds::common::CCSDSTimeSystem;
    use crate::ccsds::kvn::parse_oem;

    use serial_test::parallel;
    #[test]
    #[parallel]
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
    #[parallel]
    fn test_parse_oem_example2_unsupported_time_system() {
        // OEMExample2.txt uses TIME_SYSTEM = MRT which is not supported for epoch conversion
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample2.txt").unwrap();
        let result = parse_oem(&content);
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
    #[parallel]
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
    #[parallel]
    fn test_parse_oem_with_header_comment() {
        let content =
            std::fs::read_to_string("test_assets/ccsds/oem/OEMExampleWithHeaderComment.txt")
                .unwrap();
        let oem = parse_oem(&content).unwrap();

        assert!(!oem.header.comments.is_empty());
    }

    #[test]
    #[parallel]
    fn test_parse_oem_iss_truncated() {
        let content =
            std::fs::read_to_string("test_assets/ccsds/oem/ISS.resampled.truncated.txt").unwrap();
        let oem = parse_oem(&content).unwrap();

        assert_eq!(oem.segments.len(), 1);
        assert!(!oem.segments[0].states.is_empty());
    }

    #[test]
    #[parallel]
    fn test_parse_oem_lowercase_value() {
        let content =
            std::fs::read_to_string("test_assets/ccsds/oem/oemLowerCaseValue.oem").unwrap();
        // This should either parse successfully or give a meaningful error
        let result = parse_oem(&content);
        // For now just verify it doesn't panic
        let _ = result;
    }

    #[test]
    #[parallel]
    fn test_oem_kvn_round_trip_example1() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let oem = parse_oem(&content).unwrap();

        // Write
        let written = write_oem(&oem).unwrap();

        // Re-parse
        let oem2 = parse_oem(&written).unwrap();

        // Verify
        assert_eq!(oem.segments.len(), oem2.segments.len());
        assert_eq!(oem.header.originator, oem2.header.originator);

        for (seg1, seg2) in oem.segments.iter().zip(oem2.segments.iter()) {
            assert_eq!(seg1.metadata.object_name, seg2.metadata.object_name);
            assert_eq!(seg1.metadata.ref_frame, seg2.metadata.ref_frame);
            assert_eq!(seg1.states.len(), seg2.states.len());
            assert_eq!(seg1.covariances.len(), seg2.covariances.len());

            // Check state vectors are close
            for (s1, s2) in seg1.states.iter().zip(seg2.states.iter()) {
                for i in 0..3 {
                    assert!(
                        (s1.position[i] - s2.position[i]).abs() < 1.0,
                        "Position mismatch: {} vs {}",
                        s1.position[i],
                        s2.position[i]
                    );
                    assert!(
                        (s1.velocity[i] - s2.velocity[i]).abs() < 0.001,
                        "Velocity mismatch: {} vs {}",
                        s1.velocity[i],
                        s2.velocity[i]
                    );
                }
            }

            // Check covariance matrices
            for (c1, c2) in seg1.covariances.iter().zip(seg2.covariances.iter()) {
                assert_eq!(c1.cov_ref_frame, c2.cov_ref_frame);
                for i in 0..6 {
                    for j in 0..6 {
                        let rel_err = if c1.matrix[(i, j)].abs() > 1e-20 {
                            ((c1.matrix[(i, j)] - c2.matrix[(i, j)]) / c1.matrix[(i, j)]).abs()
                        } else {
                            (c1.matrix[(i, j)] - c2.matrix[(i, j)]).abs()
                        };
                        assert!(
                            rel_err < 1e-4,
                            "Covariance mismatch at ({},{}): {} vs {} (rel_err: {})",
                            i,
                            j,
                            c1.matrix[(i, j)],
                            c2.matrix[(i, j)],
                            rel_err
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[parallel]
    fn test_oem_kvn_round_trip_example5() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample5.txt").unwrap();
        let oem = parse_oem(&content).unwrap();
        let written = write_oem(&oem).unwrap();
        let oem2 = parse_oem(&written).unwrap();

        assert_eq!(oem.segments[0].states.len(), oem2.segments[0].states.len());
        assert_eq!(
            oem.segments[0].metadata.object_name,
            oem2.segments[0].metadata.object_name
        );
    }

    // ------------------------------------------------------------------
    // OEM writer: optional field coverage
    // ------------------------------------------------------------------

    #[test]
    #[parallel]
    fn test_oem_write_header_classification() {
        // OEMExample1 has CLASSIFICATION = public, test-data
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let oem = parse_oem(&content).unwrap();
        assert!(oem.header.classification.is_some());

        let written = write_oem(&oem).unwrap();
        assert!(written.contains("CLASSIFICATION = public, test-data"));
    }

    #[test]
    #[parallel]
    fn test_oem_write_header_message_id() {
        // OEMExample3 has MESSAGE_ID
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample3.txt").unwrap();
        let oem = parse_oem(&content).unwrap();
        assert!(oem.header.message_id.is_some());

        let written = write_oem(&oem).unwrap();
        assert!(written.contains("MESSAGE_ID = OEM 201113719185"));
    }

    #[test]
    #[parallel]
    fn test_oem_write_ref_frame_epoch() {
        use crate::ccsds::common::{CCSDSRefFrame, CCSDSTimeSystem, ODMHeader};
        use crate::ccsds::oem::{OEMMetadata, OEMSegment, OEMStateVector};
        use crate::time::Epoch;

        let epoch = Epoch::from_datetime(2024, 6, 1, 0, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let ref_epoch =
            Epoch::from_datetime(2000, 1, 1, 12, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let mut metadata = OEMMetadata::new(
            "REF_SAT".to_string(),
            "2024-001A".to_string(),
            "EARTH".to_string(),
            CCSDSRefFrame::TOD,
            CCSDSTimeSystem::UTC,
            epoch,
            epoch,
        );
        metadata.ref_frame_epoch = Some(ref_epoch);
        let mut seg = OEMSegment::new(metadata);
        seg.push_state(OEMStateVector::new(
            epoch,
            [7000e3, 0.0, 0.0],
            [0.0, 7500.0, 0.0],
        ));
        let oem = OEM {
            header: ODMHeader {
                format_version: 3.0,
                classification: None,
                creation_date: epoch,
                originator: "TEST".to_string(),
                message_id: None,
                comments: Vec::new(),
            },
            segments: vec![seg],
        };
        let written = write_oem(&oem).unwrap();
        assert!(written.contains("REF_FRAME_EPOCH ="));
        assert!(written.contains("2000-01-01"));
        let oem2 = parse_oem(&written).unwrap();
        assert!(oem2.segments[0].metadata.ref_frame_epoch.is_some());
    }

    #[test]
    #[parallel]
    fn test_oem_write_useable_times() {
        // OEMExample1 has USEABLE_START_TIME and USEABLE_STOP_TIME
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let oem = parse_oem(&content).unwrap();
        assert!(oem.segments[0].metadata.useable_start_time.is_some());
        assert!(oem.segments[0].metadata.useable_stop_time.is_some());

        let written = write_oem(&oem).unwrap();
        assert!(written.contains("USEABLE_START_TIME ="));
        assert!(written.contains("USEABLE_STOP_TIME ="));
    }

    #[test]
    #[parallel]
    fn test_oem_write_interpolation() {
        // OEMExample1 has INTERPOLATION and INTERPOLATION_DEGREE
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let oem = parse_oem(&content).unwrap();
        assert!(oem.segments[0].metadata.interpolation.is_some());
        assert!(oem.segments[0].metadata.interpolation_degree.is_some());

        let written = write_oem(&oem).unwrap();
        assert!(written.contains("INTERPOLATION = HERMITE"));
        assert!(written.contains("INTERPOLATION_DEGREE = 7"));
    }

    #[test]
    #[parallel]
    fn test_oem_write_acceleration() {
        use crate::ccsds::common::{CCSDSRefFrame, CCSDSTimeSystem, ODMHeader};
        use crate::ccsds::oem::{OEMMetadata, OEMSegment, OEMStateVector};
        use crate::time::Epoch;

        let epoch = Epoch::from_datetime(2024, 6, 1, 0, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let sv = OEMStateVector::new(epoch, [7000e3, 0.0, 0.0], [0.0, 7500.0, 0.0])
            .with_acceleration([0.001, -0.002, 0.003]);

        let metadata = OEMMetadata::new(
            "ACCEL_SAT".to_string(),
            "2024-001A".to_string(),
            "EARTH".to_string(),
            CCSDSRefFrame::J2000,
            CCSDSTimeSystem::UTC,
            epoch,
            epoch,
        );
        let mut seg = OEMSegment::new(metadata);
        seg.push_state(sv);

        let oem = OEM {
            header: ODMHeader {
                format_version: 3.0,
                classification: None,
                creation_date: epoch,
                originator: "TEST".to_string(),
                message_id: None,
                comments: Vec::new(),
            },
            segments: vec![seg],
        };

        let written = write_oem(&oem).unwrap();
        // Acceleration columns produce 9 space-separated values after epoch
        // Verify the line has 9 numeric columns (position, velocity, acceleration)
        let data_lines: Vec<&str> = written.lines().filter(|l| l.starts_with("2024")).collect();
        assert_eq!(data_lines.len(), 1);
        let cols: Vec<&str> = data_lines[0].split_whitespace().collect();
        // epoch + 3 pos + 3 vel + 3 acc = 10
        assert_eq!(
            cols.len(),
            10,
            "Expected 10 columns for state with acceleration"
        );

        // Round-trip: re-parse and verify acceleration survives
        let oem2 = parse_oem(&written).unwrap();
        let sv2 = &oem2.segments[0].states[0];
        assert!(sv2.acceleration.is_some());
        let acc = sv2.acceleration.unwrap();
        // Units are m/s², written as km/s², so after round-trip converted back
        assert!((acc[0] - 0.001).abs() < 1e-6);
        assert!((acc[1] - (-0.002)).abs() < 1e-6);
        assert!((acc[2] - 0.003).abs() < 1e-6);
    }

    #[test]
    #[parallel]
    fn test_oem_write_covariance_with_epoch_and_frame() {
        use crate::ccsds::common::{CCSDSCovariance, CCSDSRefFrame, CCSDSTimeSystem, ODMHeader};
        use crate::ccsds::oem::{OEMMetadata, OEMSegment, OEMStateVector};
        use crate::time::Epoch;
        use nalgebra::SMatrix;

        let epoch = Epoch::from_datetime(2024, 6, 1, 0, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);

        let metadata = OEMMetadata::new(
            "COV_SAT".to_string(),
            "2024-002A".to_string(),
            "EARTH".to_string(),
            CCSDSRefFrame::J2000,
            CCSDSTimeSystem::UTC,
            epoch,
            epoch,
        );
        let mut seg = OEMSegment::new(metadata);
        seg.push_state(OEMStateVector::new(
            epoch,
            [7000e3, 0.0, 0.0],
            [0.0, 7500.0, 0.0],
        ));

        // Add covariance with optional epoch and cov_ref_frame
        let mut matrix = SMatrix::<f64, 6, 6>::zeros();
        matrix[(0, 0)] = 1.0e6; // 1 km^2 in m^2
        matrix[(1, 1)] = 2.0e6;
        matrix[(2, 2)] = 3.0e6;
        matrix[(3, 3)] = 1.0;
        matrix[(4, 4)] = 2.0;
        matrix[(5, 5)] = 3.0;

        let cov_epoch =
            Epoch::from_datetime(2024, 6, 1, 0, 30, 0.0, 0.0, crate::time::TimeSystem::UTC);
        seg.covariances.push(CCSDSCovariance {
            epoch: Some(cov_epoch),
            cov_ref_frame: Some(CCSDSRefFrame::RTN),
            matrix,
            comments: vec!["Test covariance comment".to_string()],
        });

        let oem = OEM {
            header: ODMHeader {
                format_version: 3.0,
                classification: None,
                creation_date: epoch,
                originator: "TEST".to_string(),
                message_id: None,
                comments: Vec::new(),
            },
            segments: vec![seg],
        };

        let written = write_oem(&oem).unwrap();
        assert!(written.contains("COVARIANCE_START"));
        assert!(written.contains("COVARIANCE_STOP"));
        assert!(written.contains("EPOCH ="));
        assert!(written.contains("COV_REF_FRAME = RTN"));
        assert!(written.contains("COMMENT Test covariance comment"));
    }

    #[test]
    #[parallel]
    fn test_oem_write_data_block_comments() {
        use crate::ccsds::common::{CCSDSRefFrame, CCSDSTimeSystem, ODMHeader};
        use crate::ccsds::oem::{OEMMetadata, OEMSegment, OEMStateVector};
        use crate::time::Epoch;

        let epoch = Epoch::from_datetime(2024, 6, 1, 0, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);

        let metadata = OEMMetadata::new(
            "CMT_SAT".to_string(),
            "2024-003A".to_string(),
            "EARTH".to_string(),
            CCSDSRefFrame::J2000,
            CCSDSTimeSystem::UTC,
            epoch,
            epoch,
        );
        let mut seg = OEMSegment::new(metadata);
        seg.comments = vec![
            "Data block comment line 1".to_string(),
            "Data block comment line 2".to_string(),
        ];
        seg.push_state(OEMStateVector::new(
            epoch,
            [7000e3, 0.0, 0.0],
            [0.0, 7500.0, 0.0],
        ));

        let oem = OEM {
            header: ODMHeader {
                format_version: 3.0,
                classification: None,
                creation_date: epoch,
                originator: "TEST".to_string(),
                message_id: None,
                comments: Vec::new(),
            },
            segments: vec![seg],
        };

        let written = write_oem(&oem).unwrap();
        assert!(written.contains("COMMENT Data block comment line 1"));
        assert!(written.contains("COMMENT Data block comment line 2"));
    }

    #[test]
    #[parallel]
    fn test_oem_write_all_optional_fields_round_trip() {
        // Build OEM with all optional metadata fields set
        use crate::ccsds::common::{CCSDSRefFrame, CCSDSTimeSystem, ODMHeader};
        use crate::ccsds::oem::{OEMMetadata, OEMSegment, OEMStateVector};
        use crate::time::Epoch;

        let epoch = Epoch::from_datetime(2024, 6, 1, 0, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let ref_epoch =
            Epoch::from_datetime(2000, 1, 1, 12, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);
        let mut metadata = OEMMetadata::new(
            "ALL_OPT".to_string(),
            "2024-001A".to_string(),
            "EARTH".to_string(),
            CCSDSRefFrame::TOD,
            CCSDSTimeSystem::UTC,
            epoch,
            epoch,
        );
        metadata.ref_frame_epoch = Some(ref_epoch);
        metadata.useable_start_time = Some(epoch);
        metadata.useable_stop_time = Some(epoch);
        metadata.interpolation = Some("HERMITE".to_string());
        metadata.interpolation_degree = Some(7);

        let mut seg = OEMSegment::new(metadata);
        seg.push_state(OEMStateVector::new(
            epoch,
            [7000e3, 0.0, 0.0],
            [0.0, 7500.0, 0.0],
        ));

        let oem = OEM {
            header: ODMHeader {
                format_version: 3.0,
                classification: None,
                creation_date: epoch,
                originator: "TEST".to_string(),
                message_id: None,
                comments: Vec::new(),
            },
            segments: vec![seg],
        };

        let written = write_oem(&oem).unwrap();
        let oem2 = parse_oem(&written).unwrap();

        let seg = &oem2.segments[0];
        assert!(seg.metadata.ref_frame_epoch.is_some());
        assert!(seg.metadata.useable_start_time.is_some());
        assert!(seg.metadata.useable_stop_time.is_some());
        assert_eq!(seg.metadata.interpolation.as_deref(), Some("HERMITE"));
        assert_eq!(seg.metadata.interpolation_degree, Some(7));
        assert_eq!(seg.metadata.ref_frame, CCSDSRefFrame::TOD);
    }

    #[test]
    #[serial_test::parallel]
    fn test_oem_covariance_blocks_keep_their_epoch() {
        use crate::ccsds::common::CCSDSFormat;
        use crate::ccsds::oem::OEM;

        // The OEM is the one message whose covariance block defines EPOCH
        // (CCSDS 502.0-B-3 subsection 5.2.5.3), because each matrix belongs to
        // a navigation solution of its own.
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let oem = OEM::from_str(&content).unwrap();
        let covariance_epoch = oem
            .segments
            .iter()
            .flat_map(|s| s.covariances.iter())
            .find_map(|c| c.epoch)
            .expect("fixture has a covariance epoch");

        for format in [CCSDSFormat::KVN, CCSDSFormat::XML] {
            let reparsed = OEM::from_str(&oem.to_string(format).unwrap()).unwrap();
            let reparsed_epoch = reparsed
                .segments
                .iter()
                .flat_map(|s| s.covariances.iter())
                .find_map(|c| c.epoch);
            assert_eq!(
                reparsed_epoch.map(|e| format_ccsds_datetime_in(&e, &CCSDSTimeSystem::UTC)),
                Some(format_ccsds_datetime_in(
                    &covariance_epoch,
                    &CCSDSTimeSystem::UTC
                )),
                "{:?} dropped the OEM covariance epoch",
                format
            );
        }
    }
}
