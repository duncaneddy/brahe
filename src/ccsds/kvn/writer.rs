/*!
 * KVN writer for CCSDS OEM, OMM, and OPM messages.
 */

use crate::ccsds::common::{covariance_to_lower_triangular, format_ccsds_datetime};
use crate::ccsds::oem::OEM;
use crate::utils::errors::BraheError;

/// Write an OEM message to KVN format.
pub fn write_oem(oem: &OEM) -> Result<String, BraheError> {
    let mut out = String::new();

    // Header
    out.push_str(&format!(
        "CCSDS_OEM_VERS = {:.1}\n",
        oem.header.format_version
    ));
    if let Some(ref class) = oem.header.classification {
        out.push_str(&format!("CLASSIFICATION = {}\n", class));
    }
    for comment in &oem.header.comments {
        out.push_str(&format!("COMMENT {}\n", comment));
    }
    out.push_str(&format!(
        "CREATION_DATE = {}\n",
        format_ccsds_datetime(&oem.header.creation_date)
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
                format_ccsds_datetime(epoch)
            ));
        }
        out.push_str(&format!("TIME_SYSTEM = {}\n", segment.metadata.time_system));
        out.push_str(&format!(
            "START_TIME = {}\n",
            format_ccsds_datetime(&segment.metadata.start_time)
        ));
        if let Some(ref t) = segment.metadata.useable_start_time {
            out.push_str(&format!(
                "USEABLE_START_TIME = {}\n",
                format_ccsds_datetime(t)
            ));
        }
        if let Some(ref t) = segment.metadata.useable_stop_time {
            out.push_str(&format!(
                "USEABLE_STOP_TIME = {}\n",
                format_ccsds_datetime(t)
            ));
        }
        out.push_str(&format!(
            "STOP_TIME = {}\n",
            format_ccsds_datetime(&segment.metadata.stop_time)
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
            let epoch_str = format_ccsds_datetime(&sv.epoch);
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
                    out.push_str(&format!("EPOCH = {}\n", format_ccsds_datetime(epoch)));
                }
                if let Some(ref frame) = cov.cov_ref_frame {
                    out.push_str(&format!("COV_REF_FRAME = {}\n", frame));
                }
                for comment in &cov.comments {
                    out.push_str(&format!("COMMENT {}\n", comment));
                }

                // Convert m² → km² (factor 1e-6)
                let values = covariance_to_lower_triangular(&cov.matrix, 1e-6);
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

/// Write an OMM message to KVN format.
///
/// Stub — implemented in Stage 5.
pub fn write_omm(_omm: &crate::ccsds::omm::OMM) -> Result<String, BraheError> {
    Err(BraheError::Error(
        "OMM KVN writer not yet implemented".to_string(),
    ))
}

/// Write an OPM message to KVN format.
///
/// Stub — implemented in Stage 6.
pub fn write_opm(_opm: &crate::ccsds::opm::OPM) -> Result<String, BraheError> {
    Err(BraheError::Error(
        "OPM KVN writer not yet implemented".to_string(),
    ))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::ccsds::kvn::parse_oem;

    fn setup_eop() {
        use crate::eop::*;
        let eop = StaticEOPProvider::new();
        set_global_eop_provider(eop);
    }

    #[test]
    fn test_oem_kvn_round_trip_example1() {
        setup_eop();

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
    fn test_oem_kvn_round_trip_example5() {
        setup_eop();

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
}
