/*!
 * Brahe interop for CCSDS types.
 *
 * Provides conversion between CCSDS message types and brahe's native
 * trajectory, propagator, and state vector types.
 */

use nalgebra::SVector;

use crate::ccsds::common::{
    CCSDSRefFrame, CCSDSTimeSystem, ODMHeader, format_ccsds_datetime, parse_ccsds_datetime,
};
use crate::ccsds::oem::OEM;
use crate::ccsds::omm::{OMM, OMMMetadata, OMMTleParameters, OMMeanElements};
use crate::time::Epoch;
use crate::trajectories::sorbit_trajectory::SOrbitTrajectory;
use crate::trajectories::traits::{OrbitFrame, OrbitRepresentation, Trajectory};
use crate::types::GPRecord;
use crate::utils::errors::BraheError;

/// Map a CCSDS reference frame to a brahe `OrbitFrame`.
///
/// Only inertial and terrestrial frames supported by brahe are mapped.
/// Orbit-relative frames (RTN, TNW, RSW) and exotic frames return an error.
pub fn ccsds_ref_frame_to_orbit_frame(frame: &CCSDSRefFrame) -> Result<OrbitFrame, BraheError> {
    match frame {
        CCSDSRefFrame::EME2000 => Ok(OrbitFrame::EME2000),
        CCSDSRefFrame::J2000 => Ok(OrbitFrame::EME2000),
        CCSDSRefFrame::GCRF => Ok(OrbitFrame::GCRF),
        CCSDSRefFrame::ITRF2000
        | CCSDSRefFrame::ITRF93
        | CCSDSRefFrame::ITRF97
        | CCSDSRefFrame::ITRF2005
        | CCSDSRefFrame::ITRF2008
        | CCSDSRefFrame::ITRF2014 => Ok(OrbitFrame::ITRF),
        CCSDSRefFrame::TEME => Err(BraheError::Error(
            "Cannot map CCSDS frame 'TEME' to brahe OrbitFrame. TEME is not equivalent to GCRF or EME2000. \
             Use frame conversion before creating a trajectory.".to_string(),
        )),
        CCSDSRefFrame::TOD => Err(BraheError::Error(
            "Cannot map CCSDS frame 'TOD' to brahe OrbitFrame. TOD is not equivalent to GCRF or EME2000. \
             Use frame conversion before creating a trajectory.".to_string(),
        )),
        _ => Err(BraheError::Error(format!(
            "Cannot map CCSDS frame '{}' to brahe OrbitFrame",
            frame
        ))),
    }
}

impl OEM {
    /// Convert a single OEM segment to a brahe `SOrbitTrajectory`.
    ///
    /// The trajectory contains Cartesian state vectors (position/velocity)
    /// in the reference frame specified by the segment metadata.
    ///
    /// # Arguments
    ///
    /// * `segment_idx` - Index of the segment to convert (0-based)
    ///
    /// # Returns
    ///
    /// * `Result<SOrbitTrajectory, BraheError>` - Trajectory or error
    pub fn segment_to_orbit_trajectory(
        &self,
        segment_idx: usize,
    ) -> Result<SOrbitTrajectory, BraheError> {
        let segment = self.segments.get(segment_idx).ok_or_else(|| {
            BraheError::OutOfBoundsError(format!(
                "OEM segment index {} out of range (have {})",
                segment_idx,
                self.segments.len()
            ))
        })?;

        let orbit_frame = ccsds_ref_frame_to_orbit_frame(&segment.metadata.ref_frame)?;

        let mut traj = SOrbitTrajectory::new(orbit_frame, OrbitRepresentation::Cartesian, None);

        traj.name = Some(segment.metadata.object_name.clone());

        for sv in &segment.states {
            let state = SVector::<f64, 6>::new(
                sv.position[0],
                sv.position[1],
                sv.position[2],
                sv.velocity[0],
                sv.velocity[1],
                sv.velocity[2],
            );
            traj.add(sv.epoch, state);
        }

        Ok(traj)
    }

    /// Convert all segments to brahe `SOrbitTrajectory` objects.
    ///
    /// # Returns
    ///
    /// * `Result<Vec<SOrbitTrajectory>, BraheError>` - One trajectory per segment
    pub fn to_orbit_trajectories(&self) -> Result<Vec<SOrbitTrajectory>, BraheError> {
        (0..self.segments.len())
            .map(|i| self.segment_to_orbit_trajectory(i))
            .collect()
    }
}

impl TryFrom<&OEM> for SOrbitTrajectory {
    type Error = BraheError;

    /// Convert a single-segment OEM to a trajectory.
    ///
    /// Returns an error if the OEM has zero or more than one segment.
    fn try_from(oem: &OEM) -> Result<Self, Self::Error> {
        if oem.segments.len() != 1 {
            return Err(BraheError::Error(format!(
                "TryFrom<&OEM> requires exactly 1 segment, but OEM has {}",
                oem.segments.len()
            )));
        }
        oem.segment_to_orbit_trajectory(0)
    }
}

impl OMM {
    /// Convert a GPRecord into an OMM message.
    ///
    /// Validates that required orbital element fields are present (epoch,
    /// eccentricity, inclination, ra_of_asc_node, arg_of_pericenter,
    /// mean_anomaly) and builds an OMM with defaults for missing metadata.
    ///
    /// # Arguments
    ///
    /// * `record` - GPRecord to convert
    ///
    /// # Returns
    ///
    /// * `Result<OMM, BraheError>` - OMM message or error if required fields are missing
    pub fn from_gp_record(record: &GPRecord) -> Result<OMM, BraheError> {
        // Validate required fields
        let epoch_str = record.epoch.as_deref().ok_or_else(|| {
            BraheError::Error("GPRecord missing required field: EPOCH".to_string())
        })?;
        let eccentricity = record.eccentricity.ok_or_else(|| {
            BraheError::Error("GPRecord missing required field: ECCENTRICITY".to_string())
        })?;
        let inclination = record.inclination.ok_or_else(|| {
            BraheError::Error("GPRecord missing required field: INCLINATION".to_string())
        })?;
        let ra_of_asc_node = record.ra_of_asc_node.ok_or_else(|| {
            BraheError::Error("GPRecord missing required field: RA_OF_ASC_NODE".to_string())
        })?;
        let arg_of_pericenter = record.arg_of_pericenter.ok_or_else(|| {
            BraheError::Error("GPRecord missing required field: ARG_OF_PERICENTER".to_string())
        })?;
        let mean_anomaly = record.mean_anomaly.ok_or_else(|| {
            BraheError::Error("GPRecord missing required field: MEAN_ANOMALY".to_string())
        })?;

        // Parse time system (needed for epoch parsing)
        let time_system = record
            .time_system
            .as_deref()
            .map(CCSDSTimeSystem::parse)
            .transpose()
            .unwrap_or(Some(CCSDSTimeSystem::UTC))
            .unwrap_or(CCSDSTimeSystem::UTC);

        // Parse epoch
        let epoch = parse_ccsds_datetime(epoch_str, &time_system)?;

        // Parse header fields
        let format_version = record
            .ccsds_omm_vers
            .as_deref()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(3.0);

        let creation_date = record
            .creation_date
            .as_deref()
            .and_then(|s| parse_ccsds_datetime(s, &CCSDSTimeSystem::UTC).ok())
            .unwrap_or_else(Epoch::now);

        let originator = record
            .originator
            .clone()
            .unwrap_or_else(|| "UNKNOWN".to_string());

        // Parse reference frame
        let ref_frame = record
            .ref_frame
            .as_deref()
            .map(CCSDSRefFrame::parse)
            .unwrap_or(CCSDSRefFrame::TEME);

        // Parse metadata
        let metadata = OMMMetadata::new(
            record
                .object_name
                .clone()
                .unwrap_or_else(|| "UNKNOWN".to_string()),
            record
                .object_id
                .clone()
                .unwrap_or_else(|| "UNKNOWN".to_string()),
            record
                .center_name
                .clone()
                .unwrap_or_else(|| "EARTH".to_string()),
            ref_frame,
            time_system,
            record
                .mean_element_theory
                .clone()
                .unwrap_or_else(|| "SGP4".to_string()),
        );

        // Build mean elements
        let mut mean_elements = OMMeanElements::new(
            epoch,
            eccentricity,
            inclination,
            ra_of_asc_node,
            arg_of_pericenter,
            mean_anomaly,
        );
        mean_elements.mean_motion = record.mean_motion;

        // Build TLE parameters if any TLE field is present
        let has_tle_fields = record.ephemeris_type.is_some()
            || record.classification_type.is_some()
            || record.norad_cat_id.is_some()
            || record.element_set_no.is_some()
            || record.rev_at_epoch.is_some()
            || record.bstar.is_some()
            || record.mean_motion_dot.is_some()
            || record.mean_motion_ddot.is_some();

        let tle_parameters = if has_tle_fields {
            Some(OMMTleParameters {
                ephemeris_type: record.ephemeris_type.map(|v| v as u32),
                classification_type: record
                    .classification_type
                    .as_deref()
                    .and_then(|s| s.chars().next()),
                norad_cat_id: record.norad_cat_id,
                element_set_no: record.element_set_no.map(|v| v as u32),
                rev_at_epoch: record.rev_at_epoch,
                bstar: record.bstar,
                bterm: None,
                mean_motion_dot: record.mean_motion_dot,
                mean_motion_ddot: record.mean_motion_ddot,
                agom: None,
                comments: Vec::new(),
            })
        } else {
            None
        };

        Ok(OMM {
            header: ODMHeader {
                format_version,
                classification: None,
                creation_date,
                originator,
                message_id: None,
                comments: Vec::new(),
            },
            metadata,
            mean_elements,
            tle_parameters,
            spacecraft_parameters: None,
            covariance: None,
            user_defined: None,
            comments: Vec::new(),
        })
    }

    /// Convert an OMM message to a GPRecord.
    ///
    /// Maps all OMM fields back to the `Option<T>` GPRecord fields.
    /// This conversion is infallible since all GPRecord fields are optional.
    ///
    /// # Returns
    ///
    /// * `GPRecord` - GP record with fields populated from the OMM
    pub fn to_gp_record(&self) -> GPRecord {
        let epoch_str = format_ccsds_datetime(&self.mean_elements.epoch);

        GPRecord {
            ccsds_omm_vers: Some(format!("{:.1}", self.header.format_version)),
            comment: None,
            creation_date: Some(format_ccsds_datetime(&self.header.creation_date)),
            originator: Some(self.header.originator.clone()),
            object_name: Some(self.metadata.object_name.clone()),
            object_id: Some(self.metadata.object_id.clone()),
            center_name: Some(self.metadata.center_name.clone()),
            ref_frame: Some(format!("{}", self.metadata.ref_frame)),
            time_system: Some(format!("{}", self.metadata.time_system)),
            mean_element_theory: Some(self.metadata.mean_element_theory.clone()),
            epoch: Some(epoch_str),
            mean_motion: self.mean_elements.mean_motion,
            eccentricity: Some(self.mean_elements.eccentricity),
            inclination: Some(self.mean_elements.inclination),
            ra_of_asc_node: Some(self.mean_elements.ra_of_asc_node),
            arg_of_pericenter: Some(self.mean_elements.arg_of_pericenter),
            mean_anomaly: Some(self.mean_elements.mean_anomaly),
            ephemeris_type: self
                .tle_parameters
                .as_ref()
                .and_then(|t| t.ephemeris_type.map(|v| v as u8)),
            classification_type: self
                .tle_parameters
                .as_ref()
                .and_then(|t| t.classification_type.map(|c| c.to_string())),
            norad_cat_id: self.tle_parameters.as_ref().and_then(|t| t.norad_cat_id),
            element_set_no: self
                .tle_parameters
                .as_ref()
                .and_then(|t| t.element_set_no.map(|v| v as u16)),
            rev_at_epoch: self.tle_parameters.as_ref().and_then(|t| t.rev_at_epoch),
            bstar: self.tle_parameters.as_ref().and_then(|t| t.bstar),
            mean_motion_dot: self.tle_parameters.as_ref().and_then(|t| t.mean_motion_dot),
            mean_motion_ddot: self
                .tle_parameters
                .as_ref()
                .and_then(|t| t.mean_motion_ddot),
            // Derived fields not present in OMM
            semimajor_axis: None,
            period: None,
            apoapsis: None,
            periapsis: None,
            object_type: None,
            rcs_size: None,
            country_code: None,
            launch_date: None,
            site: None,
            decay_date: None,
            file: None,
            gp_id: None,
            tle_line0: None,
            tle_line1: None,
            tle_line2: None,
        }
    }
}

impl GPRecord {
    /// Convert this GPRecord to a CCSDS OMM message.
    ///
    /// Delegates to `OMM::from_gp_record`. Validates that required orbital
    /// element fields are present (epoch, eccentricity, inclination,
    /// ra_of_asc_node, arg_of_pericenter, mean_anomaly).
    ///
    /// # Returns
    ///
    /// * `Result<OMM, BraheError>` - OMM message or error if required fields are missing
    pub fn to_omm(&self) -> Result<OMM, BraheError> {
        OMM::from_gp_record(self)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::ccsds::oem::OEM;
    use crate::trajectories::traits::Trajectory;

    #[test]
    fn test_oem_to_trajectory_example4() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample4.txt").unwrap();
        let oem = OEM::from_str(&content).unwrap();

        let traj = oem.segment_to_orbit_trajectory(0).unwrap();
        assert_eq!(traj.len(), 3);
        assert_eq!(traj.name.as_deref(), Some("MARS GLOBAL SURVEYOR"));
        assert_eq!(traj.frame, OrbitFrame::EME2000);

        // Verify first state
        let (_epoch, state) = traj.first().unwrap();
        assert!((state[0] - 2789.619 * 1000.0).abs() < 1.0);
        assert!((state[3] - 4.73372 * 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_oem_to_trajectory_example5() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample5.txt").unwrap();
        let oem = OEM::from_str(&content).unwrap();

        let traj = oem.segment_to_orbit_trajectory(0).unwrap();
        assert_eq!(traj.len(), 49);
        assert_eq!(traj.frame, OrbitFrame::GCRF);
    }

    #[test]
    fn test_oem_to_trajectories_multi_segment() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let oem = OEM::from_str(&content).unwrap();

        let trajs = oem.to_orbit_trajectories().unwrap();
        assert_eq!(trajs.len(), 3);
    }

    #[test]
    fn test_oem_try_from_single_segment() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample4.txt").unwrap();
        let oem = OEM::from_str(&content).unwrap();

        let traj = SOrbitTrajectory::try_from(&oem).unwrap();
        assert_eq!(traj.len(), 3);
    }

    #[test]
    fn test_oem_try_from_multi_segment_fails() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let oem = OEM::from_str(&content).unwrap();

        assert!(SOrbitTrajectory::try_from(&oem).is_err());
    }

    #[test]
    fn test_oem_segment_out_of_bounds() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample4.txt").unwrap();
        let oem = OEM::from_str(&content).unwrap();

        assert!(oem.segment_to_orbit_trajectory(5).is_err());
    }

    #[test]
    fn test_ccsds_ref_frame_mapping() {
        assert_eq!(
            ccsds_ref_frame_to_orbit_frame(&CCSDSRefFrame::EME2000).unwrap(),
            OrbitFrame::EME2000
        );
        assert_eq!(
            ccsds_ref_frame_to_orbit_frame(&CCSDSRefFrame::GCRF).unwrap(),
            OrbitFrame::GCRF
        );
        assert_eq!(
            ccsds_ref_frame_to_orbit_frame(&CCSDSRefFrame::ITRF2000).unwrap(),
            OrbitFrame::ITRF
        );
        assert_eq!(
            ccsds_ref_frame_to_orbit_frame(&CCSDSRefFrame::J2000).unwrap(),
            OrbitFrame::EME2000
        );
        // Orbit-relative frames should fail
        assert!(ccsds_ref_frame_to_orbit_frame(&CCSDSRefFrame::RTN).is_err());
        // TEME and TOD should fail (not equivalent to GCRF/EME2000)
        assert!(ccsds_ref_frame_to_orbit_frame(&CCSDSRefFrame::TEME).is_err());
        assert!(ccsds_ref_frame_to_orbit_frame(&CCSDSRefFrame::TOD).is_err());
    }

    fn sample_gp_record_json() -> &'static str {
        r#"{
            "CCSDS_OMM_VERS": "3.0",
            "CREATION_DATE": "2024-01-15 12:00:00",
            "ORIGINATOR": "18 SDS",
            "OBJECT_NAME": "ISS (ZARYA)",
            "OBJECT_ID": "1998-067A",
            "CENTER_NAME": "EARTH",
            "REF_FRAME": "TEME",
            "TIME_SYSTEM": "UTC",
            "MEAN_ELEMENT_THEORY": "SGP4",
            "EPOCH": "2024-01-15T12:00:00.000000",
            "MEAN_MOTION": "15.50000000",
            "ECCENTRICITY": "0.00010000",
            "INCLINATION": "51.6400",
            "RA_OF_ASC_NODE": "200.0000",
            "ARG_OF_PERICENTER": "100.0000",
            "MEAN_ANOMALY": "260.0000",
            "EPHEMERIS_TYPE": "0",
            "CLASSIFICATION_TYPE": "U",
            "NORAD_CAT_ID": "25544",
            "ELEMENT_SET_NO": "999",
            "REV_AT_EPOCH": "45000",
            "BSTAR": "0.00034100",
            "MEAN_MOTION_DOT": "0.00001000",
            "MEAN_MOTION_DDOT": "0.00000000"
        }"#
    }

    #[test]
    fn test_gp_record_to_omm() {
        let record: GPRecord = serde_json::from_str(sample_gp_record_json()).unwrap();
        let omm = record.to_omm().unwrap();

        // Header
        assert!((omm.header.format_version - 3.0).abs() < 1e-10);
        assert_eq!(omm.header.originator, "18 SDS");

        // Metadata
        assert_eq!(omm.metadata.object_name, "ISS (ZARYA)");
        assert_eq!(omm.metadata.object_id, "1998-067A");
        assert_eq!(omm.metadata.center_name, "EARTH");
        assert_eq!(omm.metadata.ref_frame, CCSDSRefFrame::TEME);
        assert_eq!(omm.metadata.time_system, CCSDSTimeSystem::UTC);
        assert_eq!(omm.metadata.mean_element_theory, "SGP4");

        // Mean elements
        assert!((omm.mean_elements.eccentricity - 0.0001).abs() < 1e-10);
        assert!((omm.mean_elements.inclination - 51.64).abs() < 1e-4);
        assert!((omm.mean_elements.ra_of_asc_node - 200.0).abs() < 1e-4);
        assert!((omm.mean_elements.arg_of_pericenter - 100.0).abs() < 1e-4);
        assert!((omm.mean_elements.mean_anomaly - 260.0).abs() < 1e-4);
        assert!((omm.mean_elements.mean_motion.unwrap() - 15.5).abs() < 1e-8);

        // TLE parameters
        let tle = omm.tle_parameters.as_ref().unwrap();
        assert_eq!(tle.ephemeris_type, Some(0));
        assert_eq!(tle.classification_type, Some('U'));
        assert_eq!(tle.norad_cat_id, Some(25544));
        assert_eq!(tle.element_set_no, Some(999));
        assert_eq!(tle.rev_at_epoch, Some(45000));
        assert!((tle.bstar.unwrap() - 0.000341).abs() < 1e-10);
        assert!((tle.mean_motion_dot.unwrap() - 0.00001).abs() < 1e-12);
        assert!((tle.mean_motion_ddot.unwrap()).abs() < 1e-15);
    }

    #[test]
    fn test_gp_record_to_omm_missing_required() {
        // Missing epoch
        let json = r#"{"ECCENTRICITY": 0.001, "INCLINATION": 51.64, "RA_OF_ASC_NODE": 200.0, "ARG_OF_PERICENTER": 100.0, "MEAN_ANOMALY": 260.0}"#;
        let record: GPRecord = serde_json::from_str(json).unwrap();
        assert!(record.to_omm().is_err());

        // Missing eccentricity
        let json = r#"{"EPOCH": "2024-01-15T12:00:00.000", "INCLINATION": 51.64, "RA_OF_ASC_NODE": 200.0, "ARG_OF_PERICENTER": 100.0, "MEAN_ANOMALY": 260.0}"#;
        let record: GPRecord = serde_json::from_str(json).unwrap();
        assert!(record.to_omm().is_err());
    }

    #[test]
    fn test_omm_to_gp_record() {
        let content = std::fs::read_to_string("test_assets/ccsds/omm/OMMExample1.txt").unwrap();
        let omm = OMM::from_str(&content).unwrap();

        let gp = omm.to_gp_record();
        assert_eq!(gp.object_name.as_deref(), Some("GOES 9"));
        assert_eq!(gp.object_id.as_deref(), Some("1995-025A"));
        assert_eq!(gp.center_name.as_deref(), Some("EARTH"));
        assert_eq!(gp.ref_frame.as_deref(), Some("TEME"));
        assert_eq!(gp.time_system.as_deref(), Some("UTC"));
        assert!((gp.eccentricity.unwrap() - 0.0005013).abs() < 1e-10);
        assert!((gp.inclination.unwrap() - 3.0539).abs() < 1e-4);
        assert_eq!(gp.norad_cat_id, Some(23581));
        assert_eq!(gp.classification_type.as_deref(), Some("U"));
        assert!((gp.bstar.unwrap() - 0.0001).abs() < 1e-10);
    }

    #[test]
    fn test_omm_gp_record_roundtrip() {
        let record: GPRecord = serde_json::from_str(sample_gp_record_json()).unwrap();

        // GPRecord -> OMM -> GPRecord
        let omm = record.to_omm().unwrap();
        let roundtripped = omm.to_gp_record();

        // Verify common fields are preserved
        assert_eq!(roundtripped.object_name, record.object_name);
        assert_eq!(roundtripped.object_id, record.object_id);
        assert_eq!(roundtripped.center_name, record.center_name);
        assert_eq!(roundtripped.ref_frame, record.ref_frame);
        assert_eq!(roundtripped.time_system, record.time_system);
        assert_eq!(roundtripped.mean_element_theory, record.mean_element_theory);

        // Numeric fields
        assert!((roundtripped.eccentricity.unwrap() - record.eccentricity.unwrap()).abs() < 1e-10);
        assert!((roundtripped.inclination.unwrap() - record.inclination.unwrap()).abs() < 1e-10);
        assert!(
            (roundtripped.ra_of_asc_node.unwrap() - record.ra_of_asc_node.unwrap()).abs() < 1e-10
        );
        assert!(
            (roundtripped.arg_of_pericenter.unwrap() - record.arg_of_pericenter.unwrap()).abs()
                < 1e-10
        );
        assert!((roundtripped.mean_anomaly.unwrap() - record.mean_anomaly.unwrap()).abs() < 1e-10);
        assert!((roundtripped.mean_motion.unwrap() - record.mean_motion.unwrap()).abs() < 1e-10);

        // TLE parameters
        assert_eq!(roundtripped.norad_cat_id, record.norad_cat_id);
        assert_eq!(roundtripped.classification_type, record.classification_type);
        assert_eq!(roundtripped.rev_at_epoch, record.rev_at_epoch);
        assert!((roundtripped.bstar.unwrap() - record.bstar.unwrap()).abs() < 1e-10);
    }
}
