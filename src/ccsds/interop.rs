/*!
 * Brahe interop for CCSDS types.
 *
 * Provides conversion between CCSDS message types and brahe's native
 * trajectory, propagator, and state vector types.
 */

use nalgebra::{DVector, SVector};

use crate::attitude::frames::{
    AttitudeFrame, OrbitRelativeFrame, OrbitRelativeKind, OrbitRelativeVariant, SpacecraftFrame,
};
use crate::ccsds::common::{
    CCSDSRefFrame, CCSDSTimeSystem, ODMHeader, format_ccsds_datetime_in, parse_ccsds_datetime,
};
use crate::ccsds::frames::{
    ADMReferenceFrame, CCSDSCelestialBodyFrame, CCSDSOrbitRelativeFrame, CCSDSSpacecraftBodyFrame,
};
use crate::ccsds::oem::OEM;
use crate::ccsds::omm::{OMM, OMMMetadata, OMMTleParameters, OMMeanElements};
use crate::frames::{CelestialFrame, DStateAdapter, ObjectId, ReferenceFrame, register_object};
use crate::time::Epoch;
use crate::trajectories::dorbit_trajectory::DOrbitTrajectory;
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
    /// Convert a single OEM segment to a `DOrbitTrajectory`.
    ///
    /// Returns a dynamic-dimension trajectory that implements
    /// `DIdentifiableStateProvider`, making it directly usable with the
    /// access computation API (`location_accesses`).
    ///
    /// # Arguments
    ///
    /// * `segment_idx` - Index of the segment to convert (0-based)
    ///
    /// # Returns
    ///
    /// * `Result<DOrbitTrajectory, BraheError>` - Trajectory or error
    pub fn segment_to_dorbit_trajectory(
        &self,
        segment_idx: usize,
    ) -> Result<DOrbitTrajectory, BraheError> {
        let segment = self.segments.get(segment_idx).ok_or_else(|| {
            BraheError::OutOfBoundsError(format!(
                "OEM segment index {} out of range (have {})",
                segment_idx,
                self.segments.len()
            ))
        })?;

        let orbit_frame = ccsds_ref_frame_to_orbit_frame(&segment.metadata.ref_frame)?;

        let mut traj = DOrbitTrajectory::new(6, orbit_frame, OrbitRepresentation::Cartesian, None)?;

        traj.name = Some(segment.metadata.object_name.clone());

        for sv in &segment.states {
            let state = DVector::from_column_slice(&[
                sv.position[0],
                sv.position[1],
                sv.position[2],
                sv.velocity[0],
                sv.velocity[1],
                sv.velocity[2],
            ]);
            traj.add(sv.epoch, state)?;
        }

        Ok(traj)
    }

    /// Convert a single OEM segment to an `SOrbitTrajectory`.
    ///
    /// Returns a static 6D trajectory optimized for orbital state vectors.
    /// Note: `SOrbitTrajectory` does not implement `DIdentifiableStateProvider`,
    /// so it cannot be used directly with `location_accesses`. Use
    /// `segment_to_dorbit_trajectory` for access computation.
    ///
    /// # Arguments
    ///
    /// * `segment_idx` - Index of the segment to convert (0-based)
    ///
    /// # Returns
    ///
    /// * `Result<SOrbitTrajectory, BraheError>` - Trajectory or error
    pub fn segment_to_sorbit_trajectory(
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

        let mut traj = SOrbitTrajectory::new(orbit_frame, OrbitRepresentation::Cartesian, None)?;

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
            traj.add(sv.epoch, state)?;
        }

        Ok(traj)
    }

    /// Convert a single OEM segment to a `DOrbitTrajectory`.
    ///
    /// Convenience alias for `segment_to_dorbit_trajectory`. Returns a trajectory
    /// compatible with the access computation API and other brahe functions.
    pub fn segment_to_trajectory(
        &self,
        segment_idx: usize,
    ) -> Result<DOrbitTrajectory, BraheError> {
        self.segment_to_dorbit_trajectory(segment_idx)
    }

    /// Convert all segments to `DOrbitTrajectory` objects.
    pub fn to_trajectories(&self) -> Result<Vec<DOrbitTrajectory>, BraheError> {
        (0..self.segments.len())
            .map(|i| self.segment_to_dorbit_trajectory(i))
            .collect()
    }
}

impl TryFrom<&OEM> for DOrbitTrajectory {
    type Error = BraheError;

    /// Convert a single-segment OEM to a `DOrbitTrajectory`.
    ///
    /// Returns an error if the OEM has zero or more than one segment.
    fn try_from(oem: &OEM) -> Result<Self, Self::Error> {
        if oem.segments.len() != 1 {
            return Err(BraheError::Error(format!(
                "TryFrom<&OEM> requires exactly 1 segment, but OEM has {}",
                oem.segments.len()
            )));
        }
        oem.segment_to_dorbit_trajectory(0)
    }
}

impl OEM {
    /// Registers this OEM as an object in the global object registry.
    ///
    /// Converts the OEM to a `DOrbitTrajectory` via `TryFrom<&OEM>`
    /// (erroring for a zero- or multi-segment OEM exactly as that
    /// conversion does) and registers it under `name` with the celestial
    /// frame carried by the converted trajectory. The registered object can then be queried through
    /// `object_state`, or used as the anchor for an orbit-relative frame
    /// such as `ReferenceFrame::RTN(name)`.
    ///
    /// # Arguments
    ///
    /// * `name` - The object identity to register the trajectory under
    ///
    /// # Returns
    ///
    /// * `Result<(), BraheError>` - `Ok(())` on success, or an error if the
    ///   OEM does not have exactly one segment, or its reference frame does
    ///   not map to a `CelestialFrame`
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use brahe::ccsds::oem::OEM;
    /// use brahe::frames::clear_object_registry;
    /// use std::str::FromStr;
    ///
    /// let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample5.txt").unwrap();
    /// let oem = OEM::from_str(&content).unwrap();
    ///
    /// clear_object_registry();
    /// oem.register_for("ISS").unwrap();
    /// clear_object_registry();
    /// ```
    pub fn register_for(&self, name: impl Into<ObjectId>) -> Result<(), BraheError> {
        let traj = DOrbitTrajectory::try_from(self)?;
        let frame = match traj.frame {
            OrbitFrame::GCRF => CelestialFrame::GCRF,
            OrbitFrame::ITRF => CelestialFrame::ITRF,
            OrbitFrame::EME2000 => CelestialFrame::EME2000,
            other => {
                return Err(BraheError::Error(format!(
                    "OEM::register_for cannot map OrbitFrame '{}' to a CelestialFrame",
                    other
                )));
            }
        };
        let adapter = DStateAdapter::new(traj)?;
        register_object(name, adapter, frame)
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
        let epoch_str =
            format_ccsds_datetime_in(&self.mean_elements.epoch, &self.metadata.time_system);

        GPRecord {
            ccsds_omm_vers: Some(format!("{:.1}", self.header.format_version)),
            comment: None,
            creation_date: Some(format_ccsds_datetime_in(
                &self.header.creation_date,
                &CCSDSTimeSystem::UTC,
            )),
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

impl TryFrom<&ADMReferenceFrame> for AttitudeFrame {
    type Error = BraheError;

    /// Converts a CCSDS ADM frame into a native [`AttitudeFrame`].
    ///
    /// Celestial frames map onto [`ReferenceFrame`] where brahe implements
    /// the frame (ICRF/GCRF → GCRF, EME2000/J2000 → EME2000, ITRF
    /// realizations → ITRF, MOON_PA → LFPA, MOON_ME → LFME). Orbit-relative
    /// and spacecraft frames map structurally. All other frames return an
    /// error; the containing message still loads and writes — only conversion
    /// to native types is unsupported.
    fn try_from(frame: &ADMReferenceFrame) -> Result<Self, Self::Error> {
        match frame {
            ADMReferenceFrame::Celestial(celestial) => {
                let reference = match celestial {
                    CCSDSCelestialBodyFrame::ICRF(_) | CCSDSCelestialBodyFrame::GCRF(_) => {
                        ReferenceFrame::GCRF
                    }
                    CCSDSCelestialBodyFrame::EME2000 | CCSDSCelestialBodyFrame::J2000 => {
                        ReferenceFrame::EME2000
                    }
                    CCSDSCelestialBodyFrame::ITRF(_) => ReferenceFrame::ITRF,
                    CCSDSCelestialBodyFrame::MoonPA(_) => ReferenceFrame::LFPA,
                    CCSDSCelestialBodyFrame::MoonME => ReferenceFrame::LFME,
                    other => {
                        return Err(BraheError::Error(format!(
                            "CCSDS celestial frame '{}' has no brahe ReferenceFrame equivalent; \
                             the message can still be read and written, but not converted to \
                             native attitude types",
                            other
                        )));
                    }
                };
                Ok(AttitudeFrame::Reference(reference))
            }
            ADMReferenceFrame::OrbitRelative(orbit_relative) => {
                let (kind, variant) = match orbit_relative {
                    CCSDSOrbitRelativeFrame::EQWInertial => {
                        (OrbitRelativeKind::EQW, OrbitRelativeVariant::Inertial)
                    }
                    CCSDSOrbitRelativeFrame::LVLHInertial => {
                        (OrbitRelativeKind::LVLH, OrbitRelativeVariant::Inertial)
                    }
                    CCSDSOrbitRelativeFrame::LVLHRotating => {
                        (OrbitRelativeKind::LVLH, OrbitRelativeVariant::Rotating)
                    }
                    CCSDSOrbitRelativeFrame::NSWInertial => {
                        (OrbitRelativeKind::NSW, OrbitRelativeVariant::Inertial)
                    }
                    CCSDSOrbitRelativeFrame::NSWRotating => {
                        (OrbitRelativeKind::NSW, OrbitRelativeVariant::Rotating)
                    }
                    CCSDSOrbitRelativeFrame::NTWInertial => {
                        (OrbitRelativeKind::NTW, OrbitRelativeVariant::Inertial)
                    }
                    CCSDSOrbitRelativeFrame::NTWRotating => {
                        (OrbitRelativeKind::NTW, OrbitRelativeVariant::Rotating)
                    }
                    CCSDSOrbitRelativeFrame::PQWInertial => {
                        (OrbitRelativeKind::PQW, OrbitRelativeVariant::Inertial)
                    }
                    CCSDSOrbitRelativeFrame::RSWInertial => {
                        (OrbitRelativeKind::RTN, OrbitRelativeVariant::Inertial)
                    }
                    CCSDSOrbitRelativeFrame::RSWRotating => {
                        (OrbitRelativeKind::RTN, OrbitRelativeVariant::Rotating)
                    }
                    CCSDSOrbitRelativeFrame::SEZInertial => {
                        (OrbitRelativeKind::SEZ, OrbitRelativeVariant::Inertial)
                    }
                    CCSDSOrbitRelativeFrame::SEZRotating => {
                        (OrbitRelativeKind::SEZ, OrbitRelativeVariant::Rotating)
                    }
                    CCSDSOrbitRelativeFrame::TNWInertial => {
                        (OrbitRelativeKind::TNW, OrbitRelativeVariant::Inertial)
                    }
                    CCSDSOrbitRelativeFrame::TNWRotating => {
                        (OrbitRelativeKind::TNW, OrbitRelativeVariant::Rotating)
                    }
                    CCSDSOrbitRelativeFrame::VNCInertial => {
                        (OrbitRelativeKind::VNC, OrbitRelativeVariant::Inertial)
                    }
                    CCSDSOrbitRelativeFrame::VNCRotating => {
                        (OrbitRelativeKind::VNC, OrbitRelativeVariant::Rotating)
                    }
                    CCSDSOrbitRelativeFrame::Other(token) => {
                        return Err(BraheError::Error(format!(
                            "CCSDS orbit-relative frame '{}' is not a SANA registry frame and \
                             cannot be converted to a native attitude frame",
                            token
                        )));
                    }
                };
                Ok(AttitudeFrame::OrbitRelative(OrbitRelativeFrame {
                    kind,
                    variant,
                }))
            }
            ADMReferenceFrame::Spacecraft(spacecraft) => {
                let native = match spacecraft {
                    CCSDSSpacecraftBodyFrame::ACC(i) => SpacecraftFrame::ACC(i.clone()),
                    CCSDSSpacecraftBodyFrame::Actuator(i) => SpacecraftFrame::Actuator(i.clone()),
                    CCSDSSpacecraftBodyFrame::AST(i) => SpacecraftFrame::AST(i.clone()),
                    CCSDSSpacecraftBodyFrame::CSS(i) => SpacecraftFrame::CSS(i.clone()),
                    CCSDSSpacecraftBodyFrame::DSS(i) => SpacecraftFrame::DSS(i.clone()),
                    CCSDSSpacecraftBodyFrame::ESA(i) => SpacecraftFrame::ESA(i.clone()),
                    CCSDSSpacecraftBodyFrame::GyroFrame(i) => SpacecraftFrame::GyroFrame(i.clone()),
                    CCSDSSpacecraftBodyFrame::IMUFrame(i) => SpacecraftFrame::IMUFrame(i.clone()),
                    CCSDSSpacecraftBodyFrame::Instrument(i) => {
                        SpacecraftFrame::Instrument(i.clone())
                    }
                    CCSDSSpacecraftBodyFrame::MTA(i) => SpacecraftFrame::MTA(i.clone()),
                    CCSDSSpacecraftBodyFrame::RW(i) => SpacecraftFrame::RW(i.clone()),
                    CCSDSSpacecraftBodyFrame::SA(i) => SpacecraftFrame::SA(i.clone()),
                    CCSDSSpacecraftBodyFrame::SCBody(i) => SpacecraftFrame::SCBody(i.clone()),
                    CCSDSSpacecraftBodyFrame::Sensor(i) => SpacecraftFrame::Sensor(i.clone()),
                    CCSDSSpacecraftBodyFrame::StarTracker(i) => {
                        SpacecraftFrame::StarTracker(i.clone())
                    }
                    CCSDSSpacecraftBodyFrame::TAM(i) => SpacecraftFrame::TAM(i.clone()),
                    CCSDSSpacecraftBodyFrame::Other(token) => {
                        return Err(BraheError::Error(format!(
                            "CCSDS spacecraft frame '{}' is not a SANA registry frame and \
                             cannot be converted to a native attitude frame",
                            token
                        )));
                    }
                };
                Ok(AttitudeFrame::Spacecraft(native))
            }
            ADMReferenceFrame::Other(token) => Err(BraheError::Error(format!(
                "CCSDS frame '{}' is in none of the SANA ADM frame registries and cannot be \
                 converted to a native attitude frame",
                token
            ))),
        }
    }
}

impl TryFrom<&AttitudeFrame> for ADMReferenceFrame {
    type Error = BraheError;

    /// Converts a native [`AttitudeFrame`] into a CCSDS ADM frame token for
    /// writing messages. `ReferenceFrame` variants without a SANA celestial
    /// token (synodic, body-centered generic, Mars/lunar inertial, ...)
    /// return an error.
    fn try_from(frame: &AttitudeFrame) -> Result<Self, Self::Error> {
        match frame {
            AttitudeFrame::Reference(reference) => {
                let celestial = match reference {
                    ReferenceFrame::GCRF => CCSDSCelestialBodyFrame::GCRF(None),
                    ReferenceFrame::EME2000 => CCSDSCelestialBodyFrame::EME2000,
                    ReferenceFrame::ITRF => CCSDSCelestialBodyFrame::ITRF(None),
                    ReferenceFrame::LFPA => CCSDSCelestialBodyFrame::MoonPA(None),
                    ReferenceFrame::LFME => CCSDSCelestialBodyFrame::MoonME,
                    other => {
                        return Err(BraheError::Error(format!(
                            "brahe frame '{:?}' has no SANA celestial-body frame token and \
                             cannot be written into an ADM message",
                            other
                        )));
                    }
                };
                Ok(ADMReferenceFrame::Celestial(celestial))
            }
            AttitudeFrame::OrbitRelative(orbit_relative) => {
                use OrbitRelativeKind as K;
                use OrbitRelativeVariant as V;
                let ccsds = match (orbit_relative.kind, orbit_relative.variant) {
                    (K::EQW, V::Inertial) => CCSDSOrbitRelativeFrame::EQWInertial,
                    (K::LVLH, V::Inertial) => CCSDSOrbitRelativeFrame::LVLHInertial,
                    (K::LVLH, V::Rotating) => CCSDSOrbitRelativeFrame::LVLHRotating,
                    (K::NSW, V::Inertial) => CCSDSOrbitRelativeFrame::NSWInertial,
                    (K::NSW, V::Rotating) => CCSDSOrbitRelativeFrame::NSWRotating,
                    (K::NTW, V::Inertial) => CCSDSOrbitRelativeFrame::NTWInertial,
                    (K::NTW, V::Rotating) => CCSDSOrbitRelativeFrame::NTWRotating,
                    (K::PQW, V::Inertial) => CCSDSOrbitRelativeFrame::PQWInertial,
                    (K::RTN, V::Inertial) => CCSDSOrbitRelativeFrame::RSWInertial,
                    (K::RTN, V::Rotating) => CCSDSOrbitRelativeFrame::RSWRotating,
                    (K::SEZ, V::Inertial) => CCSDSOrbitRelativeFrame::SEZInertial,
                    (K::SEZ, V::Rotating) => CCSDSOrbitRelativeFrame::SEZRotating,
                    (K::TNW, V::Inertial) => CCSDSOrbitRelativeFrame::TNWInertial,
                    (K::TNW, V::Rotating) => CCSDSOrbitRelativeFrame::TNWRotating,
                    (K::VNC, V::Inertial) => CCSDSOrbitRelativeFrame::VNCInertial,
                    (K::VNC, V::Rotating) => CCSDSOrbitRelativeFrame::VNCRotating,
                    (kind, V::Rotating) if matches!(kind, K::EQW | K::PQW) => {
                        return Err(BraheError::Error(format!(
                            "orbit-relative frame {:?} exists only as an inertial SANA frame",
                            kind
                        )));
                    }
                    _ => unreachable!(),
                };
                Ok(ADMReferenceFrame::OrbitRelative(ccsds))
            }
            AttitudeFrame::Spacecraft(spacecraft) => {
                let ccsds = match spacecraft {
                    SpacecraftFrame::ACC(i) => CCSDSSpacecraftBodyFrame::ACC(i.clone()),
                    SpacecraftFrame::Actuator(i) => CCSDSSpacecraftBodyFrame::Actuator(i.clone()),
                    SpacecraftFrame::AST(i) => CCSDSSpacecraftBodyFrame::AST(i.clone()),
                    SpacecraftFrame::CSS(i) => CCSDSSpacecraftBodyFrame::CSS(i.clone()),
                    SpacecraftFrame::DSS(i) => CCSDSSpacecraftBodyFrame::DSS(i.clone()),
                    SpacecraftFrame::ESA(i) => CCSDSSpacecraftBodyFrame::ESA(i.clone()),
                    SpacecraftFrame::GyroFrame(i) => CCSDSSpacecraftBodyFrame::GyroFrame(i.clone()),
                    SpacecraftFrame::IMUFrame(i) => CCSDSSpacecraftBodyFrame::IMUFrame(i.clone()),
                    SpacecraftFrame::Instrument(i) => {
                        CCSDSSpacecraftBodyFrame::Instrument(i.clone())
                    }
                    SpacecraftFrame::MTA(i) => CCSDSSpacecraftBodyFrame::MTA(i.clone()),
                    SpacecraftFrame::RW(i) => CCSDSSpacecraftBodyFrame::RW(i.clone()),
                    SpacecraftFrame::SA(i) => CCSDSSpacecraftBodyFrame::SA(i.clone()),
                    SpacecraftFrame::SCBody(i) => CCSDSSpacecraftBodyFrame::SCBody(i.clone()),
                    SpacecraftFrame::Sensor(i) => CCSDSSpacecraftBodyFrame::Sensor(i.clone()),
                    SpacecraftFrame::StarTracker(i) => {
                        CCSDSSpacecraftBodyFrame::StarTracker(i.clone())
                    }
                    SpacecraftFrame::TAM(i) => CCSDSSpacecraftBodyFrame::TAM(i.clone()),
                };
                Ok(ADMReferenceFrame::Spacecraft(ccsds))
            }
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use serial_test::{parallel, serial};

    use super::*;
    use crate::ccsds::oem::OEM;
    use crate::frames::{
        CelestialFrame, ReferenceFrame, clear_object_registry, object_state,
        rotation_frame_to_frame,
    };
    use crate::trajectories::traits::{InterpolatableTrajectory, Trajectory};

    #[test]
    #[parallel]
    fn test_oem_to_trajectory_example4() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample4.txt").unwrap();
        let oem = OEM::from_str(&content).unwrap();

        let traj = oem.segment_to_trajectory(0).unwrap();
        assert_eq!(traj.len(), 3);
        assert_eq!(traj.name.as_deref(), Some("MARS GLOBAL SURVEYOR"));
        assert_eq!(traj.frame, OrbitFrame::EME2000);

        // Verify first state
        let (_epoch, state) = traj.first().unwrap();
        assert!((state[0] - 2789.619 * 1000.0).abs() < 1.0);
        assert!((state[3] - 4.73372 * 1000.0).abs() < 1.0);
    }

    #[test]
    #[parallel]
    fn test_oem_to_trajectory_example5() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample5.txt").unwrap();
        let oem = OEM::from_str(&content).unwrap();

        let traj = oem.segment_to_trajectory(0).unwrap();
        assert_eq!(traj.len(), 49);
        assert_eq!(traj.frame, OrbitFrame::GCRF);
    }

    #[test]
    #[parallel]
    fn test_oem_to_trajectories_multi_segment() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let oem = OEM::from_str(&content).unwrap();

        let trajs = oem.to_trajectories().unwrap();
        assert_eq!(trajs.len(), 3);
    }

    #[test]
    #[parallel]
    fn test_oem_try_from_single_segment() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample4.txt").unwrap();
        let oem = OEM::from_str(&content).unwrap();

        let traj = DOrbitTrajectory::try_from(&oem).unwrap();
        assert_eq!(traj.len(), 3);
    }

    #[test]
    #[parallel]
    fn test_oem_try_from_multi_segment_fails() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let oem = OEM::from_str(&content).unwrap();

        assert!(DOrbitTrajectory::try_from(&oem).is_err());
    }

    #[test]
    #[serial]
    fn test_oem_register_for() {
        clear_object_registry();

        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample5.txt").unwrap();
        let oem = OEM::from_str(&content).unwrap();

        oem.register_for("LEO").unwrap();

        let traj = DOrbitTrajectory::try_from(&oem).unwrap();
        let epoch = traj.first().unwrap().0 + 300.0;

        let r = rotation_frame_to_frame(CelestialFrame::GCRF, ReferenceFrame::RTN("LEO"), epoch);
        assert!(r.is_ok());

        let (frame, state) = object_state(&"LEO".into(), epoch).unwrap();
        assert_eq!(frame, CelestialFrame::GCRF);
        let expected = traj.interpolate(&epoch).unwrap();
        for i in 0..6 {
            assert_eq!(state[i], expected[i]);
        }

        clear_object_registry();
    }

    #[test]
    #[serial]
    fn test_oem_register_for_itrf() {
        // Covers the ITRF arm of register_for's OrbitFrame -> CelestialFrame
        // mapping (test_oem_register_for above only exercises GCRF). The
        // asset is a short single-segment ITRF2014 OEM, which is all
        // register_for needs.
        clear_object_registry();

        let content =
            std::fs::read_to_string("test_assets/ccsds/oem/OEM-single-segment-itrf.txt").unwrap();
        let oem = OEM::from_str(&content).unwrap();

        oem.register_for("ISS_ITRF").unwrap();

        let traj = DOrbitTrajectory::try_from(&oem).unwrap();
        assert_eq!(traj.frame, OrbitFrame::ITRF);
        let epoch = traj.first().unwrap().0 + 300.0;

        let (frame, state) = object_state(&"ISS_ITRF".into(), epoch).unwrap();
        assert_eq!(frame, CelestialFrame::ITRF);
        let expected = traj.interpolate(&epoch).unwrap();
        for i in 0..6 {
            assert_eq!(state[i], expected[i]);
        }

        clear_object_registry();
    }

    #[test]
    #[parallel]
    fn test_oem_segment_out_of_bounds() {
        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample4.txt").unwrap();
        let oem = OEM::from_str(&content).unwrap();

        assert!(oem.segment_to_trajectory(5).is_err());
    }

    #[test]
    #[parallel]
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
    #[parallel]
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
    #[parallel]
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
    #[parallel]
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
    #[parallel]
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

    #[test]
    #[parallel]
    fn test_adm_frame_to_attitude_frame_celestial() {
        let cases = [
            ("ICRF", ReferenceFrame::GCRF),
            ("GCRF", ReferenceFrame::GCRF),
            ("GCRF2", ReferenceFrame::GCRF),
            ("EME2000", ReferenceFrame::EME2000),
            ("J2000", ReferenceFrame::EME2000),
            ("ITRF", ReferenceFrame::ITRF),
            ("ITRF2014", ReferenceFrame::ITRF),
            ("MOON_PA", ReferenceFrame::LFPA),
            ("MOON_PA440", ReferenceFrame::LFPA),
            ("MOON_ME", ReferenceFrame::LFME),
        ];
        for (token, expected) in cases {
            let adm = ADMReferenceFrame::parse(token);
            let att = AttitudeFrame::try_from(&adm).unwrap();
            assert_eq!(att, AttitudeFrame::Reference(expected), "token {}", token);
        }
    }

    #[test]
    #[parallel]
    fn test_adm_frame_to_attitude_frame_unsupported() {
        for token in ["TOD_EARTH", "B1950", "WGS84", "TEMEOFDATE", "BODY_FRAME_A"] {
            let adm = ADMReferenceFrame::parse(token);
            let err = AttitudeFrame::try_from(&adm).unwrap_err();
            assert!(
                err.to_string().contains(token),
                "error should name {}",
                token
            );
        }
    }

    #[test]
    #[parallel]
    fn test_adm_frame_to_attitude_frame_structural() {
        let adm = ADMReferenceFrame::parse("RSW_ROTATING");
        assert_eq!(
            AttitudeFrame::try_from(&adm).unwrap(),
            AttitudeFrame::OrbitRelative(OrbitRelativeFrame {
                kind: OrbitRelativeKind::RTN,
                variant: OrbitRelativeVariant::Rotating,
            })
        );
        let adm = ADMReferenceFrame::parse("INSTRUMENT_A");
        assert_eq!(
            AttitudeFrame::try_from(&adm).unwrap(),
            AttitudeFrame::Spacecraft(SpacecraftFrame::Instrument(Some("A".to_string())))
        );
    }

    #[test]
    #[parallel]
    fn test_attitude_frame_to_adm_frame_roundtrip() {
        // Every mappable native frame must round-trip through ADM tokens
        let frames = [
            AttitudeFrame::Reference(ReferenceFrame::GCRF),
            AttitudeFrame::Reference(ReferenceFrame::EME2000),
            AttitudeFrame::Reference(ReferenceFrame::ITRF),
            AttitudeFrame::Reference(ReferenceFrame::LFPA),
            AttitudeFrame::Reference(ReferenceFrame::LFME),
            AttitudeFrame::OrbitRelative(OrbitRelativeFrame {
                kind: OrbitRelativeKind::LVLH,
                variant: OrbitRelativeVariant::Rotating,
            }),
            AttitudeFrame::Spacecraft(SpacecraftFrame::SCBody(Some("1".to_string()))),
        ];
        for frame in frames {
            let adm = ADMReferenceFrame::try_from(&frame).unwrap();
            let back = AttitudeFrame::try_from(&adm).unwrap();
            assert_eq!(back, frame);
        }
    }

    #[test]
    #[parallel]
    fn test_attitude_frame_to_adm_frame_unsupported() {
        let frame = AttitudeFrame::Reference(ReferenceFrame::EMR);
        assert!(ADMReferenceFrame::try_from(&frame).is_err());
    }

    #[test]
    #[parallel]
    fn test_adm_orbit_relative_frame_to_attitude_frame_all_kinds() {
        let cases = [
            (
                CCSDSOrbitRelativeFrame::EQWInertial,
                OrbitRelativeKind::EQW,
                OrbitRelativeVariant::Inertial,
            ),
            (
                CCSDSOrbitRelativeFrame::LVLHInertial,
                OrbitRelativeKind::LVLH,
                OrbitRelativeVariant::Inertial,
            ),
            (
                CCSDSOrbitRelativeFrame::LVLHRotating,
                OrbitRelativeKind::LVLH,
                OrbitRelativeVariant::Rotating,
            ),
            (
                CCSDSOrbitRelativeFrame::NSWInertial,
                OrbitRelativeKind::NSW,
                OrbitRelativeVariant::Inertial,
            ),
            (
                CCSDSOrbitRelativeFrame::NSWRotating,
                OrbitRelativeKind::NSW,
                OrbitRelativeVariant::Rotating,
            ),
            (
                CCSDSOrbitRelativeFrame::NTWInertial,
                OrbitRelativeKind::NTW,
                OrbitRelativeVariant::Inertial,
            ),
            (
                CCSDSOrbitRelativeFrame::NTWRotating,
                OrbitRelativeKind::NTW,
                OrbitRelativeVariant::Rotating,
            ),
            (
                CCSDSOrbitRelativeFrame::PQWInertial,
                OrbitRelativeKind::PQW,
                OrbitRelativeVariant::Inertial,
            ),
            (
                CCSDSOrbitRelativeFrame::RSWInertial,
                OrbitRelativeKind::RTN,
                OrbitRelativeVariant::Inertial,
            ),
            (
                CCSDSOrbitRelativeFrame::RSWRotating,
                OrbitRelativeKind::RTN,
                OrbitRelativeVariant::Rotating,
            ),
            (
                CCSDSOrbitRelativeFrame::SEZInertial,
                OrbitRelativeKind::SEZ,
                OrbitRelativeVariant::Inertial,
            ),
            (
                CCSDSOrbitRelativeFrame::SEZRotating,
                OrbitRelativeKind::SEZ,
                OrbitRelativeVariant::Rotating,
            ),
            (
                CCSDSOrbitRelativeFrame::TNWInertial,
                OrbitRelativeKind::TNW,
                OrbitRelativeVariant::Inertial,
            ),
            (
                CCSDSOrbitRelativeFrame::TNWRotating,
                OrbitRelativeKind::TNW,
                OrbitRelativeVariant::Rotating,
            ),
            (
                CCSDSOrbitRelativeFrame::VNCInertial,
                OrbitRelativeKind::VNC,
                OrbitRelativeVariant::Inertial,
            ),
            (
                CCSDSOrbitRelativeFrame::VNCRotating,
                OrbitRelativeKind::VNC,
                OrbitRelativeVariant::Rotating,
            ),
        ];
        for (ccsds, kind, variant) in cases {
            let adm = ADMReferenceFrame::OrbitRelative(ccsds.clone());
            let att = AttitudeFrame::try_from(&adm).unwrap();
            assert_eq!(
                att,
                AttitudeFrame::OrbitRelative(OrbitRelativeFrame { kind, variant }),
                "ccsds frame {:?}",
                ccsds
            );
        }
    }

    #[test]
    #[parallel]
    fn test_attitude_orbit_relative_frame_to_adm_frame_all_kinds() {
        let cases = [
            (
                OrbitRelativeKind::EQW,
                OrbitRelativeVariant::Inertial,
                CCSDSOrbitRelativeFrame::EQWInertial,
            ),
            (
                OrbitRelativeKind::LVLH,
                OrbitRelativeVariant::Inertial,
                CCSDSOrbitRelativeFrame::LVLHInertial,
            ),
            (
                OrbitRelativeKind::LVLH,
                OrbitRelativeVariant::Rotating,
                CCSDSOrbitRelativeFrame::LVLHRotating,
            ),
            (
                OrbitRelativeKind::NSW,
                OrbitRelativeVariant::Inertial,
                CCSDSOrbitRelativeFrame::NSWInertial,
            ),
            (
                OrbitRelativeKind::NSW,
                OrbitRelativeVariant::Rotating,
                CCSDSOrbitRelativeFrame::NSWRotating,
            ),
            (
                OrbitRelativeKind::NTW,
                OrbitRelativeVariant::Inertial,
                CCSDSOrbitRelativeFrame::NTWInertial,
            ),
            (
                OrbitRelativeKind::NTW,
                OrbitRelativeVariant::Rotating,
                CCSDSOrbitRelativeFrame::NTWRotating,
            ),
            (
                OrbitRelativeKind::PQW,
                OrbitRelativeVariant::Inertial,
                CCSDSOrbitRelativeFrame::PQWInertial,
            ),
            (
                OrbitRelativeKind::RTN,
                OrbitRelativeVariant::Inertial,
                CCSDSOrbitRelativeFrame::RSWInertial,
            ),
            (
                OrbitRelativeKind::RTN,
                OrbitRelativeVariant::Rotating,
                CCSDSOrbitRelativeFrame::RSWRotating,
            ),
            (
                OrbitRelativeKind::SEZ,
                OrbitRelativeVariant::Inertial,
                CCSDSOrbitRelativeFrame::SEZInertial,
            ),
            (
                OrbitRelativeKind::SEZ,
                OrbitRelativeVariant::Rotating,
                CCSDSOrbitRelativeFrame::SEZRotating,
            ),
            (
                OrbitRelativeKind::TNW,
                OrbitRelativeVariant::Inertial,
                CCSDSOrbitRelativeFrame::TNWInertial,
            ),
            (
                OrbitRelativeKind::TNW,
                OrbitRelativeVariant::Rotating,
                CCSDSOrbitRelativeFrame::TNWRotating,
            ),
            (
                OrbitRelativeKind::VNC,
                OrbitRelativeVariant::Inertial,
                CCSDSOrbitRelativeFrame::VNCInertial,
            ),
            (
                OrbitRelativeKind::VNC,
                OrbitRelativeVariant::Rotating,
                CCSDSOrbitRelativeFrame::VNCRotating,
            ),
        ];
        for (kind, variant, expected) in cases {
            let frame = AttitudeFrame::OrbitRelative(OrbitRelativeFrame { kind, variant });
            let adm = ADMReferenceFrame::try_from(&frame).unwrap();
            assert_eq!(adm, ADMReferenceFrame::OrbitRelative(expected));
        }
    }

    #[test]
    #[parallel]
    fn test_attitude_orbit_relative_frame_to_adm_frame_inertial_only_errors() {
        // EQW and PQW exist only as inertial SANA frames.
        for kind in [OrbitRelativeKind::EQW, OrbitRelativeKind::PQW] {
            let frame = AttitudeFrame::OrbitRelative(OrbitRelativeFrame {
                kind,
                variant: OrbitRelativeVariant::Rotating,
            });
            assert!(ADMReferenceFrame::try_from(&frame).is_err());
        }
    }

    #[test]
    #[parallel]
    fn test_adm_spacecraft_frame_to_attitude_frame_all_families() {
        let cases = [
            (
                CCSDSSpacecraftBodyFrame::ACC(Some("1".to_string())),
                SpacecraftFrame::ACC(Some("1".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::Actuator(None),
                SpacecraftFrame::Actuator(None),
            ),
            (
                CCSDSSpacecraftBodyFrame::AST(Some("1".to_string())),
                SpacecraftFrame::AST(Some("1".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::CSS(Some("2".to_string())),
                SpacecraftFrame::CSS(Some("2".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::DSS(Some("1".to_string())),
                SpacecraftFrame::DSS(Some("1".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::ESA(Some("1".to_string())),
                SpacecraftFrame::ESA(Some("1".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::GyroFrame(Some("1".to_string())),
                SpacecraftFrame::GyroFrame(Some("1".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::IMUFrame(Some("2".to_string())),
                SpacecraftFrame::IMUFrame(Some("2".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::Instrument(Some("A".to_string())),
                SpacecraftFrame::Instrument(Some("A".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::MTA(Some("1".to_string())),
                SpacecraftFrame::MTA(Some("1".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::RW(Some("4".to_string())),
                SpacecraftFrame::RW(Some("4".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::SA(Some("1".to_string())),
                SpacecraftFrame::SA(Some("1".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::SCBody(None),
                SpacecraftFrame::SCBody(None),
            ),
            (
                CCSDSSpacecraftBodyFrame::Sensor(Some("10".to_string())),
                SpacecraftFrame::Sensor(Some("10".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::StarTracker(Some("2".to_string())),
                SpacecraftFrame::StarTracker(Some("2".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::TAM(Some("1".to_string())),
                SpacecraftFrame::TAM(Some("1".to_string())),
            ),
        ];
        for (ccsds, expected) in cases {
            let adm = ADMReferenceFrame::Spacecraft(ccsds.clone());
            let att = AttitudeFrame::try_from(&adm).unwrap();
            assert_eq!(
                att,
                AttitudeFrame::Spacecraft(expected),
                "ccsds frame {:?}",
                ccsds
            );
        }
    }

    #[test]
    #[parallel]
    fn test_attitude_spacecraft_frame_to_adm_frame_all_families() {
        let cases = [
            (
                SpacecraftFrame::ACC(Some("1".to_string())),
                CCSDSSpacecraftBodyFrame::ACC(Some("1".to_string())),
            ),
            (
                SpacecraftFrame::Actuator(None),
                CCSDSSpacecraftBodyFrame::Actuator(None),
            ),
            (
                SpacecraftFrame::AST(Some("1".to_string())),
                CCSDSSpacecraftBodyFrame::AST(Some("1".to_string())),
            ),
            (
                SpacecraftFrame::CSS(Some("2".to_string())),
                CCSDSSpacecraftBodyFrame::CSS(Some("2".to_string())),
            ),
            (
                SpacecraftFrame::DSS(Some("1".to_string())),
                CCSDSSpacecraftBodyFrame::DSS(Some("1".to_string())),
            ),
            (
                SpacecraftFrame::ESA(Some("1".to_string())),
                CCSDSSpacecraftBodyFrame::ESA(Some("1".to_string())),
            ),
            (
                SpacecraftFrame::GyroFrame(Some("1".to_string())),
                CCSDSSpacecraftBodyFrame::GyroFrame(Some("1".to_string())),
            ),
            (
                SpacecraftFrame::IMUFrame(Some("2".to_string())),
                CCSDSSpacecraftBodyFrame::IMUFrame(Some("2".to_string())),
            ),
            (
                SpacecraftFrame::Instrument(Some("A".to_string())),
                CCSDSSpacecraftBodyFrame::Instrument(Some("A".to_string())),
            ),
            (
                SpacecraftFrame::MTA(Some("1".to_string())),
                CCSDSSpacecraftBodyFrame::MTA(Some("1".to_string())),
            ),
            (
                SpacecraftFrame::RW(Some("4".to_string())),
                CCSDSSpacecraftBodyFrame::RW(Some("4".to_string())),
            ),
            (
                SpacecraftFrame::SA(Some("1".to_string())),
                CCSDSSpacecraftBodyFrame::SA(Some("1".to_string())),
            ),
            (
                SpacecraftFrame::SCBody(None),
                CCSDSSpacecraftBodyFrame::SCBody(None),
            ),
            (
                SpacecraftFrame::Sensor(Some("10".to_string())),
                CCSDSSpacecraftBodyFrame::Sensor(Some("10".to_string())),
            ),
            (
                SpacecraftFrame::StarTracker(Some("2".to_string())),
                CCSDSSpacecraftBodyFrame::StarTracker(Some("2".to_string())),
            ),
            (
                SpacecraftFrame::TAM(Some("1".to_string())),
                CCSDSSpacecraftBodyFrame::TAM(Some("1".to_string())),
            ),
        ];
        for (native, expected) in cases {
            let frame = AttitudeFrame::Spacecraft(native);
            let adm = ADMReferenceFrame::try_from(&frame).unwrap();
            assert_eq!(adm, ADMReferenceFrame::Spacecraft(expected));
        }
    }

    #[test]
    #[parallel]
    fn test_adm_orbit_relative_other_to_attitude_frame_errors() {
        let adm = ADMReferenceFrame::OrbitRelative(CCSDSOrbitRelativeFrame::Other(
            "CUSTOM_ORBIT_FRAME".to_string(),
        ));
        let err = AttitudeFrame::try_from(&adm).unwrap_err();
        assert!(err.to_string().contains("CUSTOM_ORBIT_FRAME"));
    }

    #[test]
    #[parallel]
    fn test_adm_spacecraft_other_to_attitude_frame_errors() {
        let adm = ADMReferenceFrame::Spacecraft(CCSDSSpacecraftBodyFrame::Other(
            "CUSTOM_SC_FRAME".to_string(),
        ));
        let err = AttitudeFrame::try_from(&adm).unwrap_err();
        assert!(err.to_string().contains("CUSTOM_SC_FRAME"));
    }
}
