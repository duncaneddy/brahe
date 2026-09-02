/*!
 * Brahe interop for CCSDS types.
 *
 * Provides conversion between CCSDS message types and brahe's native
 * trajectory, propagator, and state vector types.
 */

use nalgebra::{DVector, SVector, Vector3};

use crate::attitude::{
    FromAttitude, Quaternion, ToAttitude, angular_velocity_from_quaternion_derivative,
    euler_rates_to_angular_velocity,
};
use crate::ccsds::aem::{
    AEM, AEMAttitudeData, AEMAttitudeState, AEMAttitudeType, AEMInterpolationMethod, AEMMetadata,
    AEMSegment,
};
use crate::ccsds::common::{
    CCSDSRefFrame, CCSDSTimeSystem, ODMHeader, format_ccsds_datetime_in, parse_ccsds_datetime,
};
use crate::ccsds::frames::{
    ADMReferenceFrame, CCSDSCelestialBodyFrame, CCSDSOrbitRelativeFrame, CCSDSSpacecraftBodyFrame,
};
use crate::ccsds::oem::OEM;
use crate::ccsds::omm::{OMM, OMMMetadata, OMMTleParameters, OMMeanElements};
use crate::frames::{
    BodyFrame, CelestialFrame, DStateAdapter, ObjectId, OrbitRelativeFrameKind,
    OrbitRelativeFrameVariant, ReferenceFrame, register_frame, register_object,
};
use crate::time::Epoch;
use crate::trajectories::dorbit_trajectory::DOrbitTrajectory;
use crate::trajectories::sorbit_trajectory::SOrbitTrajectory;
use crate::trajectories::traits::{OrbitFrame, OrbitRepresentation, Trajectory};
use crate::trajectories::{AttitudeInterpolationMethod, AttitudeState, AttitudeTrajectory};
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

/// Converts AEM `/ANGVEL` attitude data into a canonical body-frame (frame
/// B) angular velocity.
///
/// The AEM ANGVEL types express angular velocity in the segment's
/// `ANGVEL_FRAME`, which `AEMMetadata::validate` guarantees equals either
/// `ref_frame_a` or `ref_frame_b`. If it is `ref_frame_b`, the value is
/// already the canonical body-frame rate and is returned unchanged. If it is
/// `ref_frame_a`, it must be re-expressed in frame B: Diebel (2006) eq. 41
/// gives the frame-A-to-frame-B relation `ω' = R ω` for a vector expressed
/// in frame A rotated into frame B by the direction cosine matrix `R`, so
/// `ω_B = R(q) · ω_A` with `R(q)` the DCM of the state's attitude
/// quaternion.
fn aem_angvel_to_canonical(
    angular_velocity: Vector3<f64>,
    quaternion: &Quaternion,
    metadata: &AEMMetadata,
) -> Result<Vector3<f64>, BraheError> {
    let angvel_frame = metadata.angvel_frame.as_ref().ok_or_else(|| {
        BraheError::Error(
            "AEM metadata ANGVEL_FRAME is required for '/ANGVEL' ATTITUDE_TYPE values".to_string(),
        )
    })?;

    if *angvel_frame == metadata.ref_frame_a {
        let r = quaternion.to_rotation_matrix().to_matrix();
        Ok(r * angular_velocity)
    } else {
        Ok(angular_velocity)
    }
}

/// Builds the "message can still be read and written, but not converted"
/// error for the SPIN* AEM attitude types: spin-parameterized attitude
/// types have no native `AttitudeTrajectory` mapping, though the message
/// can still be read and written.
fn aem_spin_conversion_error(attitude_type: AEMAttitudeType) -> BraheError {
    BraheError::Error(format!(
        "AEM attitude type '{}' has no AttitudeTrajectory representation; the message can still \
         be read and written, but not converted to a native trajectory",
        attitude_type
    ))
}

/// Converts one AEM ephemeris line's attitude data into a canonical
/// [`AttitudeState`] (quaternion, frame A to frame B, plus optional
/// body-frame angular velocity).
fn aem_attitude_data_to_state(
    data: &AEMAttitudeData,
    metadata: &AEMMetadata,
) -> Result<AttitudeState, BraheError> {
    match data {
        AEMAttitudeData::Quaternion { quaternion } => Ok(AttitudeState::new(*quaternion)),
        AEMAttitudeData::QuaternionDerivative {
            quaternion,
            derivative,
        } => {
            let omega = angular_velocity_from_quaternion_derivative(quaternion, *derivative);
            Ok(AttitudeState::new(*quaternion).with_angular_velocity(omega))
        }
        AEMAttitudeData::QuaternionAngVel {
            quaternion,
            angular_velocity,
        } => {
            let omega = aem_angvel_to_canonical(*angular_velocity, quaternion, metadata)?;
            Ok(AttitudeState::new(*quaternion).with_angular_velocity(omega))
        }
        AEMAttitudeData::EulerAngle { angles } => {
            Ok(AttitudeState::new(Quaternion::from_euler_angle(*angles)))
        }
        AEMAttitudeData::EulerAngleDerivative { angles, rates } => {
            let quaternion = Quaternion::from_euler_angle(*angles);
            let omega = euler_rates_to_angular_velocity(angles, *rates);
            Ok(AttitudeState::new(quaternion).with_angular_velocity(omega))
        }
        AEMAttitudeData::EulerAngleAngVel {
            angles,
            angular_velocity,
        } => {
            let quaternion = Quaternion::from_euler_angle(*angles);
            let omega = aem_angvel_to_canonical(*angular_velocity, &quaternion, metadata)?;
            Ok(AttitudeState::new(quaternion).with_angular_velocity(omega))
        }
        AEMAttitudeData::Spin { .. } => Err(aem_spin_conversion_error(AEMAttitudeType::Spin)),
        AEMAttitudeData::SpinNutation { .. } => {
            Err(aem_spin_conversion_error(AEMAttitudeType::SpinNutation))
        }
        AEMAttitudeData::SpinNutationMom { .. } => {
            Err(aem_spin_conversion_error(AEMAttitudeType::SpinNutationMom))
        }
    }
}

impl AEM {
    /// Convert a single AEM segment to an `AttitudeTrajectory`.
    ///
    /// Frame endpoints convert via `ReferenceFrame::try_from(&ADMReferenceFrame)`
    /// (the PR-1 frame bridge). Each attitude representation normalizes to a
    /// canonical quaternion (frame A to frame B) plus optional body-frame
    /// angular velocity: Quaternion* variants convert directly or via PR-2's
    /// attitude kinematics (`euler_rates_to_angular_velocity`,
    /// `angular_velocity_from_quaternion_derivative`); `*AngVel` variants
    /// re-express into frame B when `ANGVEL_FRAME` names frame A (see
    /// [`aem_angvel_to_canonical`]). The SPIN* types are spin-parameterized
    /// attitude types with no `AttitudeTrajectory` representation and return
    /// a descriptive error without affecting the message's ability to be
    /// read or written.
    ///
    /// The segment's `INTERPOLATION_METHOD` maps onto
    /// [`AttitudeInterpolationMethod`]: unset defaults to `Slerp`, `LINEAR`
    /// maps to `Linear`, `LAGRANGE` maps to `Lagrange { degree }` (the degree
    /// is guaranteed present by `AEMMetadata::validate`), and `HERMITE` has
    /// no `AttitudeTrajectory` equivalent and errors, directing the caller to
    /// pick an explicit interpolation method instead.
    ///
    /// `metadata.validate()` runs first, so every conversion path (in
    /// particular the ANGVEL_FRAME re-expression) can assume the conditional
    /// metadata rules hold even for a segment built directly in code rather
    /// than parsed from the wire.
    ///
    /// # Arguments
    ///
    /// * `segment_idx` - Index of the segment to convert (0-based)
    ///
    /// # Returns
    ///
    /// * `Result<AttitudeTrajectory, BraheError>` - Trajectory or error
    pub fn segment_to_attitude_trajectory(
        &self,
        segment_idx: usize,
    ) -> Result<AttitudeTrajectory, BraheError> {
        let segment = self.segments.get(segment_idx).ok_or_else(|| {
            BraheError::OutOfBoundsError(format!(
                "AEM segment index {} out of range (have {})",
                segment_idx,
                self.segments.len()
            ))
        })?;

        let metadata = &segment.metadata;
        // Gates every downstream conversion path (in particular
        // `aem_angvel_to_canonical`'s ANGVEL_FRAME == REF_FRAME_A check) on
        // the ANGVEL_FRAME in {REF_FRAME_A, REF_FRAME_B} invariant; a
        // segment built in code rather than parsed from the wire is not
        // otherwise guaranteed to satisfy it.
        metadata.validate()?;
        let frame_a = ReferenceFrame::try_from(&metadata.ref_frame_a)?;
        let frame_b = ReferenceFrame::try_from(&metadata.ref_frame_b)?;

        let interpolation_method = match metadata.interpolation_method {
            None => AttitudeInterpolationMethod::Slerp,
            Some(AEMInterpolationMethod::Linear) => AttitudeInterpolationMethod::Linear,
            Some(AEMInterpolationMethod::Lagrange) => {
                let degree = metadata.interpolation_degree.ok_or_else(|| {
                    BraheError::Error(
                        "AEM metadata INTERPOLATION_METHOD is LAGRANGE but \
                         INTERPOLATION_DEGREE is missing"
                            .to_string(),
                    )
                })?;
                if degree == 0 {
                    return Err(BraheError::Error(
                        "AEM metadata INTERPOLATION_METHOD is LAGRANGE but \
                         INTERPOLATION_DEGREE is 0; Lagrange interpolation requires a degree \
                         >= 1"
                            .to_string(),
                    ));
                }
                AttitudeInterpolationMethod::Lagrange {
                    degree: degree as usize,
                }
            }
            Some(AEMInterpolationMethod::Hermite) => {
                return Err(BraheError::Error(
                    "AEM segment INTERPOLATION_METHOD is HERMITE, which AttitudeTrajectory does \
                     not support; construct the trajectory and call \
                     set_interpolation_method with an explicit choice instead of converting \
                     the AEM interpolation metadata"
                        .to_string(),
                ));
            }
        };

        let mut epochs = Vec::with_capacity(segment.states.len());
        let mut states = Vec::with_capacity(segment.states.len());
        for state in &segment.states {
            epochs.push(state.epoch);
            states.push(aem_attitude_data_to_state(&state.data, metadata)?);
        }

        let mut traj = AttitudeTrajectory::from_data(epochs, states, frame_a, frame_b)?;
        traj.set_interpolation_method(interpolation_method);
        traj.name = Some(metadata.object_name.clone());
        traj.metadata.insert(
            "object_id".to_string(),
            serde_json::Value::String(metadata.object_id.clone()),
        );
        if let Some(center_name) = &metadata.center_name {
            traj.metadata.insert(
                "center_name".to_string(),
                serde_json::Value::String(center_name.clone()),
            );
        }
        if let Some(useable_start_time) = metadata.useable_start_time {
            traj.metadata.insert(
                "useable_start_time".to_string(),
                serde_json::Value::String(format_ccsds_datetime_in(
                    &useable_start_time,
                    &metadata.time_system,
                )),
            );
        }
        if let Some(useable_stop_time) = metadata.useable_stop_time {
            traj.metadata.insert(
                "useable_stop_time".to_string(),
                serde_json::Value::String(format_ccsds_datetime_in(
                    &useable_stop_time,
                    &metadata.time_system,
                )),
            );
        }

        Ok(traj)
    }

    /// Convert all segments to `AttitudeTrajectory` objects.
    pub fn to_attitude_trajectories(&self) -> Result<Vec<AttitudeTrajectory>, BraheError> {
        (0..self.segments.len())
            .map(|i| self.segment_to_attitude_trajectory(i))
            .collect()
    }

    /// Builds a single-segment AEM from a native `AttitudeTrajectory`.
    ///
    /// `ATTITUDE_TYPE` is `QUATERNION/ANGVEL` when the trajectory carries
    /// angular velocity (`AttitudeTrajectory::has_rates`), with
    /// `ANGVEL_FRAME` set to `REF_FRAME_B` (the trajectory's canonical
    /// angular velocity convention already expresses rates in frame B, so no
    /// re-expression is needed on write); otherwise `QUATERNION`. Frame
    /// endpoints convert via `ADMReferenceFrame::try_from(&ReferenceFrame)`
    /// (the reverse PR-1 frame bridge). `START_TIME`/`STOP_TIME` are the
    /// trajectory's first and last epochs, and the header is built from
    /// `originator` via `AEMHeader::new`. The trajectory's interpolation
    /// method maps into `INTERPOLATION_METHOD`/`INTERPOLATION_DEGREE`:
    /// `Linear` maps to `LINEAR` with degree 1 (`AttitudeInterpolationMethod`
    /// has no degree concept for linear interpolation, and
    /// [`AEMMetadata::validate`] requires a degree whenever a method is
    /// set), `Lagrange { degree }` maps to `LAGRANGE` with that degree, and
    /// `Slerp` has no CCSDS equivalent and leaves both fields unset.
    ///
    /// # Arguments
    ///
    /// * `traj` - Trajectory to convert
    /// * `object_name` - `OBJECT_NAME` metadata value
    /// * `object_id` - `OBJECT_ID` metadata value
    /// * `originator` - Message originator (creating agency or operator)
    /// * `time_system` - Time system to record in `TIME_SYSTEM`
    ///
    /// # Returns
    ///
    /// * `Result<AEM, BraheError>` - Single-segment AEM, or an error if the
    ///   trajectory is empty, its frames have no CCSDS ADM token equivalent,
    ///   or `time_system` has no native brahe representation (`SCLK`, `MET`,
    ///   `MRT`, `GMST`, `TDR`)
    pub fn from_attitude_trajectory(
        traj: &AttitudeTrajectory,
        object_name: &str,
        object_id: &str,
        originator: &str,
        time_system: CCSDSTimeSystem,
    ) -> Result<AEM, BraheError> {
        if traj.is_empty() {
            return Err(BraheError::Error(
                "Cannot build an AEM from an empty AttitudeTrajectory".to_string(),
            ));
        }

        if time_system.to_time_system().is_none() {
            return Err(BraheError::Error(format!(
                "TIME_SYSTEM '{}' cannot be used to build an AEM: brahe has no native \
                 representation for its epochs (SCLK, MET, MRT, GMST, and TDR are spacecraft- \
                 or mission-specific clocks with no fixed relationship to brahe's physical time \
                 systems), so a message using it could not be read back by brahe's own parsers",
                time_system
            )));
        }

        let ref_frame_a = ADMReferenceFrame::try_from(&traj.frame_a)?;
        let ref_frame_b = ADMReferenceFrame::try_from(&traj.frame_b)?;

        let attitude_type = if traj.has_rates() {
            AEMAttitudeType::QuaternionAngVel
        } else {
            AEMAttitudeType::Quaternion
        };

        let start_time = traj.start_epoch().unwrap();
        let stop_time = traj.end_epoch().unwrap();

        let mut metadata = AEMMetadata::new(
            object_name,
            object_id,
            ref_frame_a,
            ref_frame_b.clone(),
            time_system,
            start_time,
            stop_time,
            attitude_type,
        );
        if traj.has_rates() {
            metadata = metadata.with_angvel_frame(ref_frame_b);
        }

        // Map the trajectory's interpolation method into segment metadata.
        // `AEMMetadata::validate` requires `interpolation_degree` whenever
        // `interpolation_method` is set, so `Linear` (which has no degree
        // concept in `AttitudeInterpolationMethod`) is recorded with degree
        // 1, the minimal degree consistent with linear interpolation.
        // `Slerp` has no CCSDS interpolation-method equivalent and is left
        // unset.
        metadata = match traj.interpolation_method {
            AttitudeInterpolationMethod::Linear => {
                metadata.with_interpolation(AEMInterpolationMethod::Linear, Some(1))
            }
            AttitudeInterpolationMethod::Lagrange { degree } => {
                metadata.with_interpolation(AEMInterpolationMethod::Lagrange, Some(degree as u32))
            }
            AttitudeInterpolationMethod::Slerp => metadata,
        };

        let mut segment = AEMSegment::new(metadata);
        for i in 0..traj.len() {
            let (epoch, state) = traj.get(i)?;
            let data = match state.angular_velocity {
                Some(angular_velocity) => AEMAttitudeData::QuaternionAngVel {
                    quaternion: state.quaternion,
                    angular_velocity,
                },
                None => AEMAttitudeData::Quaternion {
                    quaternion: state.quaternion,
                },
            };
            segment.push_state(AEMAttitudeState { epoch, data })?;
        }

        let mut aem = AEM::new(originator);
        aem.push_segment(segment);
        Ok(aem)
    }
}

impl AEM {
    /// Registers this AEM's attitude as the orientation of `name`'s body
    /// frame in the global frame registry.
    ///
    /// Converts the AEM to an [`AttitudeTrajectory`] via `TryFrom<&AEM>`
    /// (erroring for a zero- or multi-segment AEM exactly as that conversion
    /// does), then registers it as the link between the segment's two
    /// `REF_FRAME` endpoints.
    ///
    /// A CCSDS message names its frames but not the object they belong to,
    /// so the body endpoint parses unbound and is bound to `name` here. One
    /// endpoint must resolve to a [`CelestialFrame`] — that becomes the
    /// parent — and the other must be a body frame, which becomes the
    /// registered frame. Because `REF_FRAME_A`/`REF_FRAME_B` order is not
    /// fixed by the standard, the quaternion series is inverted when the
    /// celestial frame is endpoint B, so the registered provider always
    /// rotates parent-frame vectors into the body frame as
    /// [`OrientationProvider`] requires.
    ///
    /// # Arguments
    ///
    /// * `name` - The object identity to bind the body frame endpoint to
    ///
    /// # Returns
    ///
    /// * `Result<(), BraheError>` - `Ok(())` on success, or an error if the
    ///   AEM does not have exactly one segment, neither endpoint resolves to
    ///   a celestial frame, or the remaining endpoint is not a body frame
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use brahe::ccsds::AEM;
    /// use brahe::frames::clear_frame_registry;
    /// use std::str::FromStr;
    ///
    /// let content = std::fs::read_to_string("test_assets/ccsds/aem/AEMExampleG5.txt").unwrap();
    /// let aem = AEM::from_str(&content).unwrap();
    ///
    /// clear_frame_registry();
    /// aem.register_for("SC").unwrap();
    /// clear_frame_registry();
    /// ```
    pub fn register_for(&self, name: impl Into<ObjectId>) -> Result<(), BraheError> {
        let traj = AttitudeTrajectory::try_from(self)?;
        let object = name.into();

        // Exactly one endpoint must be celestial: it is the parent the body
        // frame's orientation is expressed relative to.
        let (parent, body_frame, invert) = match (&traj.frame_a, &traj.frame_b) {
            (ReferenceFrame::Celestial(parent), ReferenceFrame::Body { frame, .. }) => {
                (*parent, frame.clone(), false)
            }
            (ReferenceFrame::Body { frame, .. }, ReferenceFrame::Celestial(parent)) => {
                (*parent, frame.clone(), true)
            }
            (a, b) => {
                return Err(BraheError::Error(format!(
                    "AEM::register_for requires one REF_FRAME endpoint to resolve to a \
                     celestial frame and the other to be a body frame, but this segment \
                     relates '{a}' and '{b}'; only a body frame can be bound to an object \
                     and registered"
                )));
            }
        };

        let registered = if invert {
            invert_trajectory(&traj)
        } else {
            traj
        };
        register_frame(
            ReferenceFrame::body(object, body_frame),
            ReferenceFrame::Celestial(parent),
            registered,
        )
    }
}

/// Reverses an attitude trajectory's sense, so that quaternions which
/// rotated frame A into frame B instead rotate frame B into frame A.
///
/// The quaternion of each state is conjugated. Its angular velocity, which
/// is the rate of B relative to A expressed in B, becomes the rate of A
/// relative to B expressed in A: negating gives `omega_{A/B}`, and
/// re-expressing it in A applies `R_{A->B}^T`.
fn invert_trajectory(traj: &AttitudeTrajectory) -> AttitudeTrajectory {
    let mut inverted = AttitudeTrajectory::new(traj.frame_b.clone(), traj.frame_a.clone());
    inverted.set_interpolation_method(traj.interpolation_method);
    inverted.name = traj.name.clone();
    inverted.metadata = traj.metadata.clone();

    inverted.epochs = traj.epochs.clone();
    inverted.states = traj
        .states
        .iter()
        .map(|state| {
            let mut flipped = AttitudeState::new(state.quaternion.conjugate());
            if let Some(omega) = state.angular_velocity {
                let r_a_to_b = state.quaternion.to_rotation_matrix().to_matrix();
                flipped = flipped.with_angular_velocity(-r_a_to_b.transpose() * omega);
            }
            flipped
        })
        .collect();

    inverted
}

impl TryFrom<&AEM> for AttitudeTrajectory {
    type Error = BraheError;

    /// Convert a single-segment AEM to an `AttitudeTrajectory`.
    ///
    /// Returns an error if the AEM has zero or more than one segment.
    fn try_from(aem: &AEM) -> Result<Self, Self::Error> {
        if aem.segments.len() != 1 {
            return Err(BraheError::Error(format!(
                "TryFrom<&AEM> requires exactly 1 segment, but AEM has {}",
                aem.segments.len()
            )));
        }
        aem.segment_to_attitude_trajectory(0)
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

impl TryFrom<&ADMReferenceFrame> for ReferenceFrame {
    type Error = BraheError;

    /// Converts a CCSDS ADM frame into a native [`ReferenceFrame`].
    ///
    /// Celestial frames map onto [`ReferenceFrame::Celestial`] where brahe
    /// implements the frame (ICRF/GCRF → GCRF, EME2000/J2000 → EME2000, ITRF
    /// realizations → ITRF, MOON_ME → LFME). Only the bare `MOON_PA` token and
    /// its explicit DE440 realization (`MOON_PA440`) map to `LFPA`, since
    /// brahe's LFPA is the DE440 lunar principal-axes frame and other DE
    /// realizations (e.g. `MOON_PA421`) differ materially. Orbit-relative and
    /// spacecraft frames map structurally onto unbound
    /// [`ReferenceFrame::OrbitRelative`] and [`ReferenceFrame::Body`] frames:
    /// a CCSDS message names the frame but not the object it belongs to, so
    /// binding an [`ObjectId`] is the caller's job. All other frames return an
    /// error; the containing message still loads and writes — only conversion
    /// to native types is unsupported.
    fn try_from(frame: &ADMReferenceFrame) -> Result<Self, Self::Error> {
        match frame {
            ADMReferenceFrame::Celestial(celestial) => {
                let reference = match celestial {
                    CCSDSCelestialBodyFrame::ICRF(_) | CCSDSCelestialBodyFrame::GCRF(_) => {
                        CelestialFrame::GCRF
                    }
                    CCSDSCelestialBodyFrame::EME2000 | CCSDSCelestialBodyFrame::J2000 => {
                        CelestialFrame::EME2000
                    }
                    CCSDSCelestialBodyFrame::ITRF(_) => CelestialFrame::ITRF,
                    CCSDSCelestialBodyFrame::MoonPA(None)
                    | CCSDSCelestialBodyFrame::MoonPA(Some(440)) => CelestialFrame::LFPA,
                    CCSDSCelestialBodyFrame::MoonME => CelestialFrame::LFME,
                    other => {
                        return Err(BraheError::Error(format!(
                            "CCSDS celestial frame '{}' has no brahe CelestialFrame equivalent; \
                             the message can still be read and written, but not converted to \
                             native attitude types",
                            other
                        )));
                    }
                };
                Ok(ReferenceFrame::Celestial(reference))
            }
            ADMReferenceFrame::OrbitRelative(orbit_relative) => {
                let (kind, variant) = match orbit_relative {
                    CCSDSOrbitRelativeFrame::EQWInertial => (
                        OrbitRelativeFrameKind::EQW,
                        OrbitRelativeFrameVariant::Inertial,
                    ),
                    CCSDSOrbitRelativeFrame::LVLHInertial => (
                        OrbitRelativeFrameKind::LVLH,
                        OrbitRelativeFrameVariant::Inertial,
                    ),
                    CCSDSOrbitRelativeFrame::LVLHRotating => (
                        OrbitRelativeFrameKind::LVLH,
                        OrbitRelativeFrameVariant::Rotating,
                    ),
                    CCSDSOrbitRelativeFrame::NSWInertial => (
                        OrbitRelativeFrameKind::NSW,
                        OrbitRelativeFrameVariant::Inertial,
                    ),
                    CCSDSOrbitRelativeFrame::NSWRotating => (
                        OrbitRelativeFrameKind::NSW,
                        OrbitRelativeFrameVariant::Rotating,
                    ),
                    CCSDSOrbitRelativeFrame::NTWInertial => (
                        OrbitRelativeFrameKind::NTW,
                        OrbitRelativeFrameVariant::Inertial,
                    ),
                    CCSDSOrbitRelativeFrame::NTWRotating => (
                        OrbitRelativeFrameKind::NTW,
                        OrbitRelativeFrameVariant::Rotating,
                    ),
                    CCSDSOrbitRelativeFrame::PQWInertial => (
                        OrbitRelativeFrameKind::PQW,
                        OrbitRelativeFrameVariant::Inertial,
                    ),
                    CCSDSOrbitRelativeFrame::RSWInertial => (
                        OrbitRelativeFrameKind::RTN,
                        OrbitRelativeFrameVariant::Inertial,
                    ),
                    CCSDSOrbitRelativeFrame::RSWRotating => (
                        OrbitRelativeFrameKind::RTN,
                        OrbitRelativeFrameVariant::Rotating,
                    ),
                    CCSDSOrbitRelativeFrame::SEZInertial => (
                        OrbitRelativeFrameKind::SEZ,
                        OrbitRelativeFrameVariant::Inertial,
                    ),
                    CCSDSOrbitRelativeFrame::SEZRotating => (
                        OrbitRelativeFrameKind::SEZ,
                        OrbitRelativeFrameVariant::Rotating,
                    ),
                    CCSDSOrbitRelativeFrame::TNWInertial => (
                        OrbitRelativeFrameKind::TNW,
                        OrbitRelativeFrameVariant::Inertial,
                    ),
                    CCSDSOrbitRelativeFrame::TNWRotating => (
                        OrbitRelativeFrameKind::TNW,
                        OrbitRelativeFrameVariant::Rotating,
                    ),
                    CCSDSOrbitRelativeFrame::VNCInertial => (
                        OrbitRelativeFrameKind::VNC,
                        OrbitRelativeFrameVariant::Inertial,
                    ),
                    CCSDSOrbitRelativeFrame::VNCRotating => (
                        OrbitRelativeFrameKind::VNC,
                        OrbitRelativeFrameVariant::Rotating,
                    ),
                    CCSDSOrbitRelativeFrame::Other(token) => {
                        return Err(BraheError::Error(format!(
                            "CCSDS orbit-relative frame '{}' is not a SANA registry frame and \
                             cannot be converted to a native attitude frame",
                            token
                        )));
                    }
                };
                // All 16 CCSDS orbit-relative frames map to a valid native
                // combination (EQW/PQW only ever produce `Inertial` above).
                Ok(ReferenceFrame::orbit_relative(kind, variant, None).expect(
                    "CCSDS orbit-relative frame registry never pairs EQW/PQW with Rotating",
                ))
            }
            ADMReferenceFrame::Spacecraft(spacecraft) => {
                let native = match spacecraft {
                    CCSDSSpacecraftBodyFrame::ACC(i) => BodyFrame::ACC(i.clone()),
                    CCSDSSpacecraftBodyFrame::Actuator(i) => BodyFrame::Actuator(i.clone()),
                    CCSDSSpacecraftBodyFrame::AST(i) => BodyFrame::AST(i.clone()),
                    CCSDSSpacecraftBodyFrame::CSS(i) => BodyFrame::CSS(i.clone()),
                    CCSDSSpacecraftBodyFrame::DSS(i) => BodyFrame::DSS(i.clone()),
                    CCSDSSpacecraftBodyFrame::ESA(i) => BodyFrame::ESA(i.clone()),
                    CCSDSSpacecraftBodyFrame::GyroFrame(i) => BodyFrame::GyroFrame(i.clone()),
                    CCSDSSpacecraftBodyFrame::IMUFrame(i) => BodyFrame::IMUFrame(i.clone()),
                    CCSDSSpacecraftBodyFrame::Instrument(i) => BodyFrame::Instrument(i.clone()),
                    CCSDSSpacecraftBodyFrame::MTA(i) => BodyFrame::MTA(i.clone()),
                    CCSDSSpacecraftBodyFrame::RW(i) => BodyFrame::RW(i.clone()),
                    CCSDSSpacecraftBodyFrame::SA(i) => BodyFrame::SA(i.clone()),
                    CCSDSSpacecraftBodyFrame::SCBody(i) => BodyFrame::SCBody(i.clone()),
                    CCSDSSpacecraftBodyFrame::Sensor(i) => BodyFrame::Sensor(i.clone()),
                    CCSDSSpacecraftBodyFrame::StarTracker(i) => BodyFrame::StarTracker(i.clone()),
                    CCSDSSpacecraftBodyFrame::TAM(i) => BodyFrame::TAM(i.clone()),
                    CCSDSSpacecraftBodyFrame::Other(token) => {
                        return Err(BraheError::Error(format!(
                            "CCSDS spacecraft frame '{}' is not a SANA registry frame and \
                             cannot be converted to a native attitude frame",
                            token
                        )));
                    }
                };
                Ok(ReferenceFrame::from(native))
            }
            ADMReferenceFrame::Other(token) => Err(BraheError::Error(format!(
                "CCSDS frame '{}' is in none of the SANA ADM frame registries and cannot be \
                 converted to a native attitude frame",
                token
            ))),
        }
    }
}

impl TryFrom<&ReferenceFrame> for ADMReferenceFrame {
    type Error = BraheError;

    /// Converts a native [`ReferenceFrame`] into a CCSDS ADM frame token for
    /// writing messages.
    ///
    /// [`ReferenceFrame::Celestial`] variants without a SANA celestial token
    /// (synodic, body-centered generic, Mars/lunar inertial, ...) return an
    /// error. Orbit-relative and body frames map by kind/designator alone: an
    /// ADM frame keyword carries no object field, so a bound frame writes the
    /// same token as its unbound counterpart and the binding is dropped.
    fn try_from(frame: &ReferenceFrame) -> Result<Self, Self::Error> {
        match frame {
            ReferenceFrame::Celestial(reference) => {
                let celestial = match reference {
                    CelestialFrame::GCRF => CCSDSCelestialBodyFrame::GCRF(None),
                    CelestialFrame::EME2000 => CCSDSCelestialBodyFrame::EME2000,
                    CelestialFrame::ITRF => CCSDSCelestialBodyFrame::ITRF(None),
                    CelestialFrame::LFPA => CCSDSCelestialBodyFrame::MoonPA(None),
                    CelestialFrame::LFME => CCSDSCelestialBodyFrame::MoonME,
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
            ReferenceFrame::OrbitRelative { kind, variant, .. } => {
                use OrbitRelativeFrameKind as K;
                use OrbitRelativeFrameVariant as V;
                let ccsds = match (kind, variant) {
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
                    (K::EQW, V::Rotating) | (K::PQW, V::Rotating) => {
                        // `ReferenceFrame::orbit_relative` rejects this pair,
                        // but the enum variant's fields are public, so a
                        // struct-literal frame can still carry it.
                        return Err(BraheError::Error(format!(
                            "orbit-relative frame {:?} exists only as an inertial SANA frame",
                            kind
                        )));
                    }
                };
                Ok(ADMReferenceFrame::OrbitRelative(ccsds))
            }
            ReferenceFrame::Body { frame, .. } => {
                let ccsds = match frame {
                    BodyFrame::ACC(i) => CCSDSSpacecraftBodyFrame::ACC(i.clone()),
                    BodyFrame::Actuator(i) => CCSDSSpacecraftBodyFrame::Actuator(i.clone()),
                    BodyFrame::AST(i) => CCSDSSpacecraftBodyFrame::AST(i.clone()),
                    BodyFrame::CSS(i) => CCSDSSpacecraftBodyFrame::CSS(i.clone()),
                    BodyFrame::DSS(i) => CCSDSSpacecraftBodyFrame::DSS(i.clone()),
                    BodyFrame::ESA(i) => CCSDSSpacecraftBodyFrame::ESA(i.clone()),
                    BodyFrame::GyroFrame(i) => CCSDSSpacecraftBodyFrame::GyroFrame(i.clone()),
                    BodyFrame::IMUFrame(i) => CCSDSSpacecraftBodyFrame::IMUFrame(i.clone()),
                    BodyFrame::Instrument(i) => CCSDSSpacecraftBodyFrame::Instrument(i.clone()),
                    BodyFrame::MTA(i) => CCSDSSpacecraftBodyFrame::MTA(i.clone()),
                    BodyFrame::RW(i) => CCSDSSpacecraftBodyFrame::RW(i.clone()),
                    BodyFrame::SA(i) => CCSDSSpacecraftBodyFrame::SA(i.clone()),
                    BodyFrame::SCBody(i) => CCSDSSpacecraftBodyFrame::SCBody(i.clone()),
                    BodyFrame::Sensor(i) => CCSDSSpacecraftBodyFrame::Sensor(i.clone()),
                    BodyFrame::StarTracker(i) => CCSDSSpacecraftBodyFrame::StarTracker(i.clone()),
                    BodyFrame::TAM(i) => CCSDSSpacecraftBodyFrame::TAM(i.clone()),
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
    use crate::attitude::{EulerAngle, EulerAngleOrder};
    use crate::ccsds::aem::AEM;
    use crate::ccsds::oem::OEM;
    use crate::constants::AngleFormat;
    use crate::frames::{
        CelestialFrame, OrientationProvider, ReferenceFrame, clear_frame_registry,
        clear_object_registry, object_state, rotation_frame_to_frame,
    };
    use crate::time::TimeSystem;
    use crate::trajectories::traits::{InterpolatableTrajectory, Trajectory};
    use approx::assert_abs_diff_eq;

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
    fn test_adm_frame_to_reference_frame_celestial() {
        let cases = [
            ("ICRF", CelestialFrame::GCRF),
            ("GCRF", CelestialFrame::GCRF),
            ("GCRF2", CelestialFrame::GCRF),
            ("EME2000", CelestialFrame::EME2000),
            ("J2000", CelestialFrame::EME2000),
            ("ITRF", CelestialFrame::ITRF),
            ("ITRF2014", CelestialFrame::ITRF),
            ("MOON_PA", CelestialFrame::LFPA),
            ("MOON_PA440", CelestialFrame::LFPA),
            ("MOON_ME", CelestialFrame::LFME),
        ];
        for (token, expected) in cases {
            let adm = ADMReferenceFrame::parse(token);
            let att = ReferenceFrame::try_from(&adm).unwrap();
            assert_eq!(att, ReferenceFrame::Celestial(expected), "token {}", token);
        }
    }

    #[test]
    #[parallel]
    fn test_adm_frame_to_reference_frame_unsupported() {
        for token in ["TOD_EARTH", "B1950", "WGS84", "TEMEOFDATE", "BODY_FRAME_A"] {
            let adm = ADMReferenceFrame::parse(token);
            let err = ReferenceFrame::try_from(&adm).unwrap_err();
            assert!(
                err.to_string().contains(token),
                "error should name {}",
                token
            );
        }
    }

    #[test]
    #[parallel]
    fn test_adm_frame_to_reference_frame_structural() {
        let adm = ADMReferenceFrame::parse("RSW_ROTATING");
        assert_eq!(
            ReferenceFrame::try_from(&adm).unwrap(),
            ReferenceFrame::orbit_relative(
                OrbitRelativeFrameKind::RTN,
                OrbitRelativeFrameVariant::Rotating,
                None
            )
            .unwrap()
        );
        let adm = ADMReferenceFrame::parse("INSTRUMENT_A");
        assert_eq!(
            ReferenceFrame::try_from(&adm).unwrap(),
            ReferenceFrame::from(BodyFrame::Instrument(Some("A".to_string())))
        );
    }

    #[test]
    #[parallel]
    fn test_reference_frame_to_adm_frame_roundtrip() {
        // Every mappable native frame must round-trip through ADM tokens
        let frames = [
            ReferenceFrame::Celestial(CelestialFrame::GCRF),
            ReferenceFrame::Celestial(CelestialFrame::EME2000),
            ReferenceFrame::Celestial(CelestialFrame::ITRF),
            ReferenceFrame::Celestial(CelestialFrame::LFPA),
            ReferenceFrame::Celestial(CelestialFrame::LFME),
            ReferenceFrame::orbit_relative(
                OrbitRelativeFrameKind::LVLH,
                OrbitRelativeFrameVariant::Rotating,
                None,
            )
            .unwrap(),
            ReferenceFrame::from(BodyFrame::SCBody(Some("1".to_string()))),
        ];
        for frame in frames {
            let adm = ADMReferenceFrame::try_from(&frame).unwrap();
            let back = ReferenceFrame::try_from(&adm).unwrap();
            assert_eq!(back, frame);
        }
    }

    #[test]
    #[parallel]
    fn test_bound_reference_frame_to_adm_frame_drops_object() {
        // An ADM frame keyword carries no object field, so a bound frame
        // writes the same token as its unbound counterpart.
        let bound = ReferenceFrame::RTN("SC");
        let unbound = ReferenceFrame::orbit_relative(
            OrbitRelativeFrameKind::RTN,
            OrbitRelativeFrameVariant::Rotating,
            None,
        )
        .unwrap();
        assert_eq!(
            ADMReferenceFrame::try_from(&bound).unwrap(),
            ADMReferenceFrame::try_from(&unbound).unwrap()
        );

        let bound = ReferenceFrame::SC_BODY("SC");
        let unbound = ReferenceFrame::from(BodyFrame::SCBody(None));
        assert_eq!(
            ADMReferenceFrame::try_from(&bound).unwrap(),
            ADMReferenceFrame::try_from(&unbound).unwrap()
        );
    }

    #[test]
    #[parallel]
    fn test_reference_frame_to_adm_frame_unsupported() {
        let frame = ReferenceFrame::Celestial(CelestialFrame::EMR);
        assert!(ADMReferenceFrame::try_from(&frame).is_err());
    }

    #[test]
    #[parallel]
    fn test_adm_orbit_relative_frame_to_reference_frame_all_kinds() {
        let cases = [
            (
                CCSDSOrbitRelativeFrame::EQWInertial,
                OrbitRelativeFrameKind::EQW,
                OrbitRelativeFrameVariant::Inertial,
            ),
            (
                CCSDSOrbitRelativeFrame::LVLHInertial,
                OrbitRelativeFrameKind::LVLH,
                OrbitRelativeFrameVariant::Inertial,
            ),
            (
                CCSDSOrbitRelativeFrame::LVLHRotating,
                OrbitRelativeFrameKind::LVLH,
                OrbitRelativeFrameVariant::Rotating,
            ),
            (
                CCSDSOrbitRelativeFrame::NSWInertial,
                OrbitRelativeFrameKind::NSW,
                OrbitRelativeFrameVariant::Inertial,
            ),
            (
                CCSDSOrbitRelativeFrame::NSWRotating,
                OrbitRelativeFrameKind::NSW,
                OrbitRelativeFrameVariant::Rotating,
            ),
            (
                CCSDSOrbitRelativeFrame::NTWInertial,
                OrbitRelativeFrameKind::NTW,
                OrbitRelativeFrameVariant::Inertial,
            ),
            (
                CCSDSOrbitRelativeFrame::NTWRotating,
                OrbitRelativeFrameKind::NTW,
                OrbitRelativeFrameVariant::Rotating,
            ),
            (
                CCSDSOrbitRelativeFrame::PQWInertial,
                OrbitRelativeFrameKind::PQW,
                OrbitRelativeFrameVariant::Inertial,
            ),
            (
                CCSDSOrbitRelativeFrame::RSWInertial,
                OrbitRelativeFrameKind::RTN,
                OrbitRelativeFrameVariant::Inertial,
            ),
            (
                CCSDSOrbitRelativeFrame::RSWRotating,
                OrbitRelativeFrameKind::RTN,
                OrbitRelativeFrameVariant::Rotating,
            ),
            (
                CCSDSOrbitRelativeFrame::SEZInertial,
                OrbitRelativeFrameKind::SEZ,
                OrbitRelativeFrameVariant::Inertial,
            ),
            (
                CCSDSOrbitRelativeFrame::SEZRotating,
                OrbitRelativeFrameKind::SEZ,
                OrbitRelativeFrameVariant::Rotating,
            ),
            (
                CCSDSOrbitRelativeFrame::TNWInertial,
                OrbitRelativeFrameKind::TNW,
                OrbitRelativeFrameVariant::Inertial,
            ),
            (
                CCSDSOrbitRelativeFrame::TNWRotating,
                OrbitRelativeFrameKind::TNW,
                OrbitRelativeFrameVariant::Rotating,
            ),
            (
                CCSDSOrbitRelativeFrame::VNCInertial,
                OrbitRelativeFrameKind::VNC,
                OrbitRelativeFrameVariant::Inertial,
            ),
            (
                CCSDSOrbitRelativeFrame::VNCRotating,
                OrbitRelativeFrameKind::VNC,
                OrbitRelativeFrameVariant::Rotating,
            ),
        ];
        for (ccsds, kind, variant) in cases {
            let adm = ADMReferenceFrame::OrbitRelative(ccsds.clone());
            let att = ReferenceFrame::try_from(&adm).unwrap();
            assert_eq!(
                att,
                ReferenceFrame::orbit_relative(kind, variant, None).unwrap(),
                "ccsds frame {:?}",
                ccsds
            );
        }
    }

    #[test]
    #[parallel]
    fn test_reference_orbit_relative_frame_to_adm_frame_all_kinds() {
        let cases = [
            (
                OrbitRelativeFrameKind::EQW,
                OrbitRelativeFrameVariant::Inertial,
                CCSDSOrbitRelativeFrame::EQWInertial,
            ),
            (
                OrbitRelativeFrameKind::LVLH,
                OrbitRelativeFrameVariant::Inertial,
                CCSDSOrbitRelativeFrame::LVLHInertial,
            ),
            (
                OrbitRelativeFrameKind::LVLH,
                OrbitRelativeFrameVariant::Rotating,
                CCSDSOrbitRelativeFrame::LVLHRotating,
            ),
            (
                OrbitRelativeFrameKind::NSW,
                OrbitRelativeFrameVariant::Inertial,
                CCSDSOrbitRelativeFrame::NSWInertial,
            ),
            (
                OrbitRelativeFrameKind::NSW,
                OrbitRelativeFrameVariant::Rotating,
                CCSDSOrbitRelativeFrame::NSWRotating,
            ),
            (
                OrbitRelativeFrameKind::NTW,
                OrbitRelativeFrameVariant::Inertial,
                CCSDSOrbitRelativeFrame::NTWInertial,
            ),
            (
                OrbitRelativeFrameKind::NTW,
                OrbitRelativeFrameVariant::Rotating,
                CCSDSOrbitRelativeFrame::NTWRotating,
            ),
            (
                OrbitRelativeFrameKind::PQW,
                OrbitRelativeFrameVariant::Inertial,
                CCSDSOrbitRelativeFrame::PQWInertial,
            ),
            (
                OrbitRelativeFrameKind::RTN,
                OrbitRelativeFrameVariant::Inertial,
                CCSDSOrbitRelativeFrame::RSWInertial,
            ),
            (
                OrbitRelativeFrameKind::RTN,
                OrbitRelativeFrameVariant::Rotating,
                CCSDSOrbitRelativeFrame::RSWRotating,
            ),
            (
                OrbitRelativeFrameKind::SEZ,
                OrbitRelativeFrameVariant::Inertial,
                CCSDSOrbitRelativeFrame::SEZInertial,
            ),
            (
                OrbitRelativeFrameKind::SEZ,
                OrbitRelativeFrameVariant::Rotating,
                CCSDSOrbitRelativeFrame::SEZRotating,
            ),
            (
                OrbitRelativeFrameKind::TNW,
                OrbitRelativeFrameVariant::Inertial,
                CCSDSOrbitRelativeFrame::TNWInertial,
            ),
            (
                OrbitRelativeFrameKind::TNW,
                OrbitRelativeFrameVariant::Rotating,
                CCSDSOrbitRelativeFrame::TNWRotating,
            ),
            (
                OrbitRelativeFrameKind::VNC,
                OrbitRelativeFrameVariant::Inertial,
                CCSDSOrbitRelativeFrame::VNCInertial,
            ),
            (
                OrbitRelativeFrameKind::VNC,
                OrbitRelativeFrameVariant::Rotating,
                CCSDSOrbitRelativeFrame::VNCRotating,
            ),
        ];
        for (kind, variant, expected) in cases {
            let frame = ReferenceFrame::orbit_relative(kind, variant, None).unwrap();
            let adm = ADMReferenceFrame::try_from(&frame).unwrap();
            assert_eq!(adm, ADMReferenceFrame::OrbitRelative(expected));
        }
    }

    #[test]
    #[parallel]
    fn test_orbit_relative_frame_inertial_only_kinds_rejected() {
        // EQW and PQW exist only as inertial SANA frames.
        // `ReferenceFrame::orbit_relative` rejects the rotating variant, and
        // the ADM bridge rejects a frame built straight from the enum
        // variant's public fields.
        for kind in [OrbitRelativeFrameKind::EQW, OrbitRelativeFrameKind::PQW] {
            assert!(
                ReferenceFrame::orbit_relative(kind, OrbitRelativeFrameVariant::Rotating, None)
                    .is_err()
            );
            let frame = ReferenceFrame::OrbitRelative {
                kind,
                variant: OrbitRelativeFrameVariant::Rotating,
                object: None,
            };
            assert!(ADMReferenceFrame::try_from(&frame).is_err());
        }
    }

    #[test]
    #[parallel]
    fn test_adm_spacecraft_frame_to_reference_frame_all_families() {
        let cases = [
            (
                CCSDSSpacecraftBodyFrame::ACC(Some("1".to_string())),
                BodyFrame::ACC(Some("1".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::Actuator(None),
                BodyFrame::Actuator(None),
            ),
            (
                CCSDSSpacecraftBodyFrame::AST(Some("1".to_string())),
                BodyFrame::AST(Some("1".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::CSS(Some("2".to_string())),
                BodyFrame::CSS(Some("2".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::DSS(Some("1".to_string())),
                BodyFrame::DSS(Some("1".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::ESA(Some("1".to_string())),
                BodyFrame::ESA(Some("1".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::GyroFrame(Some("1".to_string())),
                BodyFrame::GyroFrame(Some("1".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::IMUFrame(Some("2".to_string())),
                BodyFrame::IMUFrame(Some("2".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::Instrument(Some("A".to_string())),
                BodyFrame::Instrument(Some("A".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::MTA(Some("1".to_string())),
                BodyFrame::MTA(Some("1".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::RW(Some("4".to_string())),
                BodyFrame::RW(Some("4".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::SA(Some("1".to_string())),
                BodyFrame::SA(Some("1".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::SCBody(None),
                BodyFrame::SCBody(None),
            ),
            (
                CCSDSSpacecraftBodyFrame::Sensor(Some("10".to_string())),
                BodyFrame::Sensor(Some("10".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::StarTracker(Some("2".to_string())),
                BodyFrame::StarTracker(Some("2".to_string())),
            ),
            (
                CCSDSSpacecraftBodyFrame::TAM(Some("1".to_string())),
                BodyFrame::TAM(Some("1".to_string())),
            ),
        ];
        for (ccsds, expected) in cases {
            let adm = ADMReferenceFrame::Spacecraft(ccsds.clone());
            let att = ReferenceFrame::try_from(&adm).unwrap();
            assert_eq!(
                att,
                ReferenceFrame::from(expected),
                "ccsds frame {:?}",
                ccsds
            );
        }
    }

    #[test]
    #[parallel]
    fn test_reference_spacecraft_frame_to_adm_frame_all_families() {
        let cases = [
            (
                BodyFrame::ACC(Some("1".to_string())),
                CCSDSSpacecraftBodyFrame::ACC(Some("1".to_string())),
            ),
            (
                BodyFrame::Actuator(None),
                CCSDSSpacecraftBodyFrame::Actuator(None),
            ),
            (
                BodyFrame::AST(Some("1".to_string())),
                CCSDSSpacecraftBodyFrame::AST(Some("1".to_string())),
            ),
            (
                BodyFrame::CSS(Some("2".to_string())),
                CCSDSSpacecraftBodyFrame::CSS(Some("2".to_string())),
            ),
            (
                BodyFrame::DSS(Some("1".to_string())),
                CCSDSSpacecraftBodyFrame::DSS(Some("1".to_string())),
            ),
            (
                BodyFrame::ESA(Some("1".to_string())),
                CCSDSSpacecraftBodyFrame::ESA(Some("1".to_string())),
            ),
            (
                BodyFrame::GyroFrame(Some("1".to_string())),
                CCSDSSpacecraftBodyFrame::GyroFrame(Some("1".to_string())),
            ),
            (
                BodyFrame::IMUFrame(Some("2".to_string())),
                CCSDSSpacecraftBodyFrame::IMUFrame(Some("2".to_string())),
            ),
            (
                BodyFrame::Instrument(Some("A".to_string())),
                CCSDSSpacecraftBodyFrame::Instrument(Some("A".to_string())),
            ),
            (
                BodyFrame::MTA(Some("1".to_string())),
                CCSDSSpacecraftBodyFrame::MTA(Some("1".to_string())),
            ),
            (
                BodyFrame::RW(Some("4".to_string())),
                CCSDSSpacecraftBodyFrame::RW(Some("4".to_string())),
            ),
            (
                BodyFrame::SA(Some("1".to_string())),
                CCSDSSpacecraftBodyFrame::SA(Some("1".to_string())),
            ),
            (
                BodyFrame::SCBody(None),
                CCSDSSpacecraftBodyFrame::SCBody(None),
            ),
            (
                BodyFrame::Sensor(Some("10".to_string())),
                CCSDSSpacecraftBodyFrame::Sensor(Some("10".to_string())),
            ),
            (
                BodyFrame::StarTracker(Some("2".to_string())),
                CCSDSSpacecraftBodyFrame::StarTracker(Some("2".to_string())),
            ),
            (
                BodyFrame::TAM(Some("1".to_string())),
                CCSDSSpacecraftBodyFrame::TAM(Some("1".to_string())),
            ),
        ];
        for (native, expected) in cases {
            let frame = ReferenceFrame::from(native);
            let adm = ADMReferenceFrame::try_from(&frame).unwrap();
            assert_eq!(adm, ADMReferenceFrame::Spacecraft(expected));
        }
    }

    #[test]
    #[parallel]
    fn test_adm_orbit_relative_other_to_reference_frame_errors() {
        let adm = ADMReferenceFrame::OrbitRelative(CCSDSOrbitRelativeFrame::Other(
            "CUSTOM_ORBIT_FRAME".to_string(),
        ));
        let err = ReferenceFrame::try_from(&adm).unwrap_err();
        assert!(err.to_string().contains("CUSTOM_ORBIT_FRAME"));
    }

    #[test]
    #[parallel]
    fn test_adm_spacecraft_other_to_reference_frame_errors() {
        let adm = ADMReferenceFrame::Spacecraft(CCSDSSpacecraftBodyFrame::Other(
            "CUSTOM_SC_FRAME".to_string(),
        ));
        let err = ReferenceFrame::try_from(&adm).unwrap_err();
        assert!(err.to_string().contains("CUSTOM_SC_FRAME"));
    }

    #[test]
    #[parallel]
    fn test_adm_celestial_other_to_reference_frame_errors() {
        let adm =
            ADMReferenceFrame::Celestial(CCSDSCelestialBodyFrame::Other("CUSTOM".to_string()));
        let err = ReferenceFrame::try_from(&adm).unwrap_err();
        assert!(err.to_string().contains("CUSTOM"));
    }

    #[test]
    #[parallel]
    fn test_adm_moon_pa_non_de440_realization_errors() {
        let adm = ADMReferenceFrame::parse("MOON_PA421");
        let err = ReferenceFrame::try_from(&adm).unwrap_err();
        assert!(err.to_string().contains("MOON_PA421"));
    }

    // =========================================================================
    // AEM <-> AttitudeTrajectory interop
    // =========================================================================

    #[test]
    #[parallel]
    fn test_aem_g4_segment_to_attitude_trajectory() {
        let content = std::fs::read_to_string("test_assets/ccsds/aem/AEMExampleG4.txt").unwrap();
        let aem = AEM::from_str(&content).unwrap();

        // Segment 1 (0-indexed) carries no INTERPOLATION_METHOD, so it
        // defaults to Slerp and converts cleanly; segment 0 sets
        // INTERPOLATION_METHOD = hermite and is covered by the Hermite
        // error test below.
        let traj = aem.segment_to_attitude_trajectory(1).unwrap();

        assert_eq!(traj.len(), 4);
        assert_eq!(traj.frame_a, ReferenceFrame::from(CelestialFrame::EME2000));
        assert_eq!(
            traj.frame_b,
            ReferenceFrame::from(BodyFrame::SCBody(Some("1".to_string())))
        );
        assert_eq!(
            traj.interpolation_method,
            AttitudeInterpolationMethod::Slerp
        );
        assert_eq!(traj.name.as_deref(), Some("mars global surveyor"));
        assert!(!traj.has_rates());

        // First and last quaternion values must match the parsed AEM data
        // exactly (quaternion conversion for the Quaternion type is a
        // direct passthrough).
        let segment = &aem.segments[1];
        let AEMAttitudeData::Quaternion {
            quaternion: q_first,
        } = &segment.states[0].data
        else {
            panic!("expected Quaternion data");
        };
        let AEMAttitudeData::Quaternion { quaternion: q_last } =
            &segment.states[segment.states.len() - 1].data
        else {
            panic!("expected Quaternion data");
        };
        assert_eq!(traj.state_at_idx(0).unwrap().quaternion, *q_first);
        assert_eq!(traj.state_at_idx(3).unwrap().quaternion, *q_last);

        // Slerp query strictly between the first two data epochs.
        let t0 = traj.epoch_at_idx(0).unwrap();
        let t1 = traj.epoch_at_idx(1).unwrap();
        let mid = t0 + (t1 - t0) / 2.0;
        let interpolated = traj.interpolate(&mid).unwrap();
        let expected = q_first.slerp(traj.state_at_idx(1).unwrap().quaternion, 0.5);
        assert_eq!(interpolated.quaternion, expected);
    }

    #[test]
    #[parallel]
    fn test_aem_g5_spin_conversion_errors() {
        let content = std::fs::read_to_string("test_assets/ccsds/aem/AEMExampleG5.txt").unwrap();
        let aem = AEM::from_str(&content).unwrap();

        let result = aem.segment_to_attitude_trajectory(0);
        assert!(result.is_err());
        let message = format!("{}", result.unwrap_err());
        assert!(message.contains("SPIN"));
    }

    #[test]
    #[parallel]
    fn test_aem_g4_hermite_interpolation_method_errors() {
        let content = std::fs::read_to_string("test_assets/ccsds/aem/AEMExampleG4.txt").unwrap();
        let aem = AEM::from_str(&content).unwrap();

        // Segment 0 sets INTERPOLATION_METHOD = hermite, which has no
        // AttitudeTrajectory equivalent.
        let result = aem.segment_to_attitude_trajectory(0);
        assert!(result.is_err());
        let message = format!("{}", result.unwrap_err());
        assert!(message.contains("HERMITE"));
    }

    #[test]
    #[parallel]
    fn test_aem_to_attitude_trajectories_multi_segment() {
        // Only segment 1 of G-4 converts (segment 0 is Hermite), so
        // exercise the batch API against a message where every segment
        // converts: build one in code with two Slerp-default segments.
        let ref_frame_a = ADMReferenceFrame::parse("EME2000");
        let ref_frame_b = ADMReferenceFrame::parse("SC_BODY_1");
        let t0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;

        let mut aem = AEM::new("BRAHE");
        for _ in 0..2 {
            let metadata = AEMMetadata::new(
                "SAT1",
                "2024-001A",
                ref_frame_a.clone(),
                ref_frame_b.clone(),
                CCSDSTimeSystem::UTC,
                t0,
                t1,
                AEMAttitudeType::Quaternion,
            );
            let mut segment = AEMSegment::new(metadata);
            segment
                .push_state(AEMAttitudeState {
                    epoch: t0,
                    data: AEMAttitudeData::Quaternion {
                        quaternion: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                    },
                })
                .unwrap();
            segment
                .push_state(AEMAttitudeState {
                    epoch: t1,
                    data: AEMAttitudeData::Quaternion {
                        quaternion: Quaternion::new(0.9998, 0.0, 0.0, 0.0196),
                    },
                })
                .unwrap();
            aem.push_segment(segment);
        }

        let trajs = aem.to_attitude_trajectories().unwrap();
        assert_eq!(trajs.len(), 2);
        for traj in &trajs {
            assert_eq!(traj.len(), 2);
        }
    }

    #[test]
    #[parallel]
    fn test_aem_try_from_single_segment() {
        let content = std::fs::read_to_string("test_assets/ccsds/aem/AEMExampleG5.txt").unwrap();
        let aem = AEM::from_str(&content).unwrap();

        // G-5 has a single SPIN segment; TryFrom requires exactly one
        // segment but the underlying conversion still errors on SPIN.
        let result = AttitudeTrajectory::try_from(&aem);
        assert!(result.is_err());
        let message = format!("{}", result.unwrap_err());
        assert!(message.contains("SPIN"));
    }

    #[test]
    #[parallel]
    fn test_aem_try_from_multi_segment_fails() {
        let content = std::fs::read_to_string("test_assets/ccsds/aem/AEMExampleG4.txt").unwrap();
        let aem = AEM::from_str(&content).unwrap();

        let result = AttitudeTrajectory::try_from(&aem);
        assert!(result.is_err());
        let message = format!("{}", result.unwrap_err());
        assert!(message.contains("exactly 1 segment"));
    }

    #[test]
    #[parallel]
    fn test_aem_angvel_frame_a_reexpression() {
        // In-code AEM whose ANGVEL_FRAME names REF_FRAME_A: the stored
        // angular velocity is in frame A and must be re-expressed in frame
        // B (the canonical AttitudeState convention) as omega_B = R(q) *
        // omega_A per Diebel eq. 41.
        let ref_frame_a = ADMReferenceFrame::parse("EME2000");
        let ref_frame_b = ADMReferenceFrame::parse("SC_BODY_1");
        let t0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;

        let metadata = AEMMetadata::new(
            "SAT1",
            "2024-001A",
            ref_frame_a.clone(),
            ref_frame_b,
            CCSDSTimeSystem::UTC,
            t0,
            t1,
            AEMAttitudeType::QuaternionAngVel,
        )
        .with_angvel_frame(ref_frame_a);

        let quaternion = Quaternion::from_euler_angle(EulerAngle::new(
            EulerAngleOrder::ZYX,
            0.3,
            -0.4,
            0.2,
            AngleFormat::Radians,
        ));
        let omega_a = Vector3::new(0.01, -0.02, 0.03);

        let mut segment = AEMSegment::new(metadata);
        segment
            .push_state(AEMAttitudeState {
                epoch: t0,
                data: AEMAttitudeData::QuaternionAngVel {
                    quaternion,
                    angular_velocity: omega_a,
                },
            })
            .unwrap();

        let mut aem = AEM::new("BRAHE");
        aem.push_segment(segment);

        let traj = aem.segment_to_attitude_trajectory(0).unwrap();
        let stored_omega = traj.state_at_idx(0).unwrap().angular_velocity.unwrap();

        let expected_omega = quaternion.to_rotation_matrix().to_matrix() * omega_a;
        assert_abs_diff_eq!(stored_omega[0], expected_omega[0], epsilon = 1e-12);
        assert_abs_diff_eq!(stored_omega[1], expected_omega[1], epsilon = 1e-12);
        assert_abs_diff_eq!(stored_omega[2], expected_omega[2], epsilon = 1e-12);

        // Sanity check: the re-expressed rate must differ from the raw
        // frame-A value (otherwise the re-expression path silently no-ops).
        assert!((stored_omega - omega_a).norm() > 1e-6);
    }

    #[test]
    #[parallel]
    fn test_aem_angvel_frame_neither_a_nor_b_errors_via_validate() {
        // A segment built directly in code (not via the KVN/XML/JSON
        // parsers, which call metadata.validate() themselves) can carry an
        // ANGVEL_FRAME that names neither REF_FRAME_A nor REF_FRAME_B.
        // segment_to_attitude_trajectory must reject this rather than
        // silently treating the "not frame A" branch as frame B.
        let ref_frame_a = ADMReferenceFrame::parse("EME2000");
        let ref_frame_b = ADMReferenceFrame::parse("SC_BODY_1");
        let other_frame = ADMReferenceFrame::parse("ITRF2014");
        let t0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;

        let metadata = AEMMetadata::new(
            "SAT1",
            "2024-001A",
            ref_frame_a,
            ref_frame_b,
            CCSDSTimeSystem::UTC,
            t0,
            t1,
            AEMAttitudeType::QuaternionAngVel,
        )
        .with_angvel_frame(other_frame);

        let mut segment = AEMSegment::new(metadata);
        segment
            .push_state(AEMAttitudeState {
                epoch: t0,
                data: AEMAttitudeData::QuaternionAngVel {
                    quaternion: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                    angular_velocity: Vector3::new(0.01, -0.02, 0.03),
                },
            })
            .unwrap();

        let mut aem = AEM::new("BRAHE");
        aem.push_segment(segment);

        let result = aem.segment_to_attitude_trajectory(0);
        assert!(result.is_err());
        let message = format!("{}", result.unwrap_err());
        assert!(message.contains("ANGVEL_FRAME"));
        assert!(message.contains("must equal"));
    }

    #[test]
    #[parallel]
    fn test_aem_trajectory_round_trip_with_rates() {
        let frame_a = ReferenceFrame::from(CelestialFrame::EME2000);
        let frame_b = ReferenceFrame::from(BodyFrame::SCBody(None));
        let t0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let omega = Vector3::new(0.001, 0.002, -0.003);

        let epochs = vec![t0, t0 + 60.0, t0 + 120.0];
        let states = vec![
            AttitudeState::new(Quaternion::new(1.0, 0.0, 0.0, 0.0)).with_angular_velocity(omega),
            AttitudeState::new(Quaternion::new(0.9998, 0.0, 0.0, 0.0196))
                .with_angular_velocity(omega),
            AttitudeState::new(Quaternion::new(0.9992, 0.0, 0.0, 0.0392))
                .with_angular_velocity(omega),
        ];
        let mut traj =
            AttitudeTrajectory::from_data(epochs, states, frame_a.clone(), frame_b.clone())
                .unwrap();
        traj.name = Some("SAT1".to_string());

        let aem = AEM::from_attitude_trajectory(
            &traj,
            "SAT1",
            "2024-001A",
            "BRAHE",
            CCSDSTimeSystem::UTC,
        )
        .unwrap();
        assert_eq!(aem.segments.len(), 1);
        assert_eq!(
            aem.segments[0].metadata.attitude_type,
            AEMAttitudeType::QuaternionAngVel
        );
        assert_eq!(
            aem.segments[0].metadata.angvel_frame,
            Some(ADMReferenceFrame::try_from(&frame_b).unwrap())
        );

        let round_tripped = AttitudeTrajectory::try_from(&aem).unwrap();

        assert_eq!(round_tripped.len(), traj.len());
        assert_eq!(round_tripped.frame_a, traj.frame_a);
        assert_eq!(round_tripped.frame_b, traj.frame_b);
        assert_eq!(round_tripped.name.as_deref(), Some("SAT1"));
        for i in 0..traj.len() {
            let original = traj.state_at_idx(i).unwrap();
            let recovered = round_tripped.state_at_idx(i).unwrap();
            assert_eq!(
                traj.epoch_at_idx(i).unwrap(),
                round_tripped.epoch_at_idx(i).unwrap()
            );
            assert_eq!(recovered.quaternion, original.quaternion);
            assert_eq!(recovered.angular_velocity, original.angular_velocity);
        }
    }

    #[test]
    #[parallel]
    fn test_aem_trajectory_round_trip_interpolation_method_lagrange() {
        let frame_a = ReferenceFrame::from(CelestialFrame::EME2000);
        let frame_b = ReferenceFrame::from(BodyFrame::SCBody(None));
        let t0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        let epochs = vec![t0, t0 + 60.0, t0 + 120.0];
        let states = vec![
            AttitudeState::new(Quaternion::new(1.0, 0.0, 0.0, 0.0)),
            AttitudeState::new(Quaternion::new(0.9998, 0.0, 0.0, 0.0196)),
            AttitudeState::new(Quaternion::new(0.9992, 0.0, 0.0, 0.0392)),
        ];
        let mut traj = AttitudeTrajectory::from_data(epochs, states, frame_a, frame_b).unwrap();
        traj.set_interpolation_method(AttitudeInterpolationMethod::Lagrange { degree: 5 });

        let aem = AEM::from_attitude_trajectory(
            &traj,
            "SAT1",
            "2024-001A",
            "BRAHE",
            CCSDSTimeSystem::UTC,
        )
        .unwrap();
        assert_eq!(
            aem.segments[0].metadata.interpolation_method,
            Some(AEMInterpolationMethod::Lagrange)
        );
        assert_eq!(aem.segments[0].metadata.interpolation_degree, Some(5));
        aem.segments[0].metadata.validate().unwrap();

        let round_tripped = AttitudeTrajectory::try_from(&aem).unwrap();
        assert_eq!(
            round_tripped.interpolation_method,
            AttitudeInterpolationMethod::Lagrange { degree: 5 }
        );
    }

    #[test]
    #[parallel]
    fn test_aem_trajectory_round_trip_interpolation_method_linear() {
        let frame_a = ReferenceFrame::from(CelestialFrame::EME2000);
        let frame_b = ReferenceFrame::from(BodyFrame::SCBody(None));
        let t0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        let epochs = vec![t0, t0 + 60.0];
        let states = vec![
            AttitudeState::new(Quaternion::new(1.0, 0.0, 0.0, 0.0)),
            AttitudeState::new(Quaternion::new(0.9998, 0.0, 0.0, 0.0196)),
        ];
        let mut traj = AttitudeTrajectory::from_data(epochs, states, frame_a, frame_b).unwrap();
        traj.set_interpolation_method(AttitudeInterpolationMethod::Linear);

        let aem = AEM::from_attitude_trajectory(
            &traj,
            "SAT1",
            "2024-001A",
            "BRAHE",
            CCSDSTimeSystem::UTC,
        )
        .unwrap();
        assert_eq!(
            aem.segments[0].metadata.interpolation_method,
            Some(AEMInterpolationMethod::Linear)
        );
        assert_eq!(aem.segments[0].metadata.interpolation_degree, Some(1));
        aem.segments[0].metadata.validate().unwrap();

        let round_tripped = AttitudeTrajectory::try_from(&aem).unwrap();
        assert_eq!(
            round_tripped.interpolation_method,
            AttitudeInterpolationMethod::Linear
        );
    }

    #[test]
    #[parallel]
    fn test_aem_from_attitude_trajectory_without_rates() {
        let frame_a = ReferenceFrame::from(CelestialFrame::EME2000);
        let frame_b = ReferenceFrame::from(BodyFrame::SCBody(None));
        let t0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        let epochs = vec![t0, t0 + 60.0];
        let states = vec![
            AttitudeState::new(Quaternion::new(1.0, 0.0, 0.0, 0.0)),
            AttitudeState::new(Quaternion::new(0.9998, 0.0, 0.0, 0.0196)),
        ];
        let traj = AttitudeTrajectory::from_data(epochs, states, frame_a, frame_b).unwrap();

        let aem = AEM::from_attitude_trajectory(
            &traj,
            "SAT1",
            "2024-001A",
            "BRAHE",
            CCSDSTimeSystem::UTC,
        )
        .unwrap();
        assert_eq!(
            aem.segments[0].metadata.attitude_type,
            AEMAttitudeType::Quaternion
        );
        assert!(aem.segments[0].metadata.angvel_frame.is_none());
    }

    #[test]
    #[parallel]
    fn test_aem_from_attitude_trajectory_unmappable_time_system_errors() {
        let frame_a = ReferenceFrame::from(CelestialFrame::EME2000);
        let frame_b = ReferenceFrame::from(BodyFrame::SCBody(None));
        let t0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        let epochs = vec![t0, t0 + 60.0];
        let states = vec![
            AttitudeState::new(Quaternion::new(1.0, 0.0, 0.0, 0.0)),
            AttitudeState::new(Quaternion::new(0.9998, 0.0, 0.0, 0.0196)),
        ];
        let traj = AttitudeTrajectory::from_data(epochs, states, frame_a, frame_b).unwrap();

        let err = AEM::from_attitude_trajectory(
            &traj,
            "SAT1",
            "2024-001A",
            "BRAHE",
            CCSDSTimeSystem::MET,
        )
        .unwrap_err();
        assert!(err.to_string().contains("TIME_SYSTEM"));
    }

    #[test]
    #[parallel]
    fn test_aem_from_attitude_trajectory_empty_errors() {
        let frame_a = ReferenceFrame::from(CelestialFrame::EME2000);
        let frame_b = ReferenceFrame::from(BodyFrame::SCBody(None));
        let traj = AttitudeTrajectory::new(frame_a, frame_b);

        let result = AEM::from_attitude_trajectory(
            &traj,
            "SAT1",
            "2024-001A",
            "BRAHE",
            CCSDSTimeSystem::UTC,
        );
        assert!(result.is_err());
    }

    #[test]
    #[parallel]
    fn test_aem_segment_to_attitude_trajectory_out_of_range_index_errors() {
        let content = std::fs::read_to_string("test_assets/ccsds/aem/AEMExampleG4.txt").unwrap();
        let aem = AEM::from_str(&content).unwrap();

        let result = aem.segment_to_attitude_trajectory(aem.segments.len());
        assert!(result.is_err());
        let message = format!("{}", result.unwrap_err());
        assert!(message.contains("out of range"));
    }

    fn euler_angvel_metadata(
        ref_frame_a: ADMReferenceFrame,
        ref_frame_b: ADMReferenceFrame,
        t0: Epoch,
        t1: Epoch,
        attitude_type: AEMAttitudeType,
    ) -> AEMMetadata {
        let mut metadata = AEMMetadata::new(
            "SAT1",
            "2024-001A",
            ref_frame_a.clone(),
            ref_frame_b.clone(),
            CCSDSTimeSystem::UTC,
            t0,
            t1,
            attitude_type,
        )
        .with_euler_rot_seq(EulerAngleOrder::ZYX);
        if attitude_type == AEMAttitudeType::EulerAngleAngVel {
            metadata = metadata.with_angvel_frame(ref_frame_b);
        }
        metadata
    }

    #[test]
    #[parallel]
    fn test_aem_quaternion_derivative_conversion() {
        let ref_frame_a = ADMReferenceFrame::parse("EME2000");
        let ref_frame_b = ADMReferenceFrame::parse("SC_BODY_1");
        let t0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;

        let metadata = AEMMetadata::new(
            "SAT1",
            "2024-001A",
            ref_frame_a,
            ref_frame_b,
            CCSDSTimeSystem::UTC,
            t0,
            t1,
            AEMAttitudeType::QuaternionDerivative,
        );

        let quaternion = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let omega = Vector3::new(0.02, -0.01, 0.3);
        let derivative = crate::attitude::quaternion_derivative(&quaternion, omega);

        let mut segment = AEMSegment::new(metadata);
        segment
            .push_state(AEMAttitudeState {
                epoch: t0,
                data: AEMAttitudeData::QuaternionDerivative {
                    quaternion,
                    derivative,
                },
            })
            .unwrap();
        let mut aem = AEM::new("BRAHE");
        aem.push_segment(segment);

        let traj = aem.segment_to_attitude_trajectory(0).unwrap();
        let recovered_omega = traj.state_at_idx(0).unwrap().angular_velocity.unwrap();
        assert_abs_diff_eq!(recovered_omega[0], omega[0], epsilon = 1e-9);
        assert_abs_diff_eq!(recovered_omega[1], omega[1], epsilon = 1e-9);
        assert_abs_diff_eq!(recovered_omega[2], omega[2], epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_aem_euler_angle_conversion() {
        let ref_frame_a = ADMReferenceFrame::parse("EME2000");
        let ref_frame_b = ADMReferenceFrame::parse("SC_BODY_1");
        let t0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let metadata = euler_angvel_metadata(
            ref_frame_a,
            ref_frame_b,
            t0,
            t1,
            AEMAttitudeType::EulerAngle,
        );

        let angles = EulerAngle::new(EulerAngleOrder::ZYX, 0.3, -0.2, 0.1, AngleFormat::Radians);
        let mut segment = AEMSegment::new(metadata);
        segment
            .push_state(AEMAttitudeState {
                epoch: t0,
                data: AEMAttitudeData::EulerAngle { angles },
            })
            .unwrap();
        let mut aem = AEM::new("BRAHE");
        aem.push_segment(segment);

        let traj = aem.segment_to_attitude_trajectory(0).unwrap();
        assert!(!traj.has_rates());
        let expected = Quaternion::from_euler_angle(angles);
        assert_eq!(traj.state_at_idx(0).unwrap().quaternion, expected);
    }

    #[test]
    #[parallel]
    fn test_aem_euler_angle_derivative_conversion() {
        let ref_frame_a = ADMReferenceFrame::parse("EME2000");
        let ref_frame_b = ADMReferenceFrame::parse("SC_BODY_1");
        let t0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let metadata = euler_angvel_metadata(
            ref_frame_a,
            ref_frame_b,
            t0,
            t1,
            AEMAttitudeType::EulerAngleDerivative,
        );

        let angles = EulerAngle::new(EulerAngleOrder::ZYX, 0.3, -0.2, 0.1, AngleFormat::Radians);
        let rates = Vector3::new(0.01, 0.02, -0.03);
        let mut segment = AEMSegment::new(metadata);
        segment
            .push_state(AEMAttitudeState {
                epoch: t0,
                data: AEMAttitudeData::EulerAngleDerivative { angles, rates },
            })
            .unwrap();
        let mut aem = AEM::new("BRAHE");
        aem.push_segment(segment);

        let traj = aem.segment_to_attitude_trajectory(0).unwrap();
        let expected_omega = euler_rates_to_angular_velocity(&angles, rates);
        let recovered_omega = traj.state_at_idx(0).unwrap().angular_velocity.unwrap();
        assert_abs_diff_eq!(recovered_omega[0], expected_omega[0], epsilon = 1e-12);
        assert_abs_diff_eq!(recovered_omega[1], expected_omega[1], epsilon = 1e-12);
        assert_abs_diff_eq!(recovered_omega[2], expected_omega[2], epsilon = 1e-12);
    }

    #[test]
    #[parallel]
    fn test_aem_euler_angle_angvel_conversion() {
        let ref_frame_a = ADMReferenceFrame::parse("EME2000");
        let ref_frame_b = ADMReferenceFrame::parse("SC_BODY_1");
        let t0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let metadata = euler_angvel_metadata(
            ref_frame_a,
            ref_frame_b,
            t0,
            t1,
            AEMAttitudeType::EulerAngleAngVel,
        );

        let angles = EulerAngle::new(EulerAngleOrder::ZYX, 0.3, -0.2, 0.1, AngleFormat::Radians);
        let angular_velocity = Vector3::new(0.001, 0.002, 0.003);
        let mut segment = AEMSegment::new(metadata);
        segment
            .push_state(AEMAttitudeState {
                epoch: t0,
                data: AEMAttitudeData::EulerAngleAngVel {
                    angles,
                    angular_velocity,
                },
            })
            .unwrap();
        let mut aem = AEM::new("BRAHE");
        aem.push_segment(segment);

        let traj = aem.segment_to_attitude_trajectory(0).unwrap();
        // ANGVEL_FRAME is REF_FRAME_B, so no re-expression: the stored rate
        // must equal the raw AEM value.
        let recovered_omega = traj.state_at_idx(0).unwrap().angular_velocity.unwrap();
        assert_eq!(recovered_omega, angular_velocity);
    }

    #[test]
    #[parallel]
    fn test_aem_spin_nutation_variants_conversion_errors() {
        let ref_frame_a = ADMReferenceFrame::parse("EME2000");
        let ref_frame_b = ADMReferenceFrame::parse("SC_BODY_1");
        let t0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;

        let mut metadata_nutation = AEMMetadata::new(
            "SAT1",
            "2024-001A",
            ref_frame_a.clone(),
            ref_frame_b.clone(),
            CCSDSTimeSystem::UTC,
            t0,
            t1,
            AEMAttitudeType::SpinNutation,
        );
        metadata_nutation = metadata_nutation.with_center_name("EARTH");
        let mut segment = AEMSegment::new(metadata_nutation);
        segment
            .push_state(AEMAttitudeState {
                epoch: t0,
                data: AEMAttitudeData::SpinNutation {
                    spin_alpha: 0.1,
                    spin_delta: 0.2,
                    spin_angle: 0.3,
                    spin_angle_vel: 0.4,
                    nutation: 0.05,
                    nutation_period: 120.0,
                    nutation_phase: 0.06,
                },
            })
            .unwrap();
        let mut aem = AEM::new("BRAHE");
        aem.push_segment(segment);
        let message = format!("{}", aem.segment_to_attitude_trajectory(0).unwrap_err());
        assert!(message.contains("SPIN/NUTATION"));

        let metadata_mom = AEMMetadata::new(
            "SAT1",
            "2024-001A",
            ref_frame_a,
            ref_frame_b,
            CCSDSTimeSystem::UTC,
            t0,
            t1,
            AEMAttitudeType::SpinNutationMom,
        );
        let mut segment = AEMSegment::new(metadata_mom);
        segment
            .push_state(AEMAttitudeState {
                epoch: t0,
                data: AEMAttitudeData::SpinNutationMom {
                    spin_alpha: 0.1,
                    spin_delta: 0.2,
                    spin_angle: 0.3,
                    spin_angle_vel: 0.4,
                    momentum_alpha: 0.07,
                    momentum_delta: 0.08,
                    nutation_vel: 0.09,
                },
            })
            .unwrap();
        let mut aem = AEM::new("BRAHE");
        aem.push_segment(segment);
        let message = format!("{}", aem.segment_to_attitude_trajectory(0).unwrap_err());
        assert!(message.contains("SPIN/NUTATION_MOM"));
    }

    #[test]
    #[parallel]
    fn test_aem_lagrange_interpolation_method_maps_with_degree() {
        let ref_frame_a = ADMReferenceFrame::parse("EME2000");
        let ref_frame_b = ADMReferenceFrame::parse("SC_BODY_1");
        let t0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;

        let metadata = AEMMetadata::new(
            "SAT1",
            "2024-001A",
            ref_frame_a,
            ref_frame_b,
            CCSDSTimeSystem::UTC,
            t0,
            t1,
            AEMAttitudeType::Quaternion,
        )
        .with_interpolation(AEMInterpolationMethod::Lagrange, Some(3));

        let mut segment = AEMSegment::new(metadata);
        for i in 0..4 {
            segment
                .push_state(AEMAttitudeState {
                    epoch: t0 + (i as f64) * 15.0,
                    data: AEMAttitudeData::Quaternion {
                        quaternion: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                    },
                })
                .unwrap();
        }
        let mut aem = AEM::new("BRAHE");
        aem.push_segment(segment);

        let traj = aem.segment_to_attitude_trajectory(0).unwrap();
        assert_eq!(
            traj.interpolation_method,
            AttitudeInterpolationMethod::Lagrange { degree: 3 }
        );
    }

    #[test]
    #[parallel]
    fn test_aem_lagrange_interpolation_degree_zero_errors() {
        let ref_frame_a = ADMReferenceFrame::parse("EME2000");
        let ref_frame_b = ADMReferenceFrame::parse("SC_BODY_1");
        let t0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;

        let metadata = AEMMetadata::new(
            "SAT1",
            "2024-001A",
            ref_frame_a,
            ref_frame_b,
            CCSDSTimeSystem::UTC,
            t0,
            t1,
            AEMAttitudeType::Quaternion,
        )
        .with_interpolation(AEMInterpolationMethod::Lagrange, Some(0));

        let mut segment = AEMSegment::new(metadata);
        segment
            .push_state(AEMAttitudeState {
                epoch: t0,
                data: AEMAttitudeData::Quaternion {
                    quaternion: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                },
            })
            .unwrap();
        let mut aem = AEM::new("BRAHE");
        aem.push_segment(segment);

        let err = aem.segment_to_attitude_trajectory(0).unwrap_err();
        assert!(err.to_string().contains("degree"));
    }

    #[test]
    #[parallel]
    fn test_aem_linear_interpolation_method_maps() {
        let ref_frame_a = ADMReferenceFrame::parse("EME2000");
        let ref_frame_b = ADMReferenceFrame::parse("SC_BODY_1");
        let t0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;

        let metadata = AEMMetadata::new(
            "SAT1",
            "2024-001A",
            ref_frame_a,
            ref_frame_b,
            CCSDSTimeSystem::UTC,
            t0,
            t1,
            AEMAttitudeType::Quaternion,
        )
        .with_interpolation(AEMInterpolationMethod::Linear, Some(1));

        let mut segment = AEMSegment::new(metadata);
        segment
            .push_state(AEMAttitudeState {
                epoch: t0,
                data: AEMAttitudeData::Quaternion {
                    quaternion: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                },
            })
            .unwrap();
        let mut aem = AEM::new("BRAHE");
        aem.push_segment(segment);

        let traj = aem.segment_to_attitude_trajectory(0).unwrap();
        assert_eq!(
            traj.interpolation_method,
            AttitudeInterpolationMethod::Linear
        );
    }

    /// Builds a single-segment AEM from AEMExampleG4's segment 1 (the
    /// quaternion segment that converts cleanly), so `register_for`'s
    /// single-segment requirement is satisfied by fixture data.
    fn g4_single_segment_aem() -> AEM {
        let content = std::fs::read_to_string("test_assets/ccsds/aem/AEMExampleG4.txt").unwrap();
        let parsed = AEM::from_str(&content).unwrap();
        let mut aem = AEM::new("BRAHE");
        aem.push_segment(parsed.segments[1].clone());
        aem
    }

    fn sc_body_1() -> ReferenceFrame {
        ReferenceFrame::body("SC", BodyFrame::SCBody(Some("1".to_string())))
    }

    #[test]
    #[serial]
    fn test_aem_register_for_registers_body_frame() {
        clear_frame_registry();
        clear_object_registry();

        let aem = g4_single_segment_aem();
        let traj = AttitudeTrajectory::try_from(&aem).unwrap();
        let epoch = traj.start_epoch().unwrap() + 30.0;

        aem.register_for("SC").unwrap();

        // REF_FRAME_A is the celestial endpoint, so the stored series
        // already rotates parent into body and is registered unchanged.
        let resolved =
            rotation_frame_to_frame(CelestialFrame::EME2000, sc_body_1(), epoch).unwrap();
        let expected = traj
            .quaternion(epoch)
            .unwrap()
            .to_rotation_matrix()
            .to_matrix();
        assert_abs_diff_eq!(resolved, expected, epsilon = 1e-12);

        clear_frame_registry();
    }

    #[test]
    #[serial]
    fn test_aem_register_for_inverts_when_celestial_is_frame_b() {
        clear_frame_registry();
        clear_object_registry();

        // Swap the endpoints so the celestial frame is REF_FRAME_B. The
        // stored quaternions then rotate body into parent, and registration
        // must invert them.
        let mut aem = g4_single_segment_aem();
        let metadata = &mut aem.segments[0].metadata;
        std::mem::swap(&mut metadata.ref_frame_a, &mut metadata.ref_frame_b);

        let traj = AttitudeTrajectory::try_from(&aem).unwrap();
        let epoch = traj.start_epoch().unwrap() + 30.0;

        aem.register_for("SC").unwrap();

        let resolved =
            rotation_frame_to_frame(CelestialFrame::EME2000, sc_body_1(), epoch).unwrap();
        let expected = traj
            .quaternion(epoch)
            .unwrap()
            .conjugate()
            .to_rotation_matrix()
            .to_matrix();
        assert_abs_diff_eq!(resolved, expected, epsilon = 1e-12);

        clear_frame_registry();
    }

    #[test]
    #[serial]
    fn test_aem_register_for_errors_without_celestial_endpoint() {
        clear_frame_registry();
        clear_object_registry();

        let mut aem = g4_single_segment_aem();
        aem.segments[0].metadata.ref_frame_a = ADMReferenceFrame::parse("SC_BODY_2");

        let err = aem.register_for("SC").unwrap_err().to_string();
        assert!(err.contains("celestial frame"), "{}", err);
        assert!(err.contains("body frame"), "{}", err);

        clear_frame_registry();
    }

    #[test]
    #[serial]
    fn test_aem_register_for_rejects_multi_segment() {
        clear_frame_registry();
        clear_object_registry();

        let content = std::fs::read_to_string("test_assets/ccsds/aem/AEMExampleG4.txt").unwrap();
        let aem = AEM::from_str(&content).unwrap();

        let err = aem.register_for("SC").unwrap_err().to_string();
        assert!(err.contains("exactly 1 segment"), "{}", err);

        clear_frame_registry();
    }
}
