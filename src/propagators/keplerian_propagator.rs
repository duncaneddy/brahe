/*!
 * Keplerian propagator implementation using the new architecture
 * with nalgebra vectors and clean interfaces
 */

use nalgebra::{DVector, Vector6};
use std::f64::consts::PI;

use crate::constants::AngleFormat;
use crate::constants::{DEG2RAD, RAD2DEG, RADIANS};
use crate::coordinates::{state_eci_to_koe, state_koe_to_eci};
use crate::frames::{
    CelestialFrame, ReferenceFrame, celestial_root, state_eci_to_ecef, state_frame_to_frame,
    state_gcrf_to_eme2000, state_gcrf_to_itrf,
};
use crate::orbits::keplerian::mean_motion;
use crate::propagators::traits::{SOrbitPropagator, SStatePropagator};
use crate::spice::NAIFId;
use crate::time::Epoch;
use crate::trajectories::DOrbitTrajectory;
use crate::trajectories::traits::{OrbitRepresentation, Trajectory, keplerian_center};
use crate::utils::state_providers::{DOrbitStateProvider, DStateProvider};
use crate::utils::{BraheError, Identifiable};

/// Convert Vector6 to DVector
#[inline]
fn svec6_to_dvec(sv: Vector6<f64>) -> DVector<f64> {
    DVector::from_iterator(6, sv.iter().copied())
}

/// Keplerian propagator for analytical two-body orbital motion
#[derive(Debug, Clone)]
pub struct KeplerianPropagator {
    /// Initial epoch
    pub initial_epoch: Epoch,

    /// Initial state vector in the original representation and frame
    pub initial_state: Vector6<f64>,

    /// Frame of the input/output states
    pub frame: ReferenceFrame,

    /// Representation of the input/output states
    pub representation: OrbitRepresentation,

    /// Angle format of the input/output states (None for Cartesian, Some for Keplerian)
    pub angle_format: Option<AngleFormat>,

    /// Step size in seconds for stepping operations
    pub step_size: f64,

    /// Accumulated trajectory of all propagated states, ordered by epoch
    pub trajectory: DOrbitTrajectory,

    /// Current propagation epoch
    epoch_current: Epoch,

    /// Current state vector in the original representation and frame
    state_current: Vector6<f64>,

    /// Internal osculating orbital elements (always in radians, ECI frame)
    internal_osculating_elements: Vector6<f64>,

    /// Mean motion in radians per second
    n: f64,

    /// Optional user-defined name for identification
    pub name: Option<String>,

    /// Optional user-defined numeric ID for identification
    pub id: Option<u64>,

    /// Optional UUID for unique identification
    pub uuid: Option<uuid::Uuid>,
}

impl KeplerianPropagator {
    /// Validate a frame / representation / angle-format combination.
    ///
    /// # Arguments
    /// * `frame` - Frame the states are expressed in
    /// * `representation` - Type of state representation
    /// * `angle_format` - Format for angular elements (`None` for Cartesian)
    ///
    /// # Returns
    /// `Ok(())` when the combination is supported, or an error describing the
    /// incompatibility.
    fn validate_format(
        frame: &ReferenceFrame,
        representation: OrbitRepresentation,
        angle_format: Option<AngleFormat>,
    ) -> Result<(), BraheError> {
        if representation == OrbitRepresentation::Keplerian && angle_format.is_none() {
            return Err(BraheError::PropagatorError(
                "Angle format must be specified for Keplerian elements".to_string(),
            ));
        }

        if representation == OrbitRepresentation::Cartesian && angle_format.is_some() {
            return Err(BraheError::PropagatorError(
                "Angle format should be None for Cartesian representation".to_string(),
            ));
        }

        if representation == OrbitRepresentation::Keplerian {
            keplerian_center(frame)?;
        }

        let center = celestial_root(frame)?.center_naif_id();
        if center != NAIFId::Earth.id() {
            return Err(BraheError::PropagatorError(format!(
                "KeplerianPropagator is Earth-only; frame {frame} is centered on {center}"
            )));
        }

        Ok(())
    }

    /// Create a new KeplerianPropagator from orbital elements or Cartesian state.
    /// The input state is assumed to be in the specified frame and representation. The
    /// input frame, representation, and angle format is assumed to be the desired output format
    /// for the propagator.
    ///
    /// If the output format needs to be changed, use the `with_output_format` method after initialization.
    ///
    /// The propagator models two-body motion about the Earth, so every frame it
    /// accepts must resolve to an Earth-centered frame. Cartesian states may be
    /// expressed in any such frame, including non-celestial ones such as an
    /// orbit-relative `RTN` frame anchored on a registered object. Keplerian
    /// elements additionally require an angle format and an Earth equatorial
    /// frame that admits orbital elements, `GCRF` (or another ICRF-aligned
    /// Earth-centered frame), `EME2000`, `MOD`, `TOD`, or `TEME`, and are read about the
    /// Earth in that frame's axes.
    ///
    /// The step size must be positive.
    ///
    /// # Arguments
    /// * `epoch` - Initial epoch
    /// * `state` - State vector (Keplerian elements or Cartesian position/velocity)
    /// * `frame` - Reference frame of the input and output states
    /// * `representation` - Type of state representation
    /// * `angle_format` - Format for angular elements (only for Keplerian)
    /// * `step_size` - Step size in seconds for propagation
    ///
    /// # Returns
    /// New KeplerianPropagator instance, or an error if:
    /// - Angle format is None for Keplerian representation
    /// - Keplerian elements are declared outside the frames that admit orbital
    ///   elements (`GCRF` or another ICRF-aligned Earth-centered frame,
    ///   `EME2000`, `MOD`, `TOD`, `TEME`)
    /// - Angle format is not None for Cartesian representation
    /// - The frame is not Earth-centered (Earth-only propagator), or cannot be
    ///   resolved because it is unbound or unregistered
    /// - Step size is not positive
    ///
    /// # Examples
    ///
    /// ```rust
    /// use brahe::propagators::KeplerianPropagator;
    /// use brahe::traits::OrbitRepresentation;
    /// use brahe::frames::CelestialFrame;
    /// use brahe::constants::AngleFormat;
    /// use brahe::eop::{StaticEOPProvider, set_global_eop_provider};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use nalgebra::Vector6;
    ///
    /// set_global_eop_provider(StaticEOPProvider::from_zero());
    ///
    /// let epc = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let oe = Vector6::new(6878e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    ///
    /// let prop = KeplerianPropagator::new(
    ///     epc,
    ///     oe,
    ///     CelestialFrame::GCRF,
    ///     OrbitRepresentation::Keplerian,
    ///     Some(AngleFormat::Degrees),
    ///     60.0,
    /// ).unwrap();
    /// ```
    pub fn new(
        epoch: Epoch,
        state: Vector6<f64>,
        frame: impl Into<ReferenceFrame>,
        representation: OrbitRepresentation,
        angle_format: Option<AngleFormat>,
        step_size: f64,
    ) -> Result<Self, BraheError> {
        let frame = frame.into();
        Self::validate_format(&frame, representation, angle_format)?;

        if step_size <= 0.0 {
            return Err(BraheError::PropagatorError(
                "Step size must be positive".to_string(),
            ));
        }

        // Unwrap angle_format for internal conversion (use RADIANS for Cartesian)
        let angle_format_unwrapped = angle_format.unwrap_or(RADIANS);

        // Convert input state to internal osculating elements in GCRF with radians
        let internal_elements = Self::convert_to_internal_osculating(
            epoch,
            state,
            &frame,
            representation,
            angle_format_unwrapped,
        )?;

        // Create initial trajectory (Keplerian propagator always uses 6D states)
        let mut trajectory = DOrbitTrajectory::new(6, frame.clone(), representation, angle_format)?;
        trajectory.add(epoch, svec6_to_dvec(state))?;

        let n = mean_motion(internal_elements[0], AngleFormat::Radians);

        let mut prop = Self {
            initial_epoch: epoch,
            initial_state: state,
            frame,
            representation,
            angle_format,
            internal_osculating_elements: internal_elements,
            trajectory,
            epoch_current: epoch,
            state_current: state,
            step_size,
            n,
            name: None,
            id: None,
            uuid: None,
        };

        // Auto-generate UUID and sync to trajectory
        let uuid = uuid::Uuid::now_v7();
        prop.uuid = Some(uuid);
        prop.trajectory.uuid = Some(uuid);

        Ok(prop)
    }

    /// Create a new KeplerianPropagator from Keplerian orbital elements
    ///
    /// # Arguments
    /// * `epoch` - Initial epoch
    /// * `elements` - Keplerian elements [a, e, i, RAAN, argp, mean_anomaly]
    /// * `angle_format` - Format of angular elements (Degrees or Radians)
    /// * `step_size` - Step size in seconds for propagation. Must be positive.
    ///
    /// # Returns
    /// * New KeplerianPropagator instance, or an error if step_size is not positive
    pub fn from_keplerian(
        epoch: Epoch,
        elements: Vector6<f64>,
        angle_format: AngleFormat,
        step_size: f64,
    ) -> Result<Self, BraheError> {
        Self::new(
            epoch,
            elements,
            CelestialFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(angle_format),
            step_size,
        )
    }

    /// Create a new KeplerianPropagator from Cartesian state
    ///
    /// # Arguments
    /// * `epoch` - Initial epoch
    /// * `state` - Cartesian state vector [x, y, z, vx, vy, vz]
    /// * `frame` - Frame of the input state (ECI or ECEF)
    /// * `step_size` - Step size in seconds for propagation. Must be positive.
    ///
    /// # Returns
    /// * New KeplerianPropagator instance, or an error if step_size is not positive
    pub fn from_eci(epoch: Epoch, state: Vector6<f64>, step_size: f64) -> Result<Self, BraheError> {
        Self::new(
            epoch,
            state,
            CelestialFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
            step_size,
        )
    }

    /// Create a new KeplerianPropagator from Cartesian state in ECEF frame
    ///
    /// # Arguments
    /// * `epoch` - Initial epoch
    /// * `state` - Cartesian state vector [x, y, z, vx, vy, vz] in ECEF frame
    /// * `step_size` - Step size in seconds for propagation. Must be positive.
    ///
    /// # Returns
    /// * New KeplerianPropagator instance, or an error if step_size is not positive
    pub fn from_ecef(
        epoch: Epoch,
        state: Vector6<f64>,
        step_size: f64,
    ) -> Result<Self, BraheError> {
        Self::new(
            epoch,
            state,
            CelestialFrame::ECEF,
            OrbitRepresentation::Cartesian,
            None,
            step_size,
        )
    }

    /// This method allows changing the output format of the propagator. It updates the frame, representation, and angle format.
    /// It also resets the trajectory to only contain the initial state converted to the new format. It should be
    /// used with initialization or after a reset to avoid inconsistencies.
    ///
    /// The frame, representation, and angle format must be compatible:
    /// * Keplerian representation requires a frame that admits orbital elements (`GCRF` or another ICRF-aligned Earth-centered frame, `EME2000`, `MOD`, `TOD`, `TEME`) and a specified angle format (Degrees or Radians)
    /// * Cartesian representation accepts any Earth-centered frame, but angle format must be None
    ///
    /// # Arguments
    /// * `frame` - Desired output frame
    /// * `representation` - Desired output representation
    /// * `angle_format` - Desired angle format (only for Keplerian)
    ///
    /// # Returns
    /// * Updated KeplerianPropagator instance, or an error if the format
    ///   combination is unsupported
    #[allow(dead_code)]
    fn with_output_format(
        mut self,
        frame: impl Into<ReferenceFrame>,
        representation: OrbitRepresentation,
        angle_format: Option<AngleFormat>,
    ) -> Result<Self, BraheError> {
        let frame = frame.into();
        Self::validate_format(&frame, representation, angle_format)?;

        self.frame = frame.clone();
        self.representation = representation;
        self.angle_format = angle_format;

        // Reset trajectory to initial state only, preserving identity
        let name = self.trajectory.get_name().map(|s| s.to_string());
        let uuid = self.trajectory.get_uuid();
        let id = self.trajectory.get_id();

        self.trajectory = DOrbitTrajectory::new(6, frame, representation, angle_format)?
            .with_identity(name.as_deref(), uuid, id);

        // Convert initial state to new format and add to trajectory
        let converted_state = self.convert_from_internal_osculating(
            self.initial_epoch,
            self.internal_osculating_elements,
        )?;
        self.trajectory
            .add(self.initial_epoch, svec6_to_dvec(converted_state))?;

        self.epoch_current = self.initial_epoch;
        self.state_current = converted_state;

        Ok(self)
    }

    /// Convert any state to internal osculating elements (GCRF, radians)
    ///
    /// # Arguments
    /// * `epoch` - Epoch of the input state
    /// * `state` - Input state vector
    /// * `frame` - Frame of the input state
    /// * `representation` - Representation of the input state
    /// * `angle_format` - Angle format of the input state (only for Keplerian)
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)`: Internal osculating elements in GCRF with radians
    /// * `Err(BraheError)`: If the frame conversion from `frame` to GCRF fails
    ///
    /// # Note
    /// Assumes that the input state is valid and consistent with the specified frame, representation, and angle format.
    /// This is checked during initialization.
    fn convert_to_internal_osculating(
        epoch: Epoch,
        state: Vector6<f64>,
        frame: &ReferenceFrame,
        representation: OrbitRepresentation,
        angle_format: AngleFormat,
    ) -> Result<Vector6<f64>, BraheError> {
        match representation {
            OrbitRepresentation::Cartesian => {
                let x_gcrf =
                    state_frame_to_frame(frame.clone(), CelestialFrame::GCRF, epoch, state)?;
                Ok(state_eci_to_koe(x_gcrf, AngleFormat::Radians))
            }
            OrbitRepresentation::Keplerian => {
                let mut elements = state;
                if angle_format == AngleFormat::Degrees {
                    // Convert angles from degrees to radians (i, RAAN, argp, mean_anomaly)
                    for i in 2..6 {
                        elements[i] *= DEG2RAD;
                    }
                }

                if *frame == CelestialFrame::GCRF {
                    return Ok(elements);
                }

                // Elements about the Earth in another inertial frame's axes:
                // realize them as a Cartesian state and rotate into GCRF.
                let x_frame = state_koe_to_eci(elements, AngleFormat::Radians);
                let x_gcrf =
                    state_frame_to_frame(frame.clone(), CelestialFrame::GCRF, epoch, x_frame)?;
                Ok(state_eci_to_koe(x_gcrf, AngleFormat::Radians))
            }
        }
    }

    /// Convert internal osculating elements back to the output state format
    ///
    /// # Arguments
    /// * `epoch` - Epoch of the internal elements
    /// * `internal_elements` - Internal osculating elements (GCRF, radians)
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)`: State in the propagator's output frame, representation and angle format
    /// * `Err(BraheError)`: If the frame conversion from GCRF to the output frame fails
    fn convert_from_internal_osculating(
        &self,
        epoch: Epoch,
        internal_elements: Vector6<f64>,
    ) -> Result<Vector6<f64>, BraheError> {
        match self.representation {
            OrbitRepresentation::Cartesian => {
                let x_gcrf = state_koe_to_eci(internal_elements, AngleFormat::Radians);
                state_frame_to_frame(CelestialFrame::GCRF, self.frame.clone(), epoch, x_gcrf)
            }
            OrbitRepresentation::Keplerian => {
                // For Keplerian, angle_format is guaranteed to be Some() by validation
                let format = self.angle_format.unwrap();

                let mut elements = if self.frame == CelestialFrame::GCRF {
                    internal_elements
                } else {
                    // Elements about the Earth in another inertial frame's
                    // axes: realize them in GCRF and rotate before converting.
                    let x_gcrf = state_koe_to_eci(internal_elements, AngleFormat::Radians);
                    let x_frame = state_frame_to_frame(
                        CelestialFrame::GCRF,
                        self.frame.clone(),
                        epoch,
                        x_gcrf,
                    )?;
                    state_eci_to_koe(x_frame, AngleFormat::Radians)
                };

                if format == AngleFormat::Degrees {
                    // Convert angles from radians to degrees (i, RAAN, argp, mean_anomaly)
                    for i in 2..6 {
                        elements[i] *= RAD2DEG;
                    }
                }

                Ok(elements)
            }
        }
    }

    /// Propagate internal Keplerian elements to a target epoch
    ///
    /// # Arguments
    /// * `target_epoch` - Epoch to which to propagate
    ///
    /// # Returns
    /// * New osculating elements at the target epoch (always in radians, ECI)
    fn propagate_internal(&self, target_epoch: Epoch) -> Vector6<f64> {
        let dt = target_epoch - self.initial_epoch;

        // Use internal osculating elements (always in radians, ECI)
        let a = self.internal_osculating_elements[0]; // Semi-major axis (m)
        let e = self.internal_osculating_elements[1]; // Eccentricity
        let i = self.internal_osculating_elements[2]; // Inclination (rad)
        let raan = self.internal_osculating_elements[3]; // Right Ascension of Ascending Node (rad)
        let argp = self.internal_osculating_elements[4]; // Argument of perigee (rad)
        let m0 = self.internal_osculating_elements[5]; // Initial mean anomaly (rad)

        // Propagate mean anomaly and normalize to [0, 2π]
        let m = (m0 + self.n * dt) % (2.0 * PI);

        // Return new osculating elements
        Vector6::new(a, e, i, raan, argp, m)
    }
}

impl SStatePropagator for KeplerianPropagator {
    fn step_by(&mut self, step_size: f64) -> Result<(), BraheError> {
        let current_epoch = self.current_epoch();
        let target_epoch = current_epoch + step_size;
        let new_state = self.propagate_internal(target_epoch);

        // Convert back to original state format
        let state = self.convert_from_internal_osculating(target_epoch, new_state)?;

        self.trajectory.add(target_epoch, svec6_to_dvec(state))?;
        self.epoch_current = target_epoch;
        self.state_current = state;

        Ok(())
    }

    // Default implementation from trait is used for:
    // - step()
    // - step_past()
    // - propagate_steps()
    // - propagate_to()

    fn current_epoch(&self) -> Epoch {
        self.epoch_current
    }

    fn current_state(&self) -> Vector6<f64> {
        self.state_current
    }

    fn initial_epoch(&self) -> Epoch {
        self.initial_epoch
    }

    fn initial_state(&self) -> Vector6<f64> {
        self.initial_state
    }

    fn step_size(&self) -> f64 {
        self.step_size
    }

    fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }

    /// # Panics
    ///
    /// Panics if the initial state cannot be converted into the propagator's
    /// output frame. The trait method is infallible, and the only way the
    /// conversion can fail is if the output frame can no longer be resolved by
    /// the reference frame router, for example an orbit-relative frame whose
    /// anchor object was unregistered, or a body-fixed frame whose kernels are
    /// no longer loaded.
    fn reset(&mut self) {
        // Reset trajectory to initial state only, preserving identity
        let name = self.trajectory.get_name().map(|s| s.to_string());
        let uuid = self.trajectory.get_uuid();
        let id = self.trajectory.get_id();

        self.trajectory = DOrbitTrajectory::new(
            6,
            self.frame.clone(),
            self.representation,
            self.angle_format,
        )
        .expect("trajectory format validated at construction")
        .with_identity(name.as_deref(), uuid, id);

        // Convert initial state to new format and add to trajectory
        let converted_state = self
            .convert_from_internal_osculating(self.initial_epoch, self.internal_osculating_elements)
            .unwrap_or_else(|e| panic!("frame conversion to {} failed: {}", self.frame, e));
        self.trajectory
            .add(self.initial_epoch, svec6_to_dvec(converted_state))
            .expect("trajectory state validated at construction");

        self.epoch_current = self.initial_epoch;
        self.state_current = converted_state;
    }

    fn set_eviction_policy_max_size(&mut self, max_size: usize) -> Result<(), BraheError> {
        self.trajectory.set_eviction_policy_max_size(max_size)
    }

    fn set_eviction_policy_max_age(&mut self, max_age: f64) -> Result<(), BraheError> {
        self.trajectory.set_eviction_policy_max_age(max_age)
    }
}

impl SOrbitPropagator for KeplerianPropagator {
    fn set_initial_conditions(
        &mut self,
        epoch: Epoch,
        state: Vector6<f64>,
        frame: ReferenceFrame,
        representation: OrbitRepresentation,
        angle_format: Option<AngleFormat>,
    ) -> Result<(), BraheError> {
        Self::validate_format(&frame, representation, angle_format)?;

        // Unwrap angle_format for internal conversion (use RADIANS for Cartesian)
        let angle_format_unwrapped = angle_format.unwrap_or(RADIANS);

        // Recompute internal elements
        self.internal_osculating_elements = Self::convert_to_internal_osculating(
            epoch,
            state,
            &frame,
            representation,
            angle_format_unwrapped,
        )?;
        self.n = mean_motion(self.internal_osculating_elements[0], AngleFormat::Radians);

        // Update all state
        self.initial_epoch = epoch;
        self.initial_state = state;
        self.frame = frame.clone();
        self.representation = representation;
        self.angle_format = angle_format;

        // Reset trajectory to new initial conditions, preserving identity
        let name = self.trajectory.get_name().map(|s| s.to_string());
        let uuid = self.trajectory.get_uuid();
        let id = self.trajectory.get_id();

        self.trajectory = DOrbitTrajectory::new(6, frame, representation, angle_format)?
            .with_identity(name.as_deref(), uuid, id);
        self.trajectory.add(epoch, svec6_to_dvec(state))?;

        self.epoch_current = epoch;
        self.state_current = state;

        Ok(())
    }
}

impl Identifiable for KeplerianPropagator {
    fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self.trajectory = self.trajectory.with_name(name);
        self
    }

    fn with_uuid(mut self, uuid: uuid::Uuid) -> Self {
        self.uuid = Some(uuid);
        self.trajectory = self.trajectory.with_uuid(uuid);
        self
    }

    fn with_new_uuid(mut self) -> Self {
        self.uuid = Some(uuid::Uuid::now_v7());
        self.trajectory = self.trajectory.with_uuid(self.uuid.unwrap());
        self
    }

    fn with_id(mut self, id: u64) -> Self {
        self.id = Some(id);
        self.trajectory = self.trajectory.with_id(id);
        self
    }

    fn with_identity(
        mut self,
        name: Option<&str>,
        uuid: Option<uuid::Uuid>,
        id: Option<u64>,
    ) -> Self {
        self.name = name.map(|s| s.to_string());
        self.uuid = uuid;
        self.id = id;
        self.trajectory = self.trajectory.with_identity(name, uuid, id);
        self
    }

    fn set_identity(&mut self, name: Option<&str>, uuid: Option<uuid::Uuid>, id: Option<u64>) {
        self.name = name.map(|s| s.to_string());
        self.uuid = uuid;
        self.id = id;
        self.trajectory.set_identity(name, uuid, id);
    }

    fn set_id(&mut self, id: Option<u64>) {
        self.id = id;
        self.trajectory.set_id(id);
    }

    fn set_name(&mut self, name: Option<&str>) {
        self.name = name.map(|s| s.to_string());
        self.trajectory.set_name(name);
    }

    fn generate_uuid(&mut self) {
        self.uuid = Some(uuid::Uuid::now_v7());
        self.trajectory.generate_uuid();
    }

    fn get_id(&self) -> Option<u64> {
        self.id
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_uuid(&self) -> Option<uuid::Uuid> {
        self.uuid
    }
}

impl DStateProvider for KeplerianPropagator {
    fn state(&self, epoch: Epoch) -> Result<DVector<f64>, BraheError> {
        // Reuse existing internal propagation logic
        let internal_state = self.propagate_internal(epoch);
        let sv = self.convert_from_internal_osculating(epoch, internal_state)?;
        Ok(svec6_to_dvec(sv))
    }

    fn state_dim(&self) -> usize {
        6
    }

    // states() uses default implementation from trait
}

impl DOrbitStateProvider for KeplerianPropagator {
    /// Earth-centered propagator: the body-centered inertial frame is GCRF.
    fn state_bci(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        self.state_gcrf(epoch)
    }

    /// Earth-centered propagator: the body-centered body-fixed frame is ITRF.
    fn state_bcbf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        self.state_itrf(epoch)
    }

    /// Converts the propagated state from GCRF (this propagator's central
    /// body's inertial frame) into `frame` via the reference frame router.
    fn state_in_frame(
        &self,
        frame: crate::frames::CelestialFrame,
        epoch: Epoch,
    ) -> Result<Vector6<f64>, BraheError> {
        let x_gcrf = self.state_gcrf(epoch)?;
        crate::frames::state_frame_to_frame(
            crate::frames::CelestialFrame::GCRF,
            frame,
            epoch,
            x_gcrf,
        )
    }

    fn state_eci(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        self.state_gcrf(epoch)
    }

    fn state_ecef(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        let state_eci = self.state_eci(epoch)?;
        Ok(state_eci_to_ecef(epoch, state_eci))
    }

    /// The internal osculating elements are held in GCRF, so this is the
    /// propagator's native output and does not depend on the configured output
    /// frame.
    fn state_gcrf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        let internal_state = self.propagate_internal(epoch);
        // Always convert to Cartesian for DOrbitStateProvider methods
        Ok(state_koe_to_eci(internal_state, AngleFormat::Radians))
    }

    fn state_itrf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        let state_gcrf = self.state_gcrf(epoch)?;
        Ok(state_gcrf_to_itrf(epoch, state_gcrf))
    }

    fn state_eme2000(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        let state_gcrf = self.state_gcrf(epoch)?;
        Ok(state_gcrf_to_eme2000(state_gcrf))
    }

    fn state_koe_osc(
        &self,
        epoch: Epoch,
        angle_format: AngleFormat,
    ) -> Result<Vector6<f64>, BraheError> {
        let state_eci = self.state_eci(epoch)?;
        let mut elements = state_eci_to_koe(state_eci, AngleFormat::Radians);

        if angle_format == AngleFormat::Degrees {
            elements[2] *= RAD2DEG; // i
            elements[3] *= RAD2DEG; // RAAN
            elements[4] *= RAD2DEG; // arg periapsis
            elements[5] *= RAD2DEG; // anomaly
        }

        Ok(elements)
    }

    // All batch methods (states_eci, states_ecef, etc.) use default implementations
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::DEGREES;
    use crate::coordinates::state_eci_to_koe;
    use crate::frames::object_registry::FnProvider;
    use crate::frames::{
        CelestialFrame, DStateAdapter, ReferenceFrame, clear_object_registry, register_object,
        state_ecef_to_eci, state_eme2000_to_gcrf, state_gcrf_to_mod, state_gcrf_to_teme,
        state_gcrf_to_tod, state_itrf_to_gcrf,
    };
    use crate::orbits::keplerian::orbital_period;
    use crate::time::{Epoch, TimeSystem};
    use crate::utils::testing::setup_global_test_eop;
    use approx::assert_abs_diff_eq;
    use serial_test::{parallel, serial};

    // Test data constants
    const TEST_EPOCH_JD: f64 = 2451545.0;

    fn create_test_elements() -> Vector6<f64> {
        Vector6::new(7000e3, 0.01, 97.8, 15.0, 45.0, 60.0)
    }

    fn create_circular_elements() -> Vector6<f64> {
        Vector6::new(7000e3, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    fn create_cartesian_state() -> Vector6<f64> {
        let a = 7000e3; // Semi-major axis in meters
        let e = 0.01; // Eccentricity
        let i = 97.8; // Inclination
        let raan = 15.0; // Right Ascension of Ascending Node
        let argp = 45.0; // Argument of perigee
        let ma = 60.0; // Mean anomaly

        state_koe_to_eci(Vector6::new(a, e, i, raan, argp, ma), DEGREES)
    }

    // KeplerianPropagator Method Tests

    #[test]
    #[parallel]
    fn test_keplerianpropagator_new() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator = KeplerianPropagator::new(
            epoch,
            elements,
            CelestialFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(RADIANS),
            60.0,
        )
        .unwrap();

        assert_eq!(propagator.initial_epoch(), epoch);
        assert_eq!(propagator.current_epoch(), epoch);
        assert_abs_diff_eq!(propagator.initial_state()[0], 7000e3, epsilon = 1.0);
        assert_abs_diff_eq!(propagator.initial_state()[1], 0.01, epsilon = 1e-10);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_new_rejects_bci_frame() {
        // KeplerianPropagator is Earth-only; frames centered on another body are rejected
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let state = create_cartesian_state();

        let result = KeplerianPropagator::new(
            epoch,
            state,
            CelestialFrame::LCI,
            OrbitRepresentation::Cartesian,
            None,
            60.0,
        );

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("KeplerianPropagator is Earth-only")
        );
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_with_output_format_success() {
        // Changing the output format from radians to degrees resets the
        // trajectory to the initial state expressed in the new format.
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements(); // degrees in [i, raan, argp, M]

        let prop = KeplerianPropagator::new(
            epoch,
            elements,
            CelestialFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(DEGREES),
            60.0,
        )
        .unwrap();

        let prop = prop
            .with_output_format(
                CelestialFrame::ECI,
                OrbitRepresentation::Keplerian,
                Some(RADIANS),
            )
            .unwrap();

        assert_eq!(prop.angle_format, Some(RADIANS));
        // Inclination of 97.8 degrees expressed in radians
        assert_abs_diff_eq!(
            prop.current_state()[2],
            97.8_f64.to_radians(),
            epsilon = 1e-9
        );
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_with_output_format_invalid() {
        // Keplerian representation without an angle format is rejected
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let prop = KeplerianPropagator::new(
            epoch,
            elements,
            CelestialFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(DEGREES),
            60.0,
        )
        .unwrap();

        let result =
            prop.with_output_format(CelestialFrame::ECI, OrbitRepresentation::Keplerian, None);
        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn test_keplerianpropagator_new_errors_when_orbit_relative_provider_fails() {
        setup_global_test_eop();
        clear_object_registry();
        register_object(
            "CHIEF_FAIL",
            FnProvider(|_epoch| Err(BraheError::Error("provider unavailable".to_string()))),
            CelestialFrame::GCRF,
        )
        .unwrap();

        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let state = create_cartesian_state();
        let result = KeplerianPropagator::new(
            epoch,
            state,
            ReferenceFrame::RTN("CHIEF_FAIL"),
            OrbitRepresentation::Cartesian,
            None,
            60.0,
        );

        assert!(result.is_err());
        clear_object_registry();
    }

    #[test]
    #[serial]
    fn test_keplerianpropagator_with_output_format_errors_when_orbit_relative_provider_fails() {
        setup_global_test_eop();
        clear_object_registry();
        register_object(
            "CHIEF_FAIL",
            FnProvider(|_epoch| Err(BraheError::Error("provider unavailable".to_string()))),
            CelestialFrame::GCRF,
        )
        .unwrap();

        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let prop = KeplerianPropagator::from_eci(epoch, create_cartesian_state(), 60.0).unwrap();

        let result = prop.with_output_format(
            ReferenceFrame::RTN("CHIEF_FAIL"),
            OrbitRepresentation::Cartesian,
            None,
        );

        assert!(result.is_err());
        clear_object_registry();
    }

    #[test]
    #[serial]
    fn test_keplerianpropagator_set_initial_conditions_errors_when_orbit_relative_provider_fails() {
        setup_global_test_eop();
        clear_object_registry();
        register_object(
            "CHIEF_FAIL",
            FnProvider(|_epoch| Err(BraheError::Error("provider unavailable".to_string()))),
            CelestialFrame::GCRF,
        )
        .unwrap();

        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let mut prop =
            KeplerianPropagator::from_eci(epoch, create_cartesian_state(), 60.0).unwrap();

        let result = prop.set_initial_conditions(
            epoch,
            create_cartesian_state(),
            ReferenceFrame::RTN("CHIEF_FAIL"),
            OrbitRepresentation::Cartesian,
            None,
        );

        assert!(result.is_err());
        clear_object_registry();
    }

    #[test]
    #[should_panic(expected = "Angle format must be specified for Keplerian elements")]
    #[parallel]
    fn test_keplerianpropagator_new_invalid_angle_format() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        // This should panic because angle format is None for Keplerian
        let _propagator = KeplerianPropagator::new(
            epoch,
            elements,
            CelestialFrame::ECI,
            OrbitRepresentation::Keplerian,
            None,
            60.0,
        )
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "Keplerian element trajectories should be in an inertial frame")]
    #[parallel]
    fn test_keplerianpropagator_new_invalid_frame() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        // This should panic because Keplerian elements are not in an inertial frame
        let _propagator = KeplerianPropagator::new(
            epoch,
            elements,
            CelestialFrame::ECEF,
            OrbitRepresentation::Keplerian,
            Some(RADIANS),
            60.0,
        )
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "Angle format should be None for Cartesian representation")]
    #[parallel]
    fn test_keplerianpropagator_new_invalid_cartesian_angle_format() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let state = create_test_elements(); // Using elements for simplicity

        // This should panic because angle format is not None for Cartesian
        let _propagator = KeplerianPropagator::new(
            epoch,
            state,
            CelestialFrame::ECI,
            OrbitRepresentation::Cartesian,
            Some(RADIANS),
            60.0,
        )
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "Step size must be positive")]
    #[parallel]
    fn test_keplerianpropagator_new_invalid_step_size_neative() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        // This should panic because step size is not positive
        let _propagator = KeplerianPropagator::new(
            epoch,
            elements,
            CelestialFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(RADIANS),
            -10.0,
        )
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "Step size must be positive")]
    #[parallel]
    fn test_keplerianpropagator_new_invalid_step_size_zero() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        // This should panic because step size is not positive
        let _propagator = KeplerianPropagator::new(
            epoch,
            elements,
            CelestialFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(RADIANS),
            0.0,
        )
        .unwrap();
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_from_keplerian() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        assert_eq!(propagator.initial_epoch(), epoch);
        assert_eq!(propagator.step_size(), 60.0);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_from_eci() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let cartesian = create_cartesian_state();

        let propagator = KeplerianPropagator::from_eci(epoch, cartesian, 60.0).unwrap();

        assert_eq!(propagator.initial_epoch(), epoch);
        assert_eq!(propagator.step_size(), 60.0);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_from_ecef() {
        setup_global_test_eop();
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let cartesian = state_ecef_to_eci(epoch, create_cartesian_state());

        let propagator = KeplerianPropagator::from_ecef(epoch, cartesian, 60.0).unwrap();

        assert_eq!(propagator.initial_epoch(), epoch);
        assert_eq!(propagator.step_size(), 60.0);
        assert_eq!(propagator.frame, CelestialFrame::ECEF);
    }

    // OrbitPropagator Trait Tests

    #[test]
    #[parallel]
    fn test_keplerianpropagator_orbitpropagator_step() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        propagator.step().unwrap();

        let new_epoch = propagator.current_epoch();
        assert_eq!(new_epoch, epoch + 60.0);
        assert_eq!(propagator.trajectory.len(), 2);

        // Confirm all elements expect for mean anomaly are unchanged
        let new_state = propagator.current_state();
        for i in 0..5 {
            assert_abs_diff_eq!(new_state[i], elements[i], epsilon = 1e-6);
        }
        // Mean anomaly should have changed
        assert_ne!(new_state[5], elements[5]);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_orbitpropagator_step_by() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        propagator.step_by(120.0).unwrap();

        let new_epoch = propagator.current_epoch();
        assert_eq!(new_epoch, epoch + 120.0);

        // Confirm only 2 states in trajectory (initial + 1 step)
        assert_eq!(propagator.trajectory.len(), 2);

        // Confirm all elements expect for mean anomaly are unchanged
        let new_state = propagator.current_state();
        for i in 0..5 {
            assert_abs_diff_eq!(new_state[i], elements[i], epsilon = 1e-6);
        }
        // Mean anomaly should have changed
        assert_ne!(new_state[5], elements[5]);
    }

    #[test]
    #[parallel]
    fn test_keplerian_orbitpropagator_step_past() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        let target_epoch = epoch + 250.0;
        propagator.step_past(target_epoch).unwrap();

        let current_epoch = propagator.current_epoch();
        assert!(current_epoch > target_epoch);

        // Should have 6 steps: initial + 5 steps of 60s
        assert_eq!(propagator.trajectory.len(), 6);
        assert_eq!(current_epoch, epoch + 300.0);

        // Confirm all elements expect for mean anomaly are unchanged
        let new_state = propagator.current_state();
        for i in 0..5 {
            assert_abs_diff_eq!(new_state[i], elements[i], epsilon = 1e-6);
        }
        // Mean anomaly should have changed
        assert_ne!(new_state[5], elements[5]);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_orbitpropagator_propagate_steps() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        propagator.propagate_steps(5).unwrap();

        assert_eq!(propagator.trajectory.len(), 6); // Initial + 5 steps
        let new_epoch = propagator.current_epoch();
        assert_eq!(new_epoch, epoch + 300.0);

        // Confirm all elements expect for mean anomaly are unchanged
        let new_state = propagator.current_state();
        for i in 0..5 {
            assert_abs_diff_eq!(new_state[i], elements[i], epsilon = 1e-6);
        }
        // Mean anomaly should have changed
        assert_ne!(new_state[5], elements[5]);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_orbitpropagator_propagate_to() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        let target_epoch = epoch + 90.0;
        propagator.propagate_to(target_epoch).unwrap();

        let current_epoch = propagator.current_epoch();
        assert_eq!(current_epoch, target_epoch);

        // Should have 3 steps: initial + 1 step of 60s + 1 step of 30s
        assert_eq!(propagator.trajectory.len(), 3);

        // Confirm all elements expect for mean anomaly are unchanged
        let new_state = propagator.current_state();
        for i in 0..5 {
            assert_abs_diff_eq!(new_state[i], elements[i], epsilon = 1e-6);
        }
        // Mean anomaly should have changed
        assert_ne!(new_state[5], elements[5]);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_orbitpropagator_current_epoch() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        assert_eq!(propagator.current_epoch(), epoch);

        // step and check epoch advanced
        propagator.step().unwrap();
        assert_ne!(propagator.current_epoch(), epoch);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_orbitpropagator_current_state() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        // Initial state should match
        assert_eq!(propagator.current_state(), elements);

        // After step, should be different
        propagator.step().unwrap();
        let current_state = propagator.current_state();
        assert_ne!(current_state, elements);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_orbitpropagator_initial_epoch() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        assert_eq!(propagator.initial_epoch(), epoch);

        // Step and confirm initial epoch unchanged
        propagator.step().unwrap();
        assert_eq!(propagator.initial_epoch(), epoch);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_orbitpropagator_initial_state() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        assert_eq!(propagator.initial_state(), elements);

        // Step and confirm initial state unchanged
        propagator.step().unwrap();
        assert_eq!(propagator.initial_state(), elements);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_orbitpropagator_step_size() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        assert_eq!(propagator.step_size(), 60.0);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_orbitpropagator_set_step_size() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        // Confirm initial step size
        assert_eq!(propagator.step_size(), 60.0);

        // Change step size
        propagator.set_step_size(120.0);
        assert_eq!(propagator.step_size(), 120.0);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_orbitpropagator_reset() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        // Propagate forward
        propagator.propagate_steps(5).unwrap();
        assert_eq!(propagator.trajectory.len(), 6);

        // Reset
        propagator.reset();
        assert_eq!(propagator.trajectory.len(), 1);
        assert_eq!(propagator.current_epoch(), epoch);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_orbitpropagator_set_initial_conditions() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        // Set new initial conditions
        let new_epoch = Epoch::from_jd(TEST_EPOCH_JD + 1.0, TimeSystem::UTC);
        let new_elements = create_circular_elements();

        propagator
            .set_initial_conditions(
                new_epoch,
                new_elements,
                CelestialFrame::ECI.into(),
                OrbitRepresentation::Keplerian,
                Some(AngleFormat::Radians),
            )
            .unwrap();

        assert_eq!(propagator.initial_epoch(), new_epoch);
        assert_eq!(propagator.initial_state(), new_elements);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_orbitpropagator_set_eviction_policy_max_size() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        propagator.set_eviction_policy_max_size(5).unwrap();

        // Propagate 10 steps
        propagator.propagate_steps(10).unwrap();

        // Should only keep 5 states
        assert_eq!(propagator.trajectory.len(), 5);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_orbitpropagator_set_eviction_policy_max_age() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        // Set eviction policy - only keep states within 120 seconds of current
        let result = propagator.set_eviction_policy_max_age(120.0);
        assert!(result.is_ok());

        // Propagate several steps (10 * 60s = 600s total)
        propagator.propagate_steps(10).unwrap();

        // Should have evicted old states - should keep only last ~3 states (120s / 60s step)
        // Plus current state: 3 previous + current = 4 states max
        assert!(propagator.trajectory.len() <= 4);
        assert!(propagator.trajectory.len() > 0);
    }

    // StateProvider Trait Tests

    #[test]
    #[parallel]
    fn test_keplerianpropagator_analyticpropagator_state() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        let target_epoch = epoch + orbital_period(elements[0]);
        let state = propagator.state(target_epoch).unwrap();

        // State should be exactly the same as initial elements after one orbital period
        for i in 0..6 {
            assert_abs_diff_eq!(state[i], elements[i], epsilon = 1e-9);
        }
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_analyticpropagator_state_eci() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        let state = propagator
            .state_eci(epoch + orbital_period(elements[0]))
            .unwrap();

        // Should be Cartesian state in ECI
        assert!(state.norm() > 0.0);
        // Convert back to orbital elements and verify semi-major axis is preserved
        let computed_elements = state_eci_to_koe(state, DEGREES);

        // Confirm equality within small tolerance
        for i in 0..6 {
            assert_abs_diff_eq!(computed_elements[i], elements[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_analyticpropagator_state_ecef() {
        setup_global_test_eop();
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        let state = propagator
            .state_ecef(epoch + orbital_period(elements[0]))
            .unwrap();

        // Convert back into osculating elements via ECI
        let eci_state = state_ecef_to_eci(epoch + orbital_period(elements[0]), state);
        let computed_elements = state_eci_to_koe(eci_state, DEGREES);

        // Confirm equality within small tolerance
        for i in 0..6 {
            assert_abs_diff_eq!(computed_elements[i], elements[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_analyticpropagator_state_itrf() {
        setup_global_test_eop();
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        let state = propagator
            .state_itrf(epoch + orbital_period(elements[0]))
            .unwrap();

        // Convert back into osculating elements via ECI
        let eci_state = state_itrf_to_gcrf(epoch + orbital_period(elements[0]), state);
        let computed_elements = state_eci_to_koe(eci_state, DEGREES);

        // Confirm equality within small tolerance
        for i in 0..6 {
            assert_abs_diff_eq!(computed_elements[i], elements[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_analyticpropagator_state_gcrf() {
        setup_global_test_eop();
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        let state = propagator
            .state_gcrf(epoch + orbital_period(elements[0]))
            .unwrap();

        // Convert back into osculating elements (GCRF is inertial, direct conversion)
        let computed_elements = state_eci_to_koe(state, DEGREES);

        // Confirm equality within small tolerance
        for i in 0..6 {
            assert_abs_diff_eq!(computed_elements[i], elements[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_analyticpropagator_state_eme2000() {
        setup_global_test_eop();
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        let state = propagator
            .state_eme2000(epoch + orbital_period(elements[0]))
            .unwrap();

        // Convert back into osculating elements via GCRF
        let gcrf_state = state_eme2000_to_gcrf(state);
        let computed_elements = state_eci_to_koe(gcrf_state, DEGREES);

        // Confirm equality within small tolerance
        for i in 0..6 {
            assert_abs_diff_eq!(computed_elements[i], elements[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_analyticpropagator_state_koe_osc() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        let osc_elements = propagator
            .state_koe_osc(epoch + orbital_period(elements[0]), AngleFormat::Degrees)
            .unwrap();

        // Should match initial elements within small tolerance
        for i in 0..6 {
            assert_abs_diff_eq!(osc_elements[i], elements[i], epsilon = 1e-6);
        }

        // Now test with radians to degrees conversion
        let osc_elements_rad = propagator
            .state_koe_osc(epoch + orbital_period(elements[0]), AngleFormat::Radians)
            .unwrap();
        for i in 0..2 {
            assert_abs_diff_eq!(osc_elements_rad[i], elements[i], epsilon = 1e-6);
        }
        for i in 2..6 {
            assert_abs_diff_eq!(osc_elements_rad[i] * RAD2DEG, elements[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_analyticpropagator_states() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        let epochs = vec![
            epoch,
            epoch + orbital_period(elements[0]),
            epoch + 2.0 * orbital_period(elements[0]),
        ];

        let traj = propagator.states(&epochs).unwrap();
        assert_eq!(traj.len(), 3);

        // Confirm all elements remain unchanged within small tolerance
        for state in &traj {
            for i in 0..6 {
                assert_abs_diff_eq!(state[i], elements[i], epsilon = 1e-6);
            }
        }
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_analyticpropagator_states_eci() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        let epochs = vec![
            epoch,
            epoch + orbital_period(elements[0]),
            epoch + 2.0 * orbital_period(elements[0]),
        ];

        let states = propagator.states_eci(&epochs).unwrap();
        assert_eq!(states.len(), 3);
        // Verify states convert back to original elements within small tolerance
        for state in &states {
            let computed_elements = state_eci_to_koe(*state, DEGREES);
            for i in 0..6 {
                assert_abs_diff_eq!(computed_elements[i], elements[i], epsilon = 1e-6);
            }
        }
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_analyticpropagator_states_ecef() {
        setup_global_test_eop();
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        let epochs = vec![
            epoch,
            epoch + orbital_period(elements[0]),
            epoch + 2.0 * orbital_period(elements[0]),
        ];

        let states = propagator.states_ecef(&epochs).unwrap();
        assert_eq!(states.len(), 3);
        // Verify states convert back to original elements within small tolerance
        for (i, state) in states.iter().enumerate() {
            let eci_state = state_ecef_to_eci(epochs[i], *state);
            let computed_elements = state_eci_to_koe(eci_state, DEGREES);
            for j in 0..6 {
                assert_abs_diff_eq!(computed_elements[j], elements[j], epsilon = 1e-6);
            }
        }
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_analyticpropagator_states_bci() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();
        let propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        let epochs = vec![epoch, epoch + 60.0, epoch + 120.0];
        let states = propagator.states_bci(&epochs).unwrap();
        assert_eq!(states.len(), 3);
        for (i, epc) in epochs.iter().enumerate() {
            let single = propagator.state_bci(*epc).unwrap();
            for j in 0..6 {
                assert_eq!(states[i][j], single[j]);
            }
        }
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_analyticpropagator_states_bcbf() {
        setup_global_test_eop();
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();
        let propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        let epochs = vec![epoch, epoch + 60.0, epoch + 120.0];
        let states = propagator.states_bcbf(&epochs).unwrap();
        assert_eq!(states.len(), 3);
        for (i, epc) in epochs.iter().enumerate() {
            let single = propagator.state_bcbf(*epc).unwrap();
            for j in 0..6 {
                assert_eq!(states[i][j], single[j]);
            }
        }
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_analyticpropagator_states_in_frame() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();
        let propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        let epochs = vec![epoch, epoch + 60.0, epoch + 120.0];
        let states = propagator
            .states_in_frame(CelestialFrame::GCRF, &epochs)
            .unwrap();
        assert_eq!(states.len(), 3);
        for (i, epc) in epochs.iter().enumerate() {
            let single = propagator
                .state_in_frame(CelestialFrame::GCRF, *epc)
                .unwrap();
            for j in 0..6 {
                assert_eq!(states[i][j], single[j]);
            }
        }
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_analyticpropagator_states_koe() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        let epochs = vec![
            epoch,
            epoch + orbital_period(elements[0]),
            epoch + 2.0 * orbital_period(elements[0]),
        ];

        let traj = propagator
            .states_koe_osc(&epochs, AngleFormat::Degrees)
            .unwrap();
        assert_eq!(traj.len(), 3);

        // Confirm all elements remain unchanged within small tolerance
        for state in &traj {
            for i in 0..6 {
                assert_abs_diff_eq!(state[i], elements[i], epsilon = 1e-6);
            }
        }

        // Repeat with radians output
        let traj_rad = propagator
            .states_koe_osc(&epochs, AngleFormat::Radians)
            .unwrap();
        assert_eq!(traj_rad.len(), 3);

        for state in &traj_rad {
            for i in 0..2 {
                assert_abs_diff_eq!(state[i], elements[i], epsilon = 1e-6);
            }
            for i in 2..6 {
                assert_abs_diff_eq!(state[i] * RAD2DEG, elements[i], epsilon = 1e-6);
            }
        }
    }

    // Identifiable Trait Tests

    #[test]
    #[parallel]
    fn test_keplerianpropagator_identifiable_with_name() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let prop = KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
            .unwrap()
            .with_name("My Orbit");

        assert_eq!(prop.get_name(), Some("My Orbit"));
        assert_eq!(prop.get_id(), None);
        assert!(prop.get_uuid().is_some()); // Auto-generated in constructor
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_identifiable_with_id() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let prop = KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
            .unwrap()
            .with_id(54321);

        assert_eq!(prop.get_id(), Some(54321));
        assert_eq!(prop.get_name(), None);
        assert!(prop.get_uuid().is_some()); // Auto-generated in constructor
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_identifiable_with_uuid() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();
        let test_uuid = uuid::Uuid::now_v7();

        let prop = KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
            .unwrap()
            .with_uuid(test_uuid);

        assert_eq!(prop.get_uuid(), Some(test_uuid));
        assert_eq!(prop.get_name(), None);
        assert_eq!(prop.get_id(), None);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_identifiable_with_new_uuid() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let prop = KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
            .unwrap()
            .with_new_uuid();

        assert!(prop.get_uuid().is_some());
        assert_eq!(prop.get_name(), None);
        assert_eq!(prop.get_id(), None);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_identifiable_with_identity() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();
        let test_uuid = uuid::Uuid::now_v7();

        let prop = KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
            .unwrap()
            .with_identity(Some("Orbit X"), Some(test_uuid), Some(888));

        assert_eq!(prop.get_name(), Some("Orbit X"));
        assert_eq!(prop.get_id(), Some(888));
        assert_eq!(prop.get_uuid(), Some(test_uuid));
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_identifiable_set_name() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut prop =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        prop.set_name(Some("Name 1"));
        assert_eq!(prop.get_name(), Some("Name 1"));

        prop.set_name(None);
        assert_eq!(prop.get_name(), None);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_identifiable_set_id() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut prop =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        prop.set_id(Some(100));
        assert_eq!(prop.get_id(), Some(100));

        prop.set_id(None);
        assert_eq!(prop.get_id(), None);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_identifiable_generate_uuid() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut prop =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        // UUID is auto-generated in constructor
        let initial_uuid = prop.get_uuid();
        assert!(initial_uuid.is_some());

        prop.generate_uuid();
        let uuid1 = prop.get_uuid();
        assert!(uuid1.is_some());

        // Generate another UUID and verify it's different
        prop.generate_uuid();
        let uuid2 = prop.get_uuid();
        assert!(uuid2.is_some());
        assert_ne!(uuid1, uuid2);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_identifiable_set_identity() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();
        let test_uuid = uuid::Uuid::now_v7();

        let mut prop =
            KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
                .unwrap();

        prop.set_identity(Some("ID Test"), Some(test_uuid), Some(555));

        assert_eq!(prop.get_name(), Some("ID Test"));
        assert_eq!(prop.get_id(), Some(555));
        assert_eq!(prop.get_uuid(), Some(test_uuid));

        // Clear all
        prop.set_identity(None, None, None);
        assert_eq!(prop.get_name(), None);
        assert_eq!(prop.get_id(), None);
        assert_eq!(prop.get_uuid(), None);
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_identifiable_chaining() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();
        let test_uuid = uuid::Uuid::now_v7();

        let prop = KeplerianPropagator::from_keplerian(epoch, elements, AngleFormat::Degrees, 60.0)
            .unwrap()
            .with_name("Chained Orbit")
            .with_id(999)
            .with_uuid(test_uuid);

        assert_eq!(prop.get_name(), Some("Chained Orbit"));
        assert_eq!(prop.get_id(), Some(999));
        assert_eq!(prop.get_uuid(), Some(test_uuid));
    }

    #[test]
    #[serial]
    fn test_keplerianpropagator_output_in_tod_and_rtn() {
        setup_global_test_eop();
        clear_object_registry();

        let epc = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_gcrf = create_cartesian_state();

        let prop = KeplerianPropagator::new(
            epc,
            x_gcrf,
            CelestialFrame::GCRF,
            OrbitRepresentation::Cartesian,
            None,
            60.0,
        )
        .unwrap();

        // A true-of-date output format matches the direct GCRF -> TOD rotation.
        let tod = prop
            .clone()
            .with_output_format(CelestialFrame::TOD, OrbitRepresentation::Cartesian, None)
            .unwrap();
        let x_tod = tod.state(epc).unwrap();
        let expected = state_gcrf_to_tod(epc, x_gcrf);
        for k in 0..6 {
            assert_abs_diff_eq!(x_tod[k], expected[k], epsilon = 1e-6);
        }

        // The inertial accessors are independent of the output format.
        let x_back = tod.state_gcrf(epc).unwrap();
        for k in 0..6 {
            assert_abs_diff_eq!(x_back[k], x_gcrf[k], epsilon = 1e-6);
        }

        // An orbit-relative output frame anchored on the propagated orbit puts
        // the propagated state at the frame origin.
        let mut anchor = DOrbitTrajectory::new(
            6,
            CelestialFrame::GCRF,
            OrbitRepresentation::Cartesian,
            None,
        )
        .unwrap();
        for k in 0..5 {
            let epc_k = epc + 60.0 * k as f64;
            anchor
                .add(epc_k, svec6_to_dvec(prop.state_gcrf(epc_k).unwrap()))
                .unwrap();
        }
        register_object(
            "KEPLERIAN_RTN_ANCHOR",
            DStateAdapter::new(anchor).unwrap(),
            CelestialFrame::GCRF,
        )
        .unwrap();

        let rtn = prop
            .clone()
            .with_output_format(
                ReferenceFrame::RTN("KEPLERIAN_RTN_ANCHOR"),
                OrbitRepresentation::Cartesian,
                None,
            )
            .unwrap();
        let x_rtn = rtn.state(epc + 120.0).unwrap();
        for k in 0..3 {
            assert_abs_diff_eq!(x_rtn[k], 0.0, epsilon = 1e-6);
        }

        // Non-Earth frames are rejected: Earth-only propagator.
        assert!(
            prop.clone()
                .with_output_format(CelestialFrame::LCI, OrbitRepresentation::Cartesian, None)
                .is_err()
        );

        // Keplerian output is rejected in body-fixed frames.
        assert!(
            prop.clone()
                .with_output_format(
                    CelestialFrame::ITRF,
                    OrbitRepresentation::Keplerian,
                    Some(AngleFormat::Degrees),
                )
                .is_err()
        );

        clear_object_registry();
    }

    #[test]
    #[serial]
    fn test_keplerianpropagator_keplerian_elements_in_eme2000() {
        setup_global_test_eop();

        // Keplerian output is allowed in any inertial Earth-centered frame, and
        // the elements are about the Earth in that frame's own axes.
        let epc = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_gcrf = create_cartesian_state();

        let prop = KeplerianPropagator::new(
            epc,
            x_gcrf,
            CelestialFrame::GCRF,
            OrbitRepresentation::Cartesian,
            None,
            60.0,
        )
        .unwrap();

        let eme = prop
            .clone()
            .with_output_format(
                CelestialFrame::EME2000,
                OrbitRepresentation::Keplerian,
                Some(DEGREES),
            )
            .unwrap();

        let oe_eme = eme.state(epc).unwrap();
        let expected = state_eci_to_koe(state_gcrf_to_eme2000(x_gcrf), DEGREES);
        for k in 0..6 {
            assert_abs_diff_eq!(oe_eme[k], expected[k], epsilon = 1e-8);
        }

        // Feeding those elements back in as EME2000 elements recovers the
        // original GCRF state, rather than treating them as GCRF elements.
        let round_trip = KeplerianPropagator::new(
            epc,
            Vector6::from_iterator(oe_eme.iter().copied()),
            CelestialFrame::EME2000,
            OrbitRepresentation::Keplerian,
            Some(DEGREES),
            60.0,
        )
        .unwrap();

        let x_back = round_trip.state_gcrf(epc).unwrap();
        for k in 0..6 {
            assert_abs_diff_eq!(x_back[k], x_gcrf[k], epsilon = 1e-6);
        }
    }

    #[test]
    #[parallel]
    fn test_keplerianpropagator_keplerian_elements_in_of_date_frames() {
        setup_global_test_eop();

        // Keplerian input and output are accepted in the of-date frames, with
        // the elements about the Earth in that frame's own axes.
        let epc = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_gcrf = create_cartesian_state();

        let prop = KeplerianPropagator::new(
            epc,
            x_gcrf,
            CelestialFrame::GCRF,
            OrbitRepresentation::Cartesian,
            None,
            60.0,
        )
        .unwrap();

        type Rotate = fn(Epoch, Vector6<f64>) -> Vector6<f64>;
        let cases: [(CelestialFrame, Rotate); 3] = [
            (CelestialFrame::TOD, state_gcrf_to_tod),
            (CelestialFrame::MOD, state_gcrf_to_mod),
            (CelestialFrame::TEME, state_gcrf_to_teme),
        ];

        for (frame, rotate) in cases {
            let of_date = prop
                .clone()
                .with_output_format(frame, OrbitRepresentation::Keplerian, Some(DEGREES))
                .unwrap();
            assert_eq!(of_date.trajectory.frame, frame);

            for dt in [0.0, 600.0] {
                let epc_k = epc + dt;
                let oe = of_date.state(epc_k).unwrap();
                let expected =
                    state_eci_to_koe(rotate(epc_k, prop.state_gcrf(epc_k).unwrap()), DEGREES);
                assert_abs_diff_eq!(oe[0], expected[0], epsilon = 1e-6);
                for k in 1..6 {
                    assert_abs_diff_eq!(oe[k], expected[k], epsilon = 1e-8);
                }
            }

            // Feeding those elements back in as elements of the same frame
            // recovers the original GCRF state.
            let oe_epc = of_date.state(epc).unwrap();
            let round_trip = KeplerianPropagator::new(
                epc,
                Vector6::from_iterator(oe_epc.iter().copied()),
                frame,
                OrbitRepresentation::Keplerian,
                Some(DEGREES),
                60.0,
            )
            .unwrap();

            let x_back = round_trip.state_gcrf(epc).unwrap();
            for k in 0..6 {
                assert_abs_diff_eq!(x_back[k], x_gcrf[k], epsilon = 1e-6);
            }
        }
    }

    #[test]
    #[serial]
    fn test_keplerianpropagator_propagate_keplerian_elements_in_eme2000() {
        setup_global_test_eop();

        let epc = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_gcrf = create_cartesian_state();
        let oe_eme = state_eci_to_koe(state_gcrf_to_eme2000(x_gcrf), DEGREES);

        let mut prop = KeplerianPropagator::new(
            epc,
            oe_eme,
            CelestialFrame::EME2000,
            OrbitRepresentation::Keplerian,
            Some(DEGREES),
            60.0,
        )
        .unwrap();

        // Independent oracle: the same orbit propagated as a GCRF Cartesian
        // state, then rotated into EME2000 and converted to elements.
        let mut oracle = KeplerianPropagator::new(
            epc,
            x_gcrf,
            CelestialFrame::GCRF,
            OrbitRepresentation::Cartesian,
            None,
            60.0,
        )
        .unwrap();

        let target = epc + 3600.0;
        prop.propagate_to(target).unwrap();
        oracle.propagate_to(target).unwrap();

        assert_eq!(prop.current_epoch(), target);

        let expected = state_eci_to_koe(state_gcrf_to_eme2000(oracle.current_state()), DEGREES);
        let elements = prop.current_state();
        // Semi-major axis is in meters, the remaining elements are
        // dimensionless or in degrees.
        assert_abs_diff_eq!(elements[0], expected[0], epsilon = 1e-6);
        for k in 1..6 {
            assert_abs_diff_eq!(elements[k], expected[k], epsilon = 1e-8);
        }
    }
}
