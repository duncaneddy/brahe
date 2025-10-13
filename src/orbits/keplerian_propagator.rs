/*!
 * Keplerian propagator implementation using the new architecture
 * with nalgebra vectors and clean interfaces
 */

use nalgebra::Vector6;
use std::f64::consts::PI;

use crate::constants::{DEG2RAD, RAD2DEG, RADIANS, DEGREES};
use crate::coordinates::{state_cartesian_to_osculating, state_osculating_to_cartesian};
use crate::frames::{state_eci_to_ecef, state_ecef_to_eci};
use crate::orbits::keplerian::mean_motion;
use crate::orbits::traits::{AnalyticPropagator, OrbitPropagator};
use crate::time::Epoch;
use crate::trajectories::OrbitTrajectory;
use crate::constants::AngleFormat;
use crate::trajectories::traits::{OrbitFrame, OrbitRepresentation, Trajectory};
use crate::utils::BraheError;

/// Keplerian propagator for analytical two-body orbital motion
#[derive(Debug, Clone)]
pub struct KeplerianPropagator {
    /// Initial epoch
    pub initial_epoch: Epoch,

    /// Initial state vector in the original representation and frame
    pub initial_state: Vector6<f64>,

    /// Frame of the input/output states
    pub frame: OrbitFrame,

    /// Representation of the input/output states
    pub representation: OrbitRepresentation,

    /// Angle format of the input/output states (for Keplerian)
    pub angle_format: AngleFormat,

    /// Step size in seconds for stepping operations
    pub step_size: f64,

    /// Accumulated trajectory (current state is always the last entry)
    pub trajectory: OrbitTrajectory,

    /// Internal osculating orbital elements (always in radians, ECI frame)
    internal_osculating_elements: Vector6<f64>,

    /// Mean motion in radians per second
    n: f64,
}

impl KeplerianPropagator {
    /// Create a new KeplerianPropagator from orbital elements or Cartesian state.
    /// The input state is assumed to be in the specified frame and representation. The 
    /// input frame, representation, and angle format is assumed to be the desired output format
    /// for the propagator.
    /// 
    /// If the output format needs to be changed, use the `with_output_format` method after initialization.
    /// 
    /// The input representation and angle format must be compatible:
    /// * Keplerian representation requires ECI frame and a specified angle format (Degrees or Radians)
    /// * Cartesian representation can be in ECI or ECEF frame, but angle format must be None
    /// 
    /// The step size must be positive.
    ///
    /// # Arguments
    /// * `epoch` - Initial epoch
    /// * `state` - State vector (Keplerian elements or Cartesian position/velocity)
    /// * `frame` - Reference frame
    /// * `representation` - Type of state representation
    /// * `angle_format` - Format for angular elements (only for Keplerian)
    /// * `step_size` - Step size in seconds for propagation
    ///
    /// # Returns
    /// New KeplerianPropagator instance
    ///
    /// # Panics
    /// Panics if:
    /// - Angle format is None for Keplerian representation
    /// - Keplerian elements are not in ECI frame
    /// - Angle format is None for Keplerian representation
    /// - Angle format is not None for Cartesian representation
    /// - Step size is not positive
    pub fn new(
        epoch: Epoch,
        state: Vector6<f64>,
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: Option<AngleFormat>,
        step_size: f64,
    ) -> Self {
        // Validate inputs
        if representation == OrbitRepresentation::Keplerian && angle_format.is_none() {
            panic!("Angle format must be specified for Keplerian elements");
        }

        if representation == OrbitRepresentation::Keplerian && frame != OrbitFrame::ECI {
            panic!("Keplerian elements must be in ECI frame");
        }

        if representation == OrbitRepresentation::Cartesian && angle_format.is_some() {
            panic!("Angle format should be None for Cartesian representation");
        }

        if step_size <= 0.0 {
            panic!("Step size must be positive");
        }

        // Unwrap angle_format for Keplerian (validated above), use RADIANS for Cartesian
        let angle_format = angle_format.unwrap_or(RADIANS);

        // Convert input state to internal osculating elements in ECI frame with radians
        let internal_elements = Self::convert_to_internal_osculating(
            epoch, state, frame, representation, angle_format
        );

        // Create initial trajectory
        let mut trajectory = OrbitTrajectory::new(
            frame,
            representation,
            angle_format,
        );
        trajectory.add(epoch, state);

        let n = mean_motion(internal_elements[0], AngleFormat::Radians);

        Self {
            initial_epoch: epoch,
            initial_state: state,
            frame,
            representation,
            angle_format,
            internal_osculating_elements: internal_elements,
            trajectory,
            step_size,
            n,
        }
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
    /// * New KeplerianPropagator instance
    ///
    /// # Panics
    /// Panics if step_size is not positive
    pub fn from_keplerian(
        epoch: Epoch,
        elements: Vector6<f64>,
        angle_format: AngleFormat,
        step_size: f64,
    ) -> Self {
        Self::new(epoch, elements, OrbitFrame::ECI, OrbitRepresentation::Keplerian, Some(angle_format), step_size)
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
    /// * New KeplerianPropagator instance
    ///
    /// # Panics
    /// Panics if step_size is not positive
    pub fn from_eci(
        epoch: Epoch,
        state: Vector6<f64>,
        step_size: f64,
    ) -> Self {
        Self::new(epoch, state, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None, step_size)
    }

    /// Create a new KeplerianPropagator from Cartesian state in ECEF frame
    /// 
    /// # Arguments
    /// * `epoch` - Initial epoch
    /// * `state` - Cartesian state vector [x, y, z, vx, vy, vz] in ECEF frame
    /// * `step_size` - Step size in seconds for propagation. Must be positive.
    /// 
    /// # Returns
    /// * New KeplerianPropagator instance
    ///
    /// # Panics
    /// Panics if step_size is not positive
    pub fn from_ecef(
        epoch: Epoch,
        state: Vector6<f64>,
        step_size: f64,
    ) -> Self {
        Self::new(epoch, state, OrbitFrame::ECEF, OrbitRepresentation::Cartesian, None, step_size)
    }

    /// This method allows changing the output format of the propagator. It updates the frame, representation, and angle format.
    /// It also resets the trajectory to only contain the initial state converted to the new format. It should be
    /// used with initialization or after a reset to avoid inconsistencies.
    /// 
    /// The frame, representation, and angle format must be compatible:
    /// * Keplerian representation requires ECI frame and a specified angle format (Degrees or Radians)
    /// * Cartesian representation can be in ECI or ECEF frame, but angle format must be None
    /// 
    /// # Arguments
    /// * `frame` - Desired output frame
    /// * `representation` - Desired output representation
    /// * `angle_format` - Desired angle format (only for Keplerian)
    /// 
    /// # Returns
    /// * Updated KeplerianPropagator instance
    ///
    /// # Panics
    /// Panics if:
    /// - Angle format is None for Keplerian representation
    /// - Keplerian elements are not in ECI frame
    /// - Angle format is not None for Cartesian representation
    #[allow(dead_code)]
    fn with_output_format(mut self, frame: OrbitFrame, representation: OrbitRepresentation, angle_format: Option<AngleFormat>) -> Self {
        // Validate inputs
        if representation == OrbitRepresentation::Keplerian && angle_format.is_none() {
            panic!("Angle format must be specified for Keplerian elements");
        }

        if representation == OrbitRepresentation::Keplerian && frame != OrbitFrame::ECI {
            panic!("Keplerian elements must be in ECI frame");
        }

        if representation == OrbitRepresentation::Cartesian && angle_format.is_some() {
            panic!("Angle format should be None for Cartesian representation");
        }

        // Unwrap angle_format for Keplerian (validated above), use RADIANS for Cartesian
        let angle_format = angle_format.unwrap_or(RADIANS);

        self.frame = frame;
        self.representation = representation;
        self.angle_format = angle_format;

        // Reset trajectory to initial state only
        self.trajectory = OrbitTrajectory::new(
            frame,
            representation,
            angle_format,
        );
        
        // Convert initial state to new format and add to trajectory
        let converted_state = self.convert_from_internal_osculating(self.initial_epoch, self.internal_osculating_elements);
        self.trajectory.add(self.initial_epoch, converted_state);

        self
    }

    /// Convert any state to internal osculating elements (ECI, radians)
    /// 
    /// # Arguments
    /// * `epoch` - Epoch of the input state
    /// * `state` - Input state vector
    /// * `frame` - Frame of the input state
    /// * `representation` - Representation of the input state
    /// * `angle_format` - Angle format of the input state (only for Keplerian)
    /// 
    /// # Returns
    /// * Internal osculating elements in ECI frame with radians
    /// 
    /// # Note
    /// Assumes that the input state is valid and consistent with the specified frame, representation, and angle format.
    /// This is checked during initialization.
    fn convert_to_internal_osculating(
        epoch: Epoch,
        state: Vector6<f64>,
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: AngleFormat,
    ) -> Vector6<f64> {
        match representation {
            OrbitRepresentation::Cartesian => {
                // First convert to ECI frame if needed
                let eci_state = match frame {
                    OrbitFrame::ECI => state,
                    OrbitFrame::ECEF => state_ecef_to_eci(epoch, state),
                };

                // Convert Cartesian to osculating elements
                state_cartesian_to_osculating(eci_state, AngleFormat::Radians)
            }
            OrbitRepresentation::Keplerian => {
                // Convert angles to radians if needed
                if angle_format == AngleFormat::Radians {
                    state
                } else {
                    let mut elements = state;
                    // Convert angles from degrees to radians (i, RAAN, argp, mean_anomaly)
                    for i in 2..6 {
                        elements[i] = elements[i] * DEG2RAD;
                    }
                    elements
                }
            }
        }
    }

    /// Convert internal osculating elements back to original state format
    fn convert_from_internal_osculating(&self, epoch: Epoch, internal_elements: Vector6<f64>) -> Vector6<f64> {
        match self.representation {
            OrbitRepresentation::Cartesian => {
                // Convert osculating elements to Cartesian in ECI
                let eci_cartesian = state_osculating_to_cartesian(internal_elements, AngleFormat::Radians);

                // Convert to original frame if needed
                match self.frame {
                    OrbitFrame::ECI => eci_cartesian,
                    OrbitFrame::ECEF => state_eci_to_ecef(epoch, eci_cartesian),
                }
            }
            OrbitRepresentation::Keplerian => {
                // Convert to original angle format
                match self.angle_format {
                    AngleFormat::Radians => internal_elements,
                    AngleFormat::Degrees => {
                        let mut elements = internal_elements;
                        // Convert angles from radians to degrees (i, RAAN, argp, mean_anomaly)
                        for i in 2..6 {
                            elements[i] = elements[i] * RAD2DEG;
                        }
                        elements
                    }
                }
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

impl OrbitPropagator for KeplerianPropagator {

    fn step_by(&mut self, step_size: f64) -> () {
        let current_epoch = self.current_epoch();
        let target_epoch = current_epoch + step_size;
        let new_state = self.propagate_internal(target_epoch);

        // Convert back to original state format
        let state = self.convert_from_internal_osculating(target_epoch, new_state);

        self.trajectory.add(target_epoch, state)
    }

    // Default implementation from trait is used for:
    // - step()
    // - step_past()
    // - propagate_steps()
    // - propagate_to()

    fn current_epoch(&self) -> Epoch {
        // Return the most recent epoch from trajectory
        self.trajectory.last().unwrap().0
    }

    fn current_state(&self) -> Vector6<f64> {
        // Return the most recent state from trajectory
        self.trajectory.last().unwrap().1
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

    fn reset(&mut self) {
        // Reset trajectory to initial state only
        self.trajectory = OrbitTrajectory::new(
            self.frame,
            self.representation,
            self.angle_format,
        );
        
        // Convert initial state to new format and add to trajectory
        let converted_state = self.convert_from_internal_osculating(self.initial_epoch, self.internal_osculating_elements);
        self.trajectory.add(self.initial_epoch, converted_state);
    }

    fn set_initial_conditions(
        &mut self,
        epoch: Epoch,
        state: Vector6<f64>,
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: Option<AngleFormat>,
    ) {
        // Validate inputs
        if representation == OrbitRepresentation::Keplerian && angle_format.is_none() {
            panic!("Angle format must be specified for Keplerian elements");
        }

        if representation == OrbitRepresentation::Keplerian && frame != OrbitFrame::ECI {
            panic!("Keplerian elements must be in ECI frame");
        }

        if representation == OrbitRepresentation::Cartesian && angle_format.is_some() {
            panic!("Angle format should be None for Cartesian representation");
        }

        // Unwrap angle_format for Keplerian (validated above), use RADIANS for Cartesian
        let angle_format = angle_format.unwrap_or(RADIANS);

        // Update all state
        self.initial_epoch = epoch;
        self.initial_state = state;
        self.frame = frame;
        self.representation = representation;
        self.angle_format = angle_format;

        // Recompute internal elements
        self.internal_osculating_elements = Self::convert_to_internal_osculating(
            epoch, state, frame, representation, angle_format
        );
        self.n = mean_motion(self.internal_osculating_elements[0], AngleFormat::Radians);

        // Reset trajectory to new initial conditions
        self.trajectory = OrbitTrajectory::new(
            frame,
            representation,
            angle_format,
        );
        self.trajectory.add(epoch, state);
    }

    fn set_eviction_policy_max_size(&mut self, max_size: usize) -> Result<(), BraheError> {
        self.trajectory.set_eviction_policy_max_size(max_size)
    }

    fn set_eviction_policy_max_age(&mut self, max_age: f64) -> Result<(), BraheError> {
        self.trajectory.set_eviction_policy_max_age(max_age)
    }
}

impl AnalyticPropagator for KeplerianPropagator {
    fn state(&self, epoch: Epoch) -> Vector6<f64> {
        // Get state in original format
        let internal_state = self.propagate_internal(epoch);
        self.convert_from_internal_osculating(epoch, internal_state)
    }

    fn state_eci(&self, epoch: Epoch) -> Vector6<f64> {
        // Get state in original format
        let state = self.propagate_internal(epoch);

        // Convert to ECI
        state_osculating_to_cartesian(state, AngleFormat::Radians)
    }

    fn state_ecef(&self, epoch: Epoch) -> Vector6<f64> {
        let eci_state = self.state_eci(epoch);
        state_eci_to_ecef(epoch, eci_state)
    }

    fn state_as_osculating_elements(&self, epoch: Epoch, angle_format: AngleFormat) -> Vector6<f64> {
        let internal_state = self.propagate_internal(epoch);
        match angle_format {
            AngleFormat::Degrees => {
                let mut elements = internal_state;
                // Convert angles from radians to degrees (i, RAAN, argp, mean_anomaly)
                for i in 2..6 {
                    elements[i] = elements[i] * RAD2DEG;
                }
                elements
            }
            AngleFormat::Radians => internal_state,
        }
    }

    // Default implementations from trait are used for:
    // - states()
    // - states_eci()
    // - states_ecef()
    // - states_as_osculating_elements()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{Epoch, TimeSystem};
    use crate::utils::testing::setup_global_test_eop;
    use crate::orbits::keplerian::orbital_period;
    use crate::coordinates::state_cartesian_to_osculating;
    use approx::assert_abs_diff_eq;

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

        state_osculating_to_cartesian(Vector6::new(a, e, i, raan, argp, ma), DEGREES)
    }

    // KeplerianPropagator Method Tests

    #[test]
    fn test_keplerianpropagator_new() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator = KeplerianPropagator::new(
            epoch,
            elements,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(RADIANS),
            60.0,
        );

        assert_eq!(propagator.initial_epoch(), epoch);
        assert_eq!(propagator.current_epoch(), epoch);
        assert_abs_diff_eq!(propagator.initial_state()[0], 7000e3, epsilon = 1.0);
        assert_abs_diff_eq!(propagator.initial_state()[1], 0.01, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "Angle format must be specified for Keplerian elements")]
    fn test_keplerianpropagator_new_invalid_angle_format() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        // This should panic because angle format is None for Keplerian
        let _propagator = KeplerianPropagator::new(
            epoch,
            elements,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            None,
            60.0,
        );
    }

    #[test]
    #[should_panic(expected = "Keplerian elements must be in ECI frame")]
    fn test_keplerianpropagator_new_invalid_frame() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        // This should panic because Keplerian elements are not in ECI frame
        let _propagator = KeplerianPropagator::new(
            epoch,
            elements,
            OrbitFrame::ECEF,
            OrbitRepresentation::Keplerian,
            Some(RADIANS),
            60.0,
        );
    }

    #[test]
    #[should_panic(expected = "Angle format should be None for Cartesian representation")]
    fn test_keplerianpropagator_new_invalid_cartesian_angle_format() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let state = create_test_elements(); // Using elements for simplicity

        // This should panic because angle format is not None for Cartesian
        let _propagator = KeplerianPropagator::new(
            epoch,
            state,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            Some(RADIANS),
            60.0,
        );
    }

    #[test]
    #[should_panic(expected = "Step size must be positive")]
    fn test_keplerianpropagator_new_invalid_step_size_neative(){
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        // This should panic because step size is not positive
        let _propagator = KeplerianPropagator::new(
            epoch,
            elements,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(RADIANS),
            -10.0,
        );
    }

    #[test]
    #[should_panic(expected = "Step size must be positive")]
    fn test_keplerianpropagator_new_invalid_step_size_zero(){
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        // This should panic because step size is not positive
        let _propagator = KeplerianPropagator::new(
            epoch,
            elements,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(RADIANS),
            0.0,
        );
    }

    #[test]
    fn test_keplerianpropagator_from_keplerian() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        assert_eq!(propagator.initial_epoch(), epoch);
        assert_eq!(propagator.step_size(), 60.0);
    }

    #[test]
    fn test_keplerianpropagator_from_eci() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let cartesian = create_cartesian_state();

        let propagator = KeplerianPropagator::from_eci(
            epoch,
            cartesian,
            60.0,
        );

        assert_eq!(propagator.initial_epoch(), epoch);
        assert_eq!(propagator.step_size(), 60.0);
    }

    #[test]
    fn test_keplerianpropagator_from_ecef() {
        setup_global_test_eop();
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let cartesian = state_ecef_to_eci(epoch, create_cartesian_state());

        let propagator = KeplerianPropagator::from_ecef(
            epoch,
            cartesian,
            60.0,
        );

        assert_eq!(propagator.initial_epoch(), epoch);
        assert_eq!(propagator.step_size(), 60.0);
        assert_eq!(propagator.frame, OrbitFrame::ECEF);
    }

    // OrbitPropagator Trait Tests

    #[test]
    fn test_keplerianpropagator_orbitpropagator_step() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        propagator.step();

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
    fn test_keplerianpropagator_orbitpropagator_step_by() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        propagator.step_by(120.0);

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
    fn test_keplerian_orbitpropagator_step_past() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        let target_epoch = epoch + 250.0;
        propagator.step_past(target_epoch);

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
    fn test_keplerianpropagator_orbitpropagator_propagate_steps() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        propagator.propagate_steps(5);

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
    fn test_keplerianpropagator_orbitpropagator_propagate_to() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        let target_epoch = epoch + 90.0;
        propagator.propagate_to(target_epoch);

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
    fn test_keplerianpropagator_orbitpropagator_current_epoch() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        assert_eq!(propagator.current_epoch(), epoch);

        // step and check epoch advanced
        propagator.step();
        assert_ne!(propagator.current_epoch(), epoch);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_current_state() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        // Initial state should match
        assert_eq!(propagator.current_state(), elements);

        // After step, should be different
        propagator.step();
        let current_state = propagator.current_state();
        assert_ne!(current_state, elements);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_initial_epoch() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        assert_eq!(propagator.initial_epoch(), epoch);

        // Step and confirm initial epoch unchanged
        propagator.step();
        assert_eq!(propagator.initial_epoch(), epoch);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_initial_state() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        assert_eq!(propagator.initial_state(), elements);

        // Step and confirm initial state unchanged
        propagator.step();
        assert_eq!(propagator.initial_state(), elements);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_step_size() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        assert_eq!(propagator.step_size(), 60.0);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_set_step_size() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        // Confirm initial step size
        assert_eq!(propagator.step_size(), 60.0);

        // Change step size
        propagator.set_step_size(120.0);
        assert_eq!(propagator.step_size(), 120.0);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_reset() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        // Propagate forward
        propagator.propagate_steps(5);
        assert_eq!(propagator.trajectory.len(), 6);

        // Reset
        propagator.reset();
        assert_eq!(propagator.trajectory.len(), 1);
        assert_eq!(propagator.current_epoch(), epoch);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_set_initial_conditions() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        // Set new initial conditions
        let new_epoch = Epoch::from_jd(TEST_EPOCH_JD + 1.0, TimeSystem::UTC);
        let new_elements = create_circular_elements();

        propagator.set_initial_conditions(
            new_epoch,
            new_elements,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
        );

        assert_eq!(propagator.initial_epoch(), new_epoch);
        assert_eq!(propagator.initial_state(), new_elements);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_set_eviction_policy_max_size() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        propagator.set_eviction_policy_max_size(5).unwrap();

        // Propagate 10 steps
        propagator.propagate_steps(10);

        // Should only keep 5 states
        assert_eq!(propagator.trajectory.len(), 5);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_set_eviction_policy_max_age() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        let result = propagator.set_eviction_policy_max_age(120.0);
        assert!(result.is_ok());

        propagator.propagate_steps(10);
        assert!(propagator.trajectory.len() > 0);
    }

    // AnalyticPropagator Trait Tests

    #[test]
    fn test_keplerianpropagator_analyticpropagator_state() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        let target_epoch = epoch + orbital_period(elements[0]);
        let state = propagator.state(target_epoch);

        // State should be exactly the same as initial elements after one orbital period
        for i in 0..6 {
            assert_abs_diff_eq!(state[i], elements[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_keplerianpropagator_analyticpropagator_state_eci() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        let state = propagator.state_eci(epoch + orbital_period(elements[0]));

        // Should be Cartesian state in ECI
        assert!(state.norm() > 0.0);
        // Convert back to orbital elements and verify semi-major axis is preserved
        let computed_elements = state_cartesian_to_osculating(state, DEGREES);
        
        // Confirm equality within small tolerance
        for i in 0..6 {
            assert_abs_diff_eq!(computed_elements[i], elements[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_keplerianpropagator_analyticpropagator_state_ecef() {
        setup_global_test_eop();
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        let state = propagator.state_ecef(epoch + orbital_period(elements[0]));

        // Convert back into osculating elements via ECI
        let eci_state = state_ecef_to_eci(epoch + orbital_period(elements[0]), state);
        let computed_elements = state_cartesian_to_osculating(eci_state, DEGREES);

        // Confirm equality within small tolerance
        for i in 0..6 {
            assert_abs_diff_eq!(computed_elements[i], elements[i], epsilon = 1e-6);
        }

    }

    #[test]
    fn test_keplerianpropagator_analyticpropagator_state_as_osculating_elements() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        let osc_elements = propagator.state_as_osculating_elements(epoch + orbital_period(elements[0]), AngleFormat::Degrees);

        // Should match initial elements within small tolerance
        for i in 0..6 {
            assert_abs_diff_eq!(osc_elements[i], elements[i], epsilon = 1e-6);
        }

        // Now test with radians to degrees conversion
        let osc_elements_rad = propagator.state_as_osculating_elements(epoch + orbital_period(elements[0]), AngleFormat::Radians);
        for i in 0..2 {
            assert_abs_diff_eq!(osc_elements_rad[i], elements[i], epsilon = 1e-6);
        }
        for i in 2..6 {
            assert_abs_diff_eq!(osc_elements_rad[i] * RAD2DEG, elements[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_keplerianpropagator_analyticpropagator_states() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        let epochs = vec![
            epoch,
            epoch + orbital_period(elements[0]),
            epoch + 2.0 * orbital_period(elements[0])
        ];

        let traj = propagator.states(&epochs);
        assert_eq!(traj.len(), 3);

        // Confirm all elements remain unchanged within small tolerance
        for state in &traj {
            for i in 0..6 {
                assert_abs_diff_eq!(state[i], elements[i], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_keplerianpropagator_analyticpropagator_states_eci() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        let epochs = vec![
            epoch,
            epoch + orbital_period(elements[0]),
            epoch + 2.0 * orbital_period(elements[0])
        ];

        let states = propagator.states_eci(&epochs);
        assert_eq!(states.len(), 3);
        // Verify states convert back to original elements within small tolerance
        for state in &states {
            let computed_elements = state_cartesian_to_osculating(*state, DEGREES);
            for i in 0..6 {
                assert_abs_diff_eq!(computed_elements[i], elements[i], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_keplerianpropagator_analyticpropagator_states_ecef() {
        setup_global_test_eop();
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        let epochs = vec![
            epoch,
            epoch + orbital_period(elements[0]),
            epoch + 2.0 * orbital_period(elements[0])
        ];

        let states = propagator.states_ecef(&epochs);
        assert_eq!(states.len(), 3);
        // Verify states convert back to original elements within small tolerance
        for (i, state) in states.iter().enumerate() {
            let eci_state = state_ecef_to_eci(epochs[i], *state);
            let computed_elements = state_cartesian_to_osculating(eci_state, DEGREES);
            for j in 0..6 {
                assert_abs_diff_eq!(computed_elements[j], elements[j], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_keplerianpropagator_analyticpropagator_states_as_osculating_elements() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            AngleFormat::Degrees,
            60.0,
        );

        let epochs = vec![
            epoch,
            epoch + orbital_period(elements[0]),
            epoch + 2.0 * orbital_period(elements[0])
        ];

        let traj = propagator.states_as_osculating_elements(&epochs, AngleFormat::Degrees);
        assert_eq!(traj.len(), 3);

        // Confirm all elements remain unchanged within small tolerance
        for state in &traj {
            for i in 0..6 {
                assert_abs_diff_eq!(state[i], elements[i], epsilon = 1e-6);
            }
        }

        // Repeat with radians output
        let traj_rad = propagator.states_as_osculating_elements(&epochs, AngleFormat::Radians);
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
}