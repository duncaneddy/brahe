/*!
 * Keplerian propagator implementation using the new architecture
 * with nalgebra vectors and clean interfaces
 */

use nalgebra::Vector6;
use std::f64::consts::PI;

use crate::constants::DEG2RAD;
use crate::coordinates::{state_cartesian_to_osculating, state_osculating_to_cartesian};
use crate::frames::{state_eci_to_ecef, state_ecef_to_eci};
use crate::orbits::keplerian::mean_motion;
use crate::orbits::traits::{AnalyticPropagator, OrbitPropagator};
use crate::time::Epoch;
use crate::trajectories::{AngleFormat, OrbitFrame, OrbitRepresentation, OrbitTrajectory, OrbitalTrajectory, Trajectory};
use crate::utils::BraheError;

/// Keplerian propagator for analytical two-body orbital motion
#[derive(Debug, Clone)]
pub struct KeplerianPropagator {
    /// Initial epoch
    initial_epoch: Epoch,

    /// Initial state vector in the original representation and frame
    initial_state: Vector6<f64>,

    /// Frame of the input/output states
    frame: OrbitFrame,

    /// Representation of the input/output states
    representation: OrbitRepresentation,

    /// Angle format of the input/output states (for Keplerian)
    angle_format: AngleFormat,

    /// Internal osculating orbital elements (always in radians, ECI frame)
    internal_osculating_elements: Vector6<f64>,

    /// Accumulated trajectory (current state is always the last entry)
    trajectory: OrbitTrajectory,

    /// Step size in seconds for stepping operations
    step_size: f64,

    /// Mean motion in radians per second
    n: f64,
}

impl KeplerianPropagator {
    /// Create a new KeplerianPropagator from orbital elements or Cartesian state
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
    pub fn new(
        epoch: Epoch,
        state: Vector6<f64>,
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: AngleFormat,
        step_size: f64,
    ) -> Result<Self, BraheError> {
        // Validate inputs
        if representation == OrbitRepresentation::Keplerian && angle_format == AngleFormat::None {
            return Err(BraheError::Error(
                "Angle format must be specified for Keplerian elements".to_string(),
            ));
        }

        if representation == OrbitRepresentation::Cartesian && angle_format != AngleFormat::None {
            return Err(BraheError::Error(
                "Angle format should be None for Cartesian representation".to_string(),
            ));
        }

        // Convert input state to internal osculating elements in ECI frame with radians
        let internal_elements = Self::convert_to_internal_osculating(
            epoch, state, frame, representation, angle_format
        )?;

        // Create initial trajectory
        let mut trajectory = OrbitTrajectory::new(
            frame,
            representation,
            angle_format,
        );
        trajectory.add_state(epoch, state)?;

        let n = mean_motion(internal_elements[0], false);

        Ok(Self {
            initial_epoch: epoch,
            initial_state: state,
            frame,
            representation,
            angle_format,
            internal_osculating_elements: internal_elements,
            trajectory,
            step_size,
            n,
        })
    }

    /// Create a new KeplerianPropagator from Keplerian orbital elements
    pub fn from_keplerian(
        epoch: Epoch,
        elements: Vector6<f64>,
        frame: OrbitFrame,
        angle_format: AngleFormat,
        step_size: f64,
    ) -> Result<Self, BraheError> {
        Self::new(epoch, elements, frame, OrbitRepresentation::Keplerian, angle_format, step_size)
    }

    /// Create a new KeplerianPropagator from Cartesian state
    pub fn from_cartesian(
        epoch: Epoch,
        cartesian_state: Vector6<f64>,
        frame: OrbitFrame,
        step_size: f64,
    ) -> Result<Self, BraheError> {
        Self::new(epoch, cartesian_state, frame, OrbitRepresentation::Cartesian, AngleFormat::None, step_size)
    }

    /// Convert any state to internal osculating elements (ECI, radians)
    fn convert_to_internal_osculating(
        epoch: Epoch,
        state: Vector6<f64>,
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: AngleFormat,
    ) -> Result<Vector6<f64>, BraheError> {
        match representation {
            OrbitRepresentation::Cartesian => {
                // First convert to ECI frame if needed
                let eci_state = match frame {
                    OrbitFrame::ECI => state,
                    OrbitFrame::ECEF => state_ecef_to_eci(epoch, state),
                };

                // Convert Cartesian to osculating elements
                Ok(state_cartesian_to_osculating(eci_state, false))
            }
            OrbitRepresentation::Keplerian => {
                if frame != OrbitFrame::ECI {
                    return Err(BraheError::Error(
                        "Keplerian elements must be in ECI frame".to_string(),
                    ));
                }

                // Convert angles to radians if needed
                if angle_format == AngleFormat::Radians {
                    Ok(state)
                } else {
                    let mut elements = state;
                    // Convert angles from degrees to radians (i, RAAN, argp, mean_anomaly)
                    for i in 2..6 {
                        elements[i] = elements[i] * DEG2RAD;
                    }
                    Ok(elements)
                }
            }
        }
    }

    /// Convert internal osculating elements back to original state format
    fn convert_from_internal_osculating(&self, epoch: Epoch, internal_elements: Vector6<f64>) -> Result<Vector6<f64>, BraheError> {
        match self.representation {
            OrbitRepresentation::Cartesian => {
                // Convert osculating elements to Cartesian in ECI
                let eci_cartesian = state_osculating_to_cartesian(internal_elements, false);

                // Convert to original frame if needed
                match self.frame {
                    OrbitFrame::ECI => Ok(eci_cartesian),
                    OrbitFrame::ECEF => Ok(state_eci_to_ecef(epoch, eci_cartesian)),
                }
            }
            OrbitRepresentation::Keplerian => {
                // Convert to original angle format
                match self.angle_format {
                    AngleFormat::Radians => Ok(internal_elements),
                    AngleFormat::Degrees => {
                        let mut elements = internal_elements;
                        // Convert angles from radians to degrees (i, RAAN, argp, mean_anomaly)
                        for i in 2..6 {
                            elements[i] = elements[i] * crate::constants::RAD2DEG;
                        }
                        Ok(elements)
                    }
                    AngleFormat::None => Err(BraheError::Error(
                        "Invalid angle format for Keplerian elements".to_string(),
                    )),
                }
            }
        }
    }

    /// Propagate internal Keplerian elements to a target epoch
    fn propagate_internal(&self, target_epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
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

        // Create new internal state with propagated mean anomaly
        let propagated_elements = Vector6::new(a, e, i, raan, argp, m);

        // Convert back to original state format
        self.convert_from_internal_osculating(target_epoch, propagated_elements)
    }
}

impl OrbitPropagator for KeplerianPropagator {
    fn step(&mut self) -> Result<(), BraheError> {
        let current_epoch = self.current_epoch();
        let target_epoch = current_epoch + self.step_size;
        let new_state = self.propagate_internal(target_epoch)?;
        self.trajectory.add_state(target_epoch, new_state)
    }

    fn step_by(&mut self, step_size: f64) -> Result<(), BraheError> {
        let current_epoch = self.current_epoch();
        let target_epoch = current_epoch + step_size;
        let new_state = self.propagate_internal(target_epoch)?;
        self.trajectory.add_state(target_epoch, new_state)
    }

    fn propagate_steps(&mut self, num_steps: usize) -> Result<(), BraheError> {
        for _ in 0..num_steps {
            self.step()?;
        }
        Ok(())
    }

    fn propagate_to(&mut self, target_epoch: Epoch) -> Result<(), BraheError> {
        let new_state = self.propagate_internal(target_epoch)?;
        self.trajectory.add_state(target_epoch, new_state)
    }

    fn current_state(&self) -> Vector6<f64> {
        // Return the most recent state from trajectory
        if let Some(_last_epoch) = self.trajectory.end_epoch() {
            self.trajectory.state(self.trajectory.len() - 1).unwrap_or(Vector6::zeros())
        } else {
            Vector6::zeros()
        }
    }

    fn current_epoch(&self) -> Epoch {
        // Return the most recent epoch from trajectory
        self.trajectory.end_epoch().unwrap_or(self.initial_epoch)
    }

    fn initial_state(&self) -> Vector6<f64> {
        self.initial_state
    }

    fn initial_epoch(&self) -> Epoch {
        self.initial_epoch
    }

    fn step_size(&self) -> f64 {
        self.step_size
    }

    fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }

    fn reset(&mut self) -> Result<(), BraheError> {
        self.internal_osculating_elements = Self::convert_to_internal_osculating(
            self.initial_epoch,
            self.initial_state,
            self.frame,
            self.representation,
            self.angle_format,
        )?;
        self.n = mean_motion(self.internal_osculating_elements[0], false);

        // Reset trajectory to initial state only
        self.trajectory = OrbitTrajectory::new(
            self.frame,
            self.representation,
            self.angle_format,
        );
        self.trajectory.add_state(self.initial_epoch, self.initial_state)?;

        Ok(())
    }

    fn set_initial_conditions(
        &mut self,
        epoch: Epoch,
        state: Vector6<f64>,
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: AngleFormat,
    ) -> Result<(), BraheError> {
        // Validate inputs
        if representation == OrbitRepresentation::Keplerian && angle_format == AngleFormat::None {
            return Err(BraheError::Error(
                "Angle format must be specified for Keplerian elements".to_string(),
            ));
        }

        if representation == OrbitRepresentation::Cartesian && angle_format != AngleFormat::None {
            return Err(BraheError::Error(
                "Angle format should be None for Cartesian representation".to_string(),
            ));
        }

        // Update all state
        self.initial_epoch = epoch;
        self.initial_state = state;
        self.frame = frame;
        self.representation = representation;
        self.angle_format = angle_format;

        // Recompute internal elements
        self.internal_osculating_elements = Self::convert_to_internal_osculating(
            epoch, state, frame, representation, angle_format
        )?;
        self.n = mean_motion(self.internal_osculating_elements[0], false);

        // Reset trajectory to new initial conditions
        self.trajectory = OrbitTrajectory::new(
            frame,
            representation,
            angle_format,
        );
        self.trajectory.add_state(epoch, state)?;

        Ok(())
    }

    fn trajectory(&self) -> &OrbitTrajectory {
        &self.trajectory
    }

    fn trajectory_mut(&mut self) -> &mut OrbitTrajectory {
        &mut self.trajectory
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
        self.propagate_internal(epoch).unwrap_or(Vector6::zeros())
    }

    fn state_eci(&self, epoch: Epoch) -> Vector6<f64> {
        // Propagate internal elements and convert to ECI Cartesian
        let dt = epoch - self.initial_epoch;
        let a = self.internal_osculating_elements[0];
        let e = self.internal_osculating_elements[1];
        let i = self.internal_osculating_elements[2];
        let raan = self.internal_osculating_elements[3];
        let argp = self.internal_osculating_elements[4];
        let m0 = self.internal_osculating_elements[5];
        let m = (m0 + self.n * dt) % (2.0 * PI);
        let propagated_elements = Vector6::new(a, e, i, raan, argp, m);

        state_osculating_to_cartesian(propagated_elements, false)
    }

    fn state_ecef(&self, epoch: Epoch) -> Vector6<f64> {
        let eci_state = self.state_eci(epoch);
        state_eci_to_ecef(epoch, eci_state)
    }

    fn state_osculating_elements(&self, epoch: Epoch) -> Vector6<f64> {
        // Propagate internal elements (always in radians)
        let dt = epoch - self.initial_epoch;
        let a = self.internal_osculating_elements[0];
        let e = self.internal_osculating_elements[1];
        let i = self.internal_osculating_elements[2];
        let raan = self.internal_osculating_elements[3];
        let argp = self.internal_osculating_elements[4];
        let m0 = self.internal_osculating_elements[5];
        let m = (m0 + self.n * dt) % (2.0 * PI);

        Vector6::new(a, e, i, raan, argp, m)
    }

    fn states(&self, epochs: &[Epoch]) -> OrbitTrajectory {
        let mut states = Vec::new();
        for &epoch in epochs {
            if let Ok(state) = self.propagate_internal(epoch) {
                states.push(state);
            }
        }

        OrbitTrajectory::from_orbital_data(
            epochs.to_vec(),
            states,
            self.frame,
            self.representation,
            self.angle_format,
        )
    }

    fn states_eci(&self, epochs: &[Epoch]) -> OrbitTrajectory {
        let mut states = Vec::new();
        for &epoch in epochs {
            states.push(self.state_eci(epoch));
        }

        OrbitTrajectory::from_orbital_data(
            epochs.to_vec(),
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        )
    }

    fn states_ecef(&self, epochs: &[Epoch]) -> OrbitTrajectory {
        let mut states = Vec::new();
        for &epoch in epochs {
            states.push(self.state_ecef(epoch));
        }

        OrbitTrajectory::from_orbital_data(
            epochs.to_vec(),
            states,
            OrbitFrame::ECEF,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        )
    }

    fn states_osculating_elements(&self, epochs: &[Epoch]) -> OrbitTrajectory {
        let mut states = Vec::new();
        for &epoch in epochs {
            states.push(self.state_osculating_elements(epoch));
        }

        OrbitTrajectory::from_orbital_data(
            epochs.to_vec(),
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            AngleFormat::Radians,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{Epoch, TimeSystem};
    use crate::trajectories::OrbitalTrajectory;
    use crate::utils::testing::setup_global_test_eop;
    use approx::assert_abs_diff_eq;

    // Test data constants
    const TEST_EPOCH_JD: f64 = 2451545.0;

    fn create_test_elements() -> Vector6<f64> {
        Vector6::new(7000e3, 0.01, 0.1, 0.0, 0.0, 0.0)
    }

    fn create_circular_elements() -> Vector6<f64> {
        Vector6::new(7000e3, 0.0, 0.0, 0.0, 0.0, 0.0)
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
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        assert_eq!(propagator.initial_epoch(), epoch);
        assert_eq!(propagator.current_epoch(), epoch);
        assert_abs_diff_eq!(propagator.initial_state()[0], 7000e3, epsilon = 1.0);
        assert_abs_diff_eq!(propagator.initial_state()[1], 0.01, epsilon = 1e-10);
    }

    #[test]
    fn test_keplerianpropagator_from_keplerian() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        assert_eq!(propagator.initial_epoch(), epoch);
        assert_eq!(propagator.step_size(), 60.0);
    }

    #[test]
    fn test_keplerianpropagator_from_cartesian() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let cartesian = Vector6::new(6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0);

        let propagator = KeplerianPropagator::from_cartesian(
            epoch,
            cartesian,
            OrbitFrame::ECI,
            60.0,
        ).unwrap();

        assert_eq!(propagator.initial_epoch(), epoch);
        assert_eq!(propagator.step_size(), 60.0);
    }

    // OrbitPropagator Trait Tests

    #[test]
    fn test_keplerianpropagator_orbitpropagator_step() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        propagator.step().unwrap();

        let new_epoch = propagator.current_epoch();
        assert_eq!(new_epoch, epoch + 60.0);
        assert_eq!(propagator.trajectory().len(), 2);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_step_by() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        propagator.step_by(120.0).unwrap();

        let new_epoch = propagator.current_epoch();
        assert_eq!(new_epoch, epoch + 120.0);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_propagate_steps() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        propagator.propagate_steps(5).unwrap();

        assert_eq!(propagator.trajectory().len(), 6); // Initial + 5 steps
        let new_epoch = propagator.current_epoch();
        assert_eq!(new_epoch, epoch + 300.0);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_propagate_to() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        let target_epoch = epoch + 1.0; // 1 day forward
        propagator.propagate_to(target_epoch).unwrap();

        let current_epoch = propagator.current_epoch();
        assert_eq!(current_epoch, target_epoch);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_current_state() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        // Initial state should match
        assert_eq!(propagator.current_state(), elements);

        // After step, should be different
        propagator.step().unwrap();
        let current_state = propagator.current_state();
        assert_ne!(current_state, elements);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_current_epoch() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        assert_eq!(propagator.current_epoch(), epoch);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_initial_state() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        assert_eq!(propagator.initial_state(), elements);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_initial_epoch() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        assert_eq!(propagator.initial_epoch(), epoch);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_step_size() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        assert_eq!(propagator.step_size(), 60.0);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_set_step_size() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        propagator.set_step_size(120.0);
        assert_eq!(propagator.step_size(), 120.0);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_reset() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        // Propagate forward
        propagator.propagate_steps(5).unwrap();
        assert_eq!(propagator.trajectory().len(), 6);

        // Reset
        propagator.reset().unwrap();
        assert_eq!(propagator.trajectory().len(), 1);
        assert_eq!(propagator.current_epoch(), epoch);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_set_initial_conditions() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        // Set new initial conditions
        let new_epoch = Epoch::from_jd(TEST_EPOCH_JD + 1.0, TimeSystem::UTC);
        let new_elements = create_test_elements();

        propagator.set_initial_conditions(
            new_epoch,
            new_elements,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            AngleFormat::Radians,
        ).unwrap();

        assert_eq!(propagator.initial_epoch(), new_epoch);
        assert_eq!(propagator.initial_state(), new_elements);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_trajectory() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        let traj = propagator.trajectory();
        assert_eq!(traj.len(), 1);
        assert_eq!(traj.frame, OrbitFrame::ECI);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_trajectory_mut() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        let traj = propagator.trajectory_mut();
        assert_eq!(traj.len(), 1);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_set_eviction_policy_max_size() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        propagator.set_eviction_policy_max_size(5).unwrap();

        // Propagate 10 steps
        propagator.propagate_steps(10).unwrap();

        // Should only keep 5 states
        assert_eq!(propagator.trajectory().len(), 5);
    }

    #[test]
    fn test_keplerianpropagator_orbitpropagator_set_eviction_policy_max_age() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        let result = propagator.set_eviction_policy_max_age(120.0);
        assert!(result.is_ok());

        propagator.propagate_steps(10).unwrap();
        assert!(propagator.trajectory().len() > 0);
    }

    // AnalyticPropagator Trait Tests

    #[test]
    fn test_keplerianpropagator_analyticpropagator_state() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        let target_epoch = epoch + 0.01; // 0.01 days forward
        let state = propagator.state(target_epoch);

        // State should be valid
        assert!(state.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_keplerianpropagator_analyticpropagator_state_eci() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        let state = propagator.state_eci(epoch);

        // Should be Cartesian state in ECI
        assert!(state.norm() > 0.0);
        // Semi-major axis of 7000km should give radius ~7000km
        assert_abs_diff_eq!(state.fixed_rows::<3>(0).norm(), 7000e3, epsilon = 1000.0);
    }

    #[test]
    fn test_keplerianpropagator_analyticpropagator_state_ecef() {
        setup_global_test_eop();
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        let state = propagator.state_ecef(epoch);

        // ECEF state should be different from ECI due to frame rotation
        let eci_state = propagator.state_eci(epoch);
        assert!((state - eci_state).norm() > 0.0);
    }

    #[test]
    fn test_keplerianpropagator_analyticpropagator_state_osculating_elements() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_test_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        let osc_elements = propagator.state_osculating_elements(epoch);

        // Should match initial elements at initial epoch
        assert_abs_diff_eq!(osc_elements[0], elements[0], epsilon = 1.0);
        assert_abs_diff_eq!(osc_elements[1], elements[1], epsilon = 1e-10);
        assert_abs_diff_eq!(osc_elements[2], elements[2], epsilon = 1e-10);
    }

    #[test]
    fn test_keplerianpropagator_analyticpropagator_states() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        let epochs = vec![
            epoch,
            epoch + 0.01,
            epoch + 0.02,
        ];

        let traj = propagator.states(&epochs);
        assert_eq!(traj.len(), 3);
    }

    #[test]
    fn test_keplerianpropagator_analyticpropagator_states_eci() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        let epochs = vec![epoch, epoch + 0.01];

        let traj = propagator.states_eci(&epochs);
        assert_eq!(traj.len(), 2);
        assert_eq!(traj.frame, OrbitFrame::ECI);
        assert_eq!(traj.representation, OrbitRepresentation::Cartesian);
    }

    #[test]
    fn test_keplerianpropagator_analyticpropagator_states_ecef() {
        setup_global_test_eop();
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        let epochs = vec![epoch, epoch + 0.01];

        let traj = propagator.states_ecef(&epochs);
        assert_eq!(traj.len(), 2);
        assert_eq!(traj.frame, OrbitFrame::ECEF);
    }

    #[test]
    fn test_keplerianpropagator_analyticpropagator_states_osculating_elements() {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = create_circular_elements();

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        let epochs = vec![epoch, epoch + 0.01];

        let traj = propagator.states_osculating_elements(&epochs);
        assert_eq!(traj.len(), 2);
        assert_eq!(traj.representation, OrbitRepresentation::Keplerian);
        assert_eq!(traj.angle_format, AngleFormat::Radians);
    }
}