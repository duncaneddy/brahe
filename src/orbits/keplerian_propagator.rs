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
use crate::trajectories::InterpolationMethod;
use crate::trajectories::{AngleFormat, OrbitFrame, OrbitRepresentation, Trajectory6};
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
    trajectory: Trajectory6,

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
        let mut trajectory = Trajectory6::new_orbital_trajectory(
            frame,
            representation,
            angle_format,
            InterpolationMethod::Linear,
        )?;
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
        if let Some(last_epoch) = self.trajectory.end_epoch() {
            self.trajectory.state_at_epoch(&last_epoch).unwrap_or(Vector6::zeros())
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
        self.trajectory = Trajectory6::new_orbital_trajectory(
            self.frame,
            self.representation,
            self.angle_format,
            InterpolationMethod::Linear,
        )?;
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
        self.trajectory = Trajectory6::new_orbital_trajectory(
            frame,
            representation,
            angle_format,
            InterpolationMethod::Linear,
        )?;
        self.trajectory.add_state(epoch, state)?;

        Ok(())
    }

    fn trajectory(&self) -> &Trajectory6 {
        &self.trajectory
    }

    fn trajectory_mut(&mut self) -> &mut Trajectory6 {
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

    fn states(&self, epochs: &[Epoch]) -> Trajectory6 {
        let mut states = Vec::new();
        for &epoch in epochs {
            if let Ok(state) = self.propagate_internal(epoch) {
                states.push(state);
            }
        }

        Trajectory6::from_orbital_data(
            epochs.to_vec(),
            states,
            self.frame,
            self.representation,
            self.angle_format,
            InterpolationMethod::Linear,
        ).unwrap_or_else(|_| {
            Trajectory6::new_orbital_trajectory(
                self.frame,
                self.representation,
                self.angle_format,
                InterpolationMethod::Linear,
            ).unwrap()
        })
    }

    fn states_eci(&self, epochs: &[Epoch]) -> Trajectory6 {
        let mut states = Vec::new();
        for &epoch in epochs {
            states.push(self.state_eci(epoch));
        }

        Trajectory6::from_orbital_data(
            epochs.to_vec(),
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
            InterpolationMethod::Linear,
        ).unwrap()
    }

    fn states_ecef(&self, epochs: &[Epoch]) -> Trajectory6 {
        let mut states = Vec::new();
        for &epoch in epochs {
            states.push(self.state_ecef(epoch));
        }

        Trajectory6::from_orbital_data(
            epochs.to_vec(),
            states,
            OrbitFrame::ECEF,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
            InterpolationMethod::Linear,
        ).unwrap()
    }

    fn states_osculating_elements(&self, epochs: &[Epoch]) -> Trajectory6 {
        let mut states = Vec::new();
        for &epoch in epochs {
            states.push(self.state_osculating_elements(epoch));
        }

        Trajectory6::from_orbital_data(
            epochs.to_vec(),
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            AngleFormat::Radians,
            InterpolationMethod::Linear,
        ).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{Epoch, TimeSystem};
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_keplerian_propagator_creation() {
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let elements = Vector6::new(7000e3, 0.01, 0.1, 0.0, 0.0, 0.0);

        let propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0, // 60 second step size
        ).unwrap();

        assert_eq!(propagator.initial_epoch(), epoch);
        assert_eq!(propagator.current_epoch(), epoch); // Should be same initially
        assert_abs_diff_eq!(propagator.initial_state()[0], 7000e3, epsilon = 1.0);
        assert_abs_diff_eq!(propagator.initial_state()[1], 0.01, epsilon = 1e-10);
    }

    #[test]
    fn test_keplerian_propagator_step() {
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let elements = Vector6::new(7000e3, 0.0, 0.0, 0.0, 0.0, 0.0); // Circular equatorial orbit

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0, // 60 second step size
        ).unwrap();

        // Step forward
        propagator.step().unwrap();

        let new_epoch = propagator.current_epoch();
        assert_eq!(new_epoch, epoch + 60.0);

        // Mean anomaly should have advanced
        let new_state = propagator.current_state();
        assert!(new_state[5] > 0.0); // Mean anomaly should be positive

        // Trajectory should have 2 states now
        assert_eq!(propagator.trajectory().len(), 2);
    }

    #[test]
    fn test_trajectory_max_size_validation() {
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let elements = Vector6::new(7000e3, 0.0, 0.0, 0.0, 0.0, 0.0);

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        // Setting max_size to 0 should be corrected to 1
        assert!(propagator.set_eviction_policy_max_size(1).is_ok());

        // Step several times
        propagator.step().unwrap();
        propagator.step().unwrap();
        propagator.step().unwrap();

        // Should still have at least 1 state (the current one)
        assert!(propagator.trajectory().len() >= 1);
    }

    #[test]
    fn test_current_state_from_trajectory() {
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let elements = Vector6::new(7000e3, 0.0, 0.0, 0.0, 0.0, 0.0);

        let mut propagator = KeplerianPropagator::from_keplerian(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
            60.0,
        ).unwrap();

        // Initial state should match
        assert_eq!(propagator.current_state(), elements);

        // Step and check that current state is now different
        propagator.step().unwrap();
        let current_state = propagator.current_state();

        // Current state should be different from initial
        assert_ne!(current_state, elements);

        // Should match the last state in trajectory
        let last_epoch = propagator.trajectory().end_epoch().unwrap();
        let trajectory_state = propagator.trajectory().state_at_epoch(&last_epoch).unwrap();
        assert_eq!(current_state, trajectory_state);
    }
}