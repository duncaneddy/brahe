/*!
The Keplerian propagator is the simplest type of propagator, and is used to propagate the state of a
system forward in time using the Keplerian orbital elements. The Keplerian propagator is an
analytical propagator, and as such is very fast and efficient. However, it only represents the
motion of a system under the influence of a single central body, and so is not suitable for
propagating the state over long periods of time if any perturbations are present.
 */

use std::collections::BTreeMap;

use nalgebra::{SVector, Vector6};

use crate::{mean_motion, StatePropagator};
use crate::time::Epoch;
use crate::utils::{BraheError, wrap_to_2pi};

/// The `KeplerianPropagator` struct is used to propagate the state of a system forward in time using
/// analytical propgation of the Keplerian orbital elements. The state is represented as a
/// `SVector<f64, 6>`, where the elements are the Keplerian orbital elements:
/// - `a`: Semi-major axis [m]
/// - `e`: Eccentricity
/// - `i`: Inclination [rad]
/// - `Ω`: Right ascension of the ascending node [rad]
/// - `ω`: Argument of periapsis [rad]
/// - `M`: Mean anomaly [rad]
///
/// The `KeplerianPropagator` implements the `StatePropagator` trait, and so provides methods for
/// propagating the state forward in time, as well as methods for accessing the state and epoch.
pub struct KeplerianPropagator {
    initial_epoch: Epoch,
    initial_state: SVector<f64, 6>,
    last_epoch: Epoch,
    last_step: Option<f64>,
    final_epoch: Option<Epoch>,
    states: Vec<SVector<f64, 6>>,
    epoch_index: BTreeMap<Epoch, usize>,
    step_size: Option<f64>,
    mean_motion: f64,
}

impl KeplerianPropagator {
    /// Compute the Keplerian state at the given epoch. This is an internal method used to compute the
    /// Keplerian state at a given epoch. This method is used by the `step` and `step_by` methods to
    /// compute the new state after propagating the state forward in time.
    ///
    /// # Arguments
    ///
    /// - `epoch`: The epoch at which to compute the Keplerian state.
    ///
    /// # Returns
    ///
    /// The Keplerian state at the given epoch, represented as a `SVector<f64, 6>`.
    #[allow(non_snake_case)]
    fn compute_keplerian_state(&self, epoch: Epoch) -> SVector<f64, 6> {
        // Compute new mean anomaly value
        let M = self.mean_motion * (epoch - self.initial_epoch) + self.initial_state[5];

        // Compute new state
        Vector6::new(
            self.initial_state[0],
            self.initial_state[1],
            self.initial_state[2],
            self.initial_state[3],
            self.initial_state[4],
            wrap_to_2pi(M))
    }
}

impl StatePropagator<6> for KeplerianPropagator {
    /// Create a new `KeplerianPropagator` with the given initial epoch and state.
    ///
    /// # Arguments
    ///
    /// - `initial_epoch`: The initial epoch of the state.
    /// - `initial_state`: The initial state of the system, represented as a `SVector<f64, 6>` where
    ///    the elements are the Keplerian orbital elements. The elements are:
    ///     - `a`: Semi-major axis [m]
    ///     - `e`: Eccentricity
    ///     - `i`: Inclination [rad]
    ///     - `Ω`: Right ascension of the ascending node [rad]
    ///     - `ω`: Argument of periapsis [rad]
    ///     - `M`: Mean anomaly [rad]
    ///
    /// # Returns
    ///
    /// A new `KeplerianPropagator` with the given initial epoch and state.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{KeplerianPropagator, StatePropagator};
    /// use brahe::time::Epoch;
    /// use brahe::{R_EARTH, RAD2DEG, TimeSystem};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let propagator = KeplerianPropagator::new(epoch, state);
    /// ```
    fn new(initial_epoch: Epoch, initial_state: SVector<f64, 6>) -> Self {
        let mut states = Vec::new();
        let mut epoch_index = BTreeMap::new();
        states.push(initial_state.clone());
        epoch_index.insert(initial_epoch.clone(), 0);

        // Compute the mean motion from the semi-major axis so we don't need to recompute it every time
        let mean_motion = mean_motion(initial_state[0], false);

        KeplerianPropagator {
            initial_epoch,
            initial_state,
            last_step: None,
            last_epoch: initial_epoch.clone(),
            final_epoch: None,
            states,
            epoch_index,
            step_size: None,
            mean_motion,
        }
    }

    /// Get the size of the state vector.
    ///
    /// # Returns
    ///
    /// The size of the state vector.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{KeplerianPropagator, StatePropagator};
    /// use brahe::time::Epoch;
    /// use brahe::{R_EARTH, RAD2DEG, TimeSystem};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let propagator = KeplerianPropagator::new(epoch, state);
    ///
    /// assert_eq!(propagator.get_state_size(), 6);
    /// ```
    fn get_state_size(&self) -> usize {
        6
    }

    /// Get the number of states stored in the propagator.
    ///
    /// # Returns
    ///
    /// The number of states stored in the propagator.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{KeplerianPropagator, StatePropagator};
    /// use brahe::time::Epoch;
    /// use brahe::{R_EARTH, RAD2DEG, TimeSystem};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let propagator = KeplerianPropagator::new(epoch, state);
    ///
    /// assert_eq!(propagator.get_num_states(), 1);
    /// ```
    fn get_num_states(&self) -> usize {
        return self.states.len();
    }

    /// Get the initial state of the propagator.
    ///
    /// # Returns
    ///
    /// The initial state of the propagator.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{KeplerianPropagator, StatePropagator};
    /// use brahe::time::Epoch;
    /// use brahe::{R_EARTH, RAD2DEG, TimeSystem};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let propagator = KeplerianPropagator::new(epoch, state);
    ///
    /// assert_eq!(*propagator.get_initial_state(), state);
    /// ```
    fn get_initial_state(&self) -> &SVector<f64, 6> {
        &self.initial_state
    }

    /// Get the initial epoch of the propagator.
    ///
    /// # Returns
    ///
    /// The initial epoch of the propagator.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{KeplerianPropagator, StatePropagator};
    /// use brahe::time::Epoch;
    /// use brahe::{R_EARTH, RAD2DEG, TimeSystem};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let propagator = KeplerianPropagator::new(epoch, state);
    ///
    /// assert_eq!(propagator.get_initial_epoch(), epoch);
    /// ```
    fn get_initial_epoch(&self) -> Epoch {
        self.initial_epoch
    }

    /// Get the last step size of the propagator. This is the last time step used to propagate the
    /// state forward in time. If the state has not been propagated, this will return `None`.
    ///
    /// # Returns
    ///
    /// The last step size of the propagator, or `None` if the state has not been propagated.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{KeplerianPropagator, StatePropagator};
    /// use brahe::time::Epoch;
    /// use brahe::{R_EARTH, RAD2DEG, TimeSystem};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let mut propagator = KeplerianPropagator::new(epoch, state);
    ///
    /// assert_eq!(propagator.get_last_step_size(), None);
    ///
    /// propagator.set_step_size(60.0).unwrap();
    /// propagator.step().unwrap();
    /// assert_eq!(propagator.get_last_step_size(), Some(60.0));
    /// ```
    fn get_last_step_size(&self) -> Option<f64> {
        self.last_step
    }

    /// Get the last epoch of the propagator. This is the epoch of the last state that was propagated.
    /// For a newly created propagator, this will be the same as the initial epoch.
    ///
    /// # Returns
    ///
    /// The last epoch of the propagator.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{KeplerianPropagator, StatePropagator};
    /// use brahe::time::Epoch;
    /// use brahe::{R_EARTH, RAD2DEG, TimeSystem};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let propagator = KeplerianPropagator::new(epoch, state);
    ///
    /// assert_eq!(propagator.get_last_epoch(), epoch);
    /// ```
    fn get_last_epoch(&self) -> Epoch {
        self.last_epoch
    }

    /// Get the last state of the propagator. This is the state of the system at the last epoch that was
    /// propagated. For a newly created propagator, this will be the same as the initial state.
    ///
    /// # Returns
    ///
    /// The last state of the propagator.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{KeplerianPropagator, StatePropagator};
    /// use brahe::time::Epoch;
    /// use brahe::{R_EARTH, RAD2DEG, TimeSystem};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let propagator = KeplerianPropagator::new(epoch, state);
    ///
    /// assert_eq!(*propagator.get_last_state(), state);
    /// ```
    fn get_last_state(&self) -> &SVector<f64, 6> {
        // This is safe because we always have at least one state in the vector, the initial state
        self.states.last().unwrap()
    }

    /// Get the final epoch of the propagator. This is the epoch to which the propagator will propagate
    /// the state. If the final epoch has not been set, this will return `None`.
    ///
    /// # Returns
    ///
    /// The final epoch of the propagator, or `None` if the final epoch has not been set.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{KeplerianPropagator, StatePropagator};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let mut propagator = KeplerianPropagator::new(epoch, state);
    ///
    /// assert_eq!(propagator.get_final_epoch(), None);
    ///
    /// propagator.set_final_epoch(Epoch::from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC));
    /// assert_eq!(propagator.get_final_epoch(), Some(Epoch::from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC)));
    /// ```
    fn get_final_epoch(&self) -> Option<Epoch> {
        self.final_epoch
    }

    /// Get the step size of the propagator. The step size is the time step used to propagate the state
    /// forward in time. The step size is in seconds. If the step size has not been set, this will
    /// return `None`.
    ///
    /// # Returns
    ///
    /// The step size of the propagator, or `None` if the step size has not been set.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{KeplerianPropagator, StatePropagator};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let mut propagator = KeplerianPropagator::new(epoch, state);
    ///
    /// assert_eq!(propagator.get_step_size(), None);
    ///
    /// propagator.set_step_size(60.0).unwrap();
    /// assert_eq!(propagator.get_step_size(), Some(60.0));
    /// ```
    fn get_step_size(&self) -> Option<f64> {
        self.step_size
    }

    /// Get the state at the given index. If the index is out of range, this will return `None`.
    ///
    /// # Arguments
    ///
    /// - `index`: The index of the state to retrieve.
    ///
    /// # Returns
    ///
    /// The state at the given index, or `None` if the index is out of range.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{KeplerianPropagator, StatePropagator};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let mut propagator = KeplerianPropagator::new(epoch, state);
    ///
    /// assert_eq!(propagator.get_state_by_index(0), Some(&state));
    /// assert_eq!(propagator.get_state_by_index(1), None);
    /// ```
    fn get_state_by_index(&self, index: usize) -> Option<&SVector<f64, 6>> {
        if index < self.states.len() {
            Some(&self.states[index])
        } else {
            None
        }
    }

    /// Get the state at the given epoch. If the epoch is not in the propagator, this will return `None`.
    ///
    /// # Arguments
    ///
    /// - `epoch`: The epoch of the state to retrieve.
    ///
    /// # Returns
    ///
    /// The state at the given epoch, or `None` if the epoch is not in the propagator.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{KeplerianPropagator, StatePropagator};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let mut propagator = KeplerianPropagator::new(epoch, state);
    ///
    /// assert_eq!(propagator.get_state_by_epoch(epoch), Some(&state));
    /// assert_eq!(propagator.get_state_by_epoch(Epoch::from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC)), None);
    /// ```
    fn get_state_by_epoch(&self, epoch: Epoch) -> Option<&SVector<f64, 6>> {
        if let Some(index) = self.epoch_index.get(&epoch) {
            self.get_state_by_index(*index)
        } else {
            // TODO: This could be improved by implementing interpolation and returning the interpolated state
            // instead of None
            None
        }
    }

    /// Set the final epoch of the propagator. This is the epoch to which the propagator will propagate
    /// the state.
    ///
    /// # Arguments
    ///
    /// - `epoch`: The final epoch of the propagator.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success (`Ok`) or an error (`Err`).
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{KeplerianPropagator, StatePropagator};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let mut propagator = KeplerianPropagator::new(epoch, state);
    ///
    /// assert_eq!(propagator.get_final_epoch(), None);
    ///
    /// propagator.set_final_epoch(Epoch::from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC));
    /// assert_eq!(propagator.get_final_epoch(), Some(Epoch::from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC)));
    /// ```
    fn set_final_epoch(&mut self, epoch: Epoch) -> Result<(), BraheError> {
        if let Some(step_size) = self.step_size {
            // Confirm that the step size is in the same direction as the final epoch
            if (epoch - self.initial_epoch).signum() == step_size.signum() {
                self.final_epoch = Some(epoch);
            } else {
                return Err(BraheError::InitializationError("The final epoch is in the opposite direction of the step size".to_string()));
            }
        } else {
            // If the step size has not been set, simply set the final epoch
            self.final_epoch = Some(epoch);
        }

        Ok(())
    }

    /// Set the step size of the propagator. The step size is the time step used to propagate the state
    /// forward in time. The step size is in seconds.
    ///
    /// If the step size is set to a positive value, the propagator will propagate the state forward in
    /// time. If the step size is set to a negative value, the propagator will propagate the state
    /// backward in time.
    ///
    /// The step size cannot be set to zero.
    ///
    /// If the final epoch has been set, the step size must be in the same direction as the final epoch.
    ///
    /// # Arguments
    ///
    /// - `step_size`: The step size of the propagator, in seconds.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success (`Ok`) or an error (`Err`).
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{KeplerianPropagator, StatePropagator};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let mut propagator = KeplerianPropagator::new(epoch, state);
    ///
    /// assert_eq!(propagator.get_step_size(), None);
    ///
    /// propagator.set_step_size(60.0).unwrap();
    /// assert_eq!(propagator.get_step_size(), Some(60.0));
    /// ```
    fn set_step_size(&mut self, step_size: f64) -> Result<(), BraheError> {
        if step_size == 0.0 {
            return Err(BraheError::InitializationError("The step size cannot be zero".to_string()));
        }

        if let Some(final_epoch) = self.final_epoch {
            // Confirm that the step size is in the same direction as the final epoch
            if (final_epoch - self.initial_epoch).signum() == step_size.signum() {
                self.step_size = Some(step_size);
            } else {
                return Err(BraheError::InitializationError("The step size is in the opposite direction of the final epoch".to_string()));
            }
        } else {
            // If the final epoch has not been set, simply set the step size
            self.step_size = Some(step_size);
        }

        Ok(())
    }

    /// Reinitialize the propagator. This will reset the propagator to its initial state and epoch
    /// and clear all stored states.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{KeplerianPropagator, StatePropagator};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let mut propagator = KeplerianPropagator::new(epoch, state);
    ///
    /// propagator.step_by(60.0);
    /// assert_eq!(propagator.get_num_states(), 2);
    ///
    /// propagator.reinitialize();
    /// assert_eq!(propagator.get_num_states(), 1);
    /// ```
    fn reinitialize(&mut self) {
        self.last_epoch = self.initial_epoch.clone();
        self.states.truncate(1);
        self.epoch_index.clear();
        self.epoch_index.insert(self.initial_epoch.clone(), 0);
    }

    /// Propagate the state forward in time by one step size. If the step size has not been set, this
    /// will return an error.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success (`Ok`) or an error (`Err`).
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{KeplerianPropagator, StatePropagator};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let mut propagator = KeplerianPropagator::new(epoch, state);
    ///
    /// propagator.set_step_size(60.0);
    /// propagator.step().unwrap();
    /// ```
    fn step(&mut self) -> Result<(), BraheError> {
        if let Some(step_size) = self.step_size {
            let new_epoch = self.last_epoch + step_size;
            self.states.push(self.compute_keplerian_state(new_epoch));
            self.epoch_index.insert(new_epoch, self.states.len() - 1);
            self.last_epoch = new_epoch;
            self.last_step = Some(step_size);
            Ok(())
        } else {
            Err(BraheError::InitializationError("The step size has not been set".to_string()))
        }
    }

    /// Propagate the state forward in time by the given time step. If the final epoch has been set,
    /// the step must be in the same direction as the final epoch, otherwise this will return an error.
    ///
    /// # Arguments
    ///
    /// - `dt`: The time step by which to propagate the state forward in time.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success (`Ok`) or an error (`Err`).
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{KeplerianPropagator, StatePropagator};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let mut propagator = KeplerianPropagator::new(epoch, state);
    ///
    /// propagator.step_by(60.0).unwrap();
    /// ```
    fn step_by(&mut self, dt: f64) -> Result<(), BraheError> {
        if let Some(final_epoch) = self.final_epoch {
            // Confirm that the step size is in the same direction as the final epoch
            if (final_epoch - self.initial_epoch).signum() == dt.signum() {
                let epc = self.last_epoch + dt;
                self.states.push(self.compute_keplerian_state(epc));
                self.epoch_index.insert(epc, self.states.len() - 1);
                self.last_epoch = epc;
                self.last_step = Some(dt);
                Ok(())
            } else {
                Err(BraheError::PropagatorError("The provided step is in the opposite direction of the final epoch".to_string()))
            }
        } else {
            let epc = self.last_epoch + dt;
            self.states.push(self.compute_keplerian_state(epc));
            self.epoch_index.insert(epc, self.states.len() - 1);
            self.last_epoch = epc;
            self.last_step = Some(dt);
            Ok(())
        }
    }

    /// Propagate the state forward in time to the given epoch. If the final epoch has been set, the
    /// epoch must be in the same direction as the final epoch, otherwise this will return an error.
    /// Requires that the step size has been set. If the final epoch is not an integer multiple of the
    /// step size, the final step will be less than the step size.
    ///
    /// # Arguments
    ///
    /// - `epoch`: The epoch to which to propagate the state forward in time.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success (`Ok`) or an error (`Err`).
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{KeplerianPropagator, StatePropagator};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let mut propagator = KeplerianPropagator::new(epoch, state);
    ///
    /// propagator.set_step_size(60.0);
    /// propagator.step_to_epoch(Epoch::from_datetime(2024, 1, 1, 12, 3, 30.0, 0.0, TimeSystem::UTC)).unwrap();
    ///
    /// assert_eq!(propagator.get_last_epoch(), Epoch::from_datetime(2024, 1, 1, 12, 3, 30.0, 0.0, TimeSystem::UTC));
    /// assert_eq!(propagator.get_num_states(), 6);
    /// ```
    fn step_to_epoch(&mut self, epoch: Epoch) -> Result<(), BraheError> {
        if let Some(step_size) = self.step_size {
            // Confirm that the step size is in the same direction as the final epoch
            if (epoch - self.last_epoch).signum() == step_size.signum() {
                while self.last_epoch < epoch {
                    self.step_by(step_size.min(epoch - self.last_epoch))?
                }
                Ok(())
            } else {
                Err(BraheError::PropagatorError("The provided epoch is in the opposite direction of the step size".to_string()))
            }
        } else {
            Err(BraheError::InitializationError("The step size has not been set".to_string()))
        }
    }

    /// Propagate the state forward in time to the final epoch. Will return an error if the final epoch
    /// has not been set. Requires that the step size has been set. If the final epoch is not an integer
    /// multiple of the step size, the final step will be less than the step size.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success (`Ok`) or an error (`Err`).
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector6;
    /// use brahe::propagators::{KeplerianPropagator, StatePropagator};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::{R_EARTH, RAD2DEG};
    ///
    /// let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0*RAD2DEG, 15.0*RAD2DEG, 30.0*RAD2DEG, 0.0);
    /// let mut propagator = KeplerianPropagator::new(epoch, state);
    ///
    /// propagator.set_step_size(60.0);
    /// propagator.set_final_epoch(Epoch::from_datetime(2024, 1, 1, 12, 3, 30.0, 0.0, TimeSystem::UTC)).unwrap();
    /// propagator.step_to_final_epoch().unwrap();
    ///
    /// assert_eq!(propagator.get_last_epoch(), Epoch::from_datetime(2024, 1, 1, 12, 3, 30.0, 0.0, TimeSystem::UTC));
    /// assert_eq!(propagator.get_num_states(), 6);
    /// ```
    fn step_to_final_epoch(&mut self) -> Result<(), BraheError> {
        if let Some(final_epoch) = self.final_epoch {
            self.step_to_epoch(final_epoch)?;
            Ok(())
        } else {
            Err(BraheError::InitializationError("The final epoch has not been set".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use approx::assert_abs_diff_eq;

    use crate::{orbital_period, R_EARTH, RAD2DEG};
    use crate::time::TimeSystem;

    use super::*;

    #[test]
    fn test_keplerian_propagator_new() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0 * RAD2DEG, 15.0 * RAD2DEG, 30.0 * RAD2DEG, 0.0);
        let propagator = KeplerianPropagator::new(epoch, state);

        assert_eq!(propagator.get_initial_epoch(), epoch);
        assert_eq!(propagator.get_initial_state(), &state);
        assert_eq!(propagator.get_last_epoch(), epoch);
        assert_eq!(propagator.get_final_epoch(), None);
        assert_eq!(propagator.get_num_states(), 1);
    }

    #[test]
    fn test_keplerian_propagator_get_state_size() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0 * RAD2DEG, 15.0 * RAD2DEG, 30.0 * RAD2DEG, 0.0);
        let propagator = KeplerianPropagator::new(epoch, state);

        assert_eq!(propagator.get_state_size(), 6);
    }

    #[test]
    fn test_keplerian_propagator_get_num_states() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0 * RAD2DEG, 15.0 * RAD2DEG, 30.0 * RAD2DEG, 0.0);
        let mut propagator = KeplerianPropagator::new(epoch, state);

        assert_eq!(propagator.get_num_states(), 1);

        propagator.step_by(60.0).unwrap();
        assert_eq!(propagator.get_num_states(), 2);
    }

    #[test]
    fn test_keplerian_propagator_get_initial_state() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0 * RAD2DEG, 15.0 * RAD2DEG, 30.0 * RAD2DEG, 0.0);
        let propagator = KeplerianPropagator::new(epoch, state);

        assert_eq!(propagator.get_initial_state(), &Vector6::new(R_EARTH + 500e3, 0.01, 90.0 * RAD2DEG, 15.0 * RAD2DEG, 30.0 * RAD2DEG, 0.0));
    }

    #[test]
    fn test_keplerian_propagator_get_initial_epoch() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0 * RAD2DEG, 15.0 * RAD2DEG, 30.0 * RAD2DEG, 0.0);
        let propagator = KeplerianPropagator::new(epoch, state);

        assert_eq!(propagator.get_initial_epoch(), Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC));
    }

    #[test]
    fn test_keplerian_propagator_get_last_step_size() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0 * RAD2DEG, 15.0 * RAD2DEG, 30.0 * RAD2DEG, 0.0);
        let mut propagator = KeplerianPropagator::new(epoch, state);

        assert_eq!(propagator.get_last_step_size(), None);

        propagator.set_step_size(60.0).unwrap();
        propagator.step().unwrap();
        assert_eq!(propagator.get_last_step_size(), Some(60.0));
    }

    #[test]
    fn test_keplerian_propagator_get_last_epoch() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0 * RAD2DEG, 15.0 * RAD2DEG, 30.0 * RAD2DEG, 0.0);
        let mut propagator = KeplerianPropagator::new(epoch, state);

        assert_eq!(propagator.get_last_epoch(), epoch);

        propagator.step_by(60.0).unwrap();
        assert_eq!(propagator.get_last_epoch(), epoch + 60);
    }

    #[test]
    fn test_keplerian_propagator_get_final_epoch() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0 * RAD2DEG, 15.0 * RAD2DEG, 30.0 * RAD2DEG, 0.0);
        let mut propagator = KeplerianPropagator::new(epoch, state);

        assert_eq!(propagator.get_final_epoch(), None);

        propagator.set_final_epoch(Epoch::from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC)).unwrap();
        assert_eq!(propagator.get_final_epoch(), Some(Epoch::from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC)));
    }

    #[test]
    fn test_keplerian_propagator_get_step_size() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(R_EARTH + 500e3, 0.01, 90.0 * RAD2DEG, 15.0 * RAD2DEG, 30.0 * RAD2DEG, 0.0);
        let mut propagator = KeplerianPropagator::new(epoch, state);

        assert_eq!(propagator.get_step_size(), None);

        propagator.set_step_size(60.0).unwrap();
        assert_eq!(propagator.get_step_size(), Some(60.0));
    }

    #[test]
    fn test_keplerian_propagator_get_state_by_index() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let sma = R_EARTH + 500e3;
        let state = Vector6::new(sma, 0.01, 90.0 * RAD2DEG, 15.0 * RAD2DEG, 30.0 * RAD2DEG, 0.0);
        let mut propagator = KeplerianPropagator::new(epoch, state);

        assert_eq!(propagator.get_state_by_index(0), Some(&state));
        assert_eq!(propagator.get_state_by_index(1), None);

        propagator.step_by(orbital_period(sma)).unwrap();
        let state2 = propagator.get_state_by_index(1).unwrap();
        assert_eq!(state2[0], state[0]);
        assert_eq!(state2[1], state[1]);
        assert_eq!(state2[2], state[2]);
        assert_eq!(state2[3], state[3]);
        assert_eq!(state2[4], state[4]);
        assert_abs_diff_eq!(state2[5], 2.0 * PI, epsilon = 1e-14);
    }

    #[test]
    fn test_keplerian_propagator_get_state_by_epoch() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let sma = R_EARTH + 500e3;
        let state = Vector6::new(sma, 0.01, 90.0 * RAD2DEG, 15.0 * RAD2DEG, 30.0 * RAD2DEG, 0.0);
        let mut propagator = KeplerianPropagator::new(epoch, state);

        assert_eq!(propagator.get_state_by_epoch(epoch), Some(&state));
        assert_eq!(propagator.get_state_by_epoch(Epoch::from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC)), None);

        let period = orbital_period(sma);
        propagator.step_by(period).unwrap();
        let state2 = propagator.get_state_by_epoch(epoch + period).unwrap();
        assert_eq!(state2[0], state[0]);
        assert_eq!(state2[1], state[1]);
        assert_eq!(state2[2], state[2]);
        assert_eq!(state2[3], state[3]);
        assert_eq!(state2[4], state[4]);
        assert_abs_diff_eq!(state2[5], 2.0 * PI, epsilon = 1e-14);
    }

    #[test]
    fn test_keplerian_propagator_set_final_epoch() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let mut propagator = KeplerianPropagator::new(epoch, Vector6::zeros());

        assert_eq!(propagator.get_final_epoch(), None);

        propagator.set_final_epoch(Epoch::from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC)).unwrap();
        assert_eq!(propagator.get_final_epoch(), Some(Epoch::from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC)));
    }

    #[test]
    fn test_keplerian_propagator_set_step_size() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let mut propagator = KeplerianPropagator::new(epoch, Vector6::zeros());

        assert_eq!(propagator.get_step_size(), None);

        propagator.set_step_size(60.0).unwrap();
        assert_eq!(propagator.get_step_size(), Some(60.0));
    }

    #[test]
    fn test_keplerian_propagator_reinitialize() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let mut propagator = KeplerianPropagator::new(epoch, Vector6::zeros());

        propagator.step_by(60.0).unwrap();
        assert_eq!(propagator.get_num_states(), 2);

        propagator.reinitialize();
        assert_eq!(propagator.get_num_states(), 1);
    }

    #[test]
    fn test_keplerian_propagator_step() {
        let sma = R_EARTH + 500e3;
        let period = orbital_period(sma);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(sma, 0.01, 90.0 * RAD2DEG, 15.0 * RAD2DEG, 30.0 * RAD2DEG, 0.0);

        let mut propagator = KeplerianPropagator::new(epoch, state);
        propagator.set_step_size(period / 10.0).unwrap();

        for _ in 0..10 {
            propagator.step().unwrap();
        }

        assert_eq!(propagator.get_num_states(), 11);
        assert_eq!(propagator.get_last_epoch(), epoch + period);
        assert_eq!(propagator.get_last_step_size(), Some(period / 10.0));
        let state2 = propagator.get_last_state();
        assert_eq!(state2[0], state[0]);
        assert_eq!(state2[1], state[1]);
        assert_eq!(state2[2], state[2]);
        assert_eq!(state2[3], state[3]);
        assert_eq!(state2[4], state[4]);
        assert_abs_diff_eq!(state2[5], 2.0 * PI, epsilon = 1e-14);
    }

    #[test]
    fn test_keplerian_propagator_step_by() {
        let sma = R_EARTH + 500e3;
        let period = orbital_period(sma);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(sma, 0.01, 90.0 * RAD2DEG, 15.0 * RAD2DEG, 30.0 * RAD2DEG, 0.0);

        let mut propagator = KeplerianPropagator::new(epoch, state);
        propagator.set_step_size(period / 10.0).unwrap();

        // Step by something other than the set step
        propagator.step_by(period).unwrap();

        assert_eq!(propagator.get_num_states(), 2);
        assert_eq!(propagator.get_last_epoch(), epoch + period);
        assert_eq!(propagator.get_last_step_size(), Some(period));
        let state2 = propagator.get_last_state();
        assert_eq!(state2[0], state[0]);
        assert_eq!(state2[1], state[1]);
        assert_eq!(state2[2], state[2]);
        assert_eq!(state2[3], state[3]);
        assert_eq!(state2[4], state[4]);
        assert_abs_diff_eq!(state2[5], 2.0 * PI, epsilon = 1e-14);
    }

    #[test]
    fn test_keplerian_propagator_step_to_epoch() {
        let sma = R_EARTH + 500e3;
        let period = orbital_period(sma);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(sma, 0.01, 90.0 * RAD2DEG, 15.0 * RAD2DEG, 30.0 * RAD2DEG, 0.0);

        let mut propagator = KeplerianPropagator::new(epoch, state);
        propagator.set_step_size(period / 10.0).unwrap();

        propagator.step_to_epoch(epoch + period).unwrap();

        assert_eq!(propagator.get_num_states(), 11);
        assert_eq!(propagator.get_last_epoch(), epoch + period);
        assert_abs_diff_eq!(propagator.get_last_step_size().unwrap(), period / 10.0, epsilon = 1e-12);
        let state2 = propagator.get_last_state();
        assert_eq!(state2[0], state[0]);
        assert_eq!(state2[1], state[1]);
        assert_eq!(state2[2], state[2]);
        assert_eq!(state2[3], state[3]);
        assert_eq!(state2[4], state[4]);
        assert_abs_diff_eq!(state2[5], 2.0 * PI, epsilon = 1e-14);
    }

    #[test]
    fn test_keplerian_propagator_step_to_final_epoch() {
        let sma = R_EARTH + 500e3;
        let period = orbital_period(sma);
        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(sma, 0.01, 90.0 * RAD2DEG, 15.0 * RAD2DEG, 30.0 * RAD2DEG, 0.0);

        let mut propagator = KeplerianPropagator::new(epoch, state);
        propagator.set_step_size(period / 10.0).unwrap();
        propagator.set_final_epoch(epoch + period).unwrap();

        propagator.step_to_final_epoch().unwrap();

        assert_eq!(propagator.get_num_states(), 11);
        assert_eq!(propagator.get_last_epoch(), epoch + period);
        assert_abs_diff_eq!(propagator.get_last_step_size().unwrap(), period / 10.0, epsilon = 1e-12);
        let state2 = propagator.get_last_state();
        assert_eq!(state2[0], state[0]);
        assert_eq!(state2[1], state[1]);
        assert_eq!(state2[2], state[2]);
        assert_eq!(state2[3], state[3]);
        assert_eq!(state2[4], state[4]);
        assert_abs_diff_eq!(state2[5], 2.0 * PI, epsilon = 1e-14);
    }
}