/*!
 * Propagator traits with clean interfaces and vector-based operations
 */

use nalgebra::Vector6;

use crate::constants::AngleFormat;
use crate::time::Epoch;
use crate::trajectories::traits::{OrbitFrame, OrbitRepresentation};
use crate::utils::BraheError;
use crate::utils::identifiable::Identifiable;

/// Core trait for orbit propagators with clean interface
pub trait OrbitPropagator {
    /// Step forward by the default step size
    /// Returns Result indicating success/failure, use getters to access state
    fn step(&mut self) {
        self.step_by(self.step_size());
    }

    /// Step forward by a specified time duration
    ///
    /// # Arguments
    /// * `step_size` - Time step in seconds
    fn step_by(&mut self, step_size: f64);

    /// Step past a specified target epoch
    /// If the target epoch is before or equal to the current epoch, no action is taken
    fn step_past(&mut self, target_epoch: Epoch) {
        while self.current_epoch() < target_epoch {
            self.step();
        }
    }

    /// Step forward by default step size for a specified number of steps
    ///
    /// # Arguments
    /// * `num_steps` - Number of steps to take
    fn propagate_steps(&mut self, num_steps: usize) {
        for _ in 0..num_steps {
            self.step();
        }
    }

    /// Propagate to a specific target epoch
    ///
    /// # Arguments
    /// * `target_epoch` - The epoch to propagate to
    fn propagate_to(&mut self, target_epoch: Epoch) {
        let mut current_epoch = self.current_epoch();

        while current_epoch < target_epoch {
            // Calculate step size to not overshoot
            let remaining_time = target_epoch - current_epoch;
            let step_size = remaining_time.min(self.step_size());

            // Guard against very small steps to avoid infinite loops
            if step_size <= 1e-9 {
                break;
            }

            self.step_by(step_size);
            current_epoch = self.current_epoch();
        }
    }

    // Getter methods for accessing state

    /// Get current epoch
    fn current_epoch(&self) -> Epoch;

    /// Get current state as a 6D vector
    fn current_state(&self) -> Vector6<f64>;

    /// Get initial epoch
    fn initial_epoch(&self) -> Epoch;

    /// Get initial state as a 6D vector
    fn initial_state(&self) -> Vector6<f64>;

    // Configuration methods
    /// Get step size in seconds
    fn step_size(&self) -> f64;

    /// Set step size in seconds
    fn set_step_size(&mut self, step_size: f64);

    /// Reset propagator to initial conditions
    fn reset(&mut self);

    /// Set initial conditions from components
    ///
    /// # Arguments
    /// * `epoch` - Initial epoch
    /// * `state` - 6-element state vector
    /// * `frame` - Reference frame
    /// * `representation` - Type of orbital representation
    /// * `angle_format` - Format for angular elements (None for Cartesian, Some(format) for Keplerian)
    ///
    /// # Panics
    /// May panic if the combination of frame, representation, and angle_format is incompatible
    fn set_initial_conditions(
        &mut self,
        epoch: Epoch,
        state: Vector6<f64>,
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: Option<AngleFormat>,
    );

    /// Propagate and populate trajectory at multiple epochs
    ///
    /// # Arguments
    /// * `epochs` - Epochs to propagate to and add to trajectory
    fn propagate_trajectory(&mut self, epochs: &[Epoch]) {
        for &epoch in epochs {
            self.propagate_to(epoch);
        }
    }

    // Memory management for trajectory
    /// Set eviction policy to keep a maximum number of states
    fn set_eviction_policy_max_size(&mut self, max_size: usize) -> Result<(), BraheError>;

    /// Set eviction policy to keep states within a maximum age
    fn set_eviction_policy_max_age(&mut self, max_age: f64) -> Result<(), BraheError>;
}

/// Trait for analytic orbital propagators that can compute states directly at any epoch
/// without requiring numerical integration. This trait is designed for propagators like
/// SGP4/TLE that have closed-form solutions.
pub trait StateProvider {
    /// Returns the state at the given epoch as a 6-element vector in the propagator's
    /// native coordinate frame and representation.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// A 6-element vector containing the state in the propagator's native output format
    fn state(&self, epoch: Epoch) -> Vector6<f64>;

    /// Returns the state at the given epoch in Earth-Centered Inertial (ECI)
    /// Cartesian coordinates.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// A 6-element vector containing position (km) and velocity (km/s) components
    /// in the ECI frame.
    fn state_eci(&self, epoch: Epoch) -> Vector6<f64>;

    /// Returns the state at the given epoch in Earth-Centered Earth-Fixed (ECEF)
    /// Cartesian coordinates.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// A 6-element vector containing position (km) and velocity (km/s) components
    /// in the ECEF frame.
    fn state_ecef(&self, epoch: Epoch) -> Vector6<f64>;

    /// Returns the state at the given epoch as osculating orbital elements.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    /// * `angle_format` - Angle format for angular elements (Degrees or Radians)
    ///
    /// # Returns
    /// A 6-element vector containing osculating Keplerian elements [a, e, i, RAAN, arg_periapsis, mean_anomaly]
    /// where angles are in the specified format
    fn state_as_osculating_elements(&self, epoch: Epoch, angle_format: AngleFormat)
    -> Vector6<f64>;

    /// Returns states at multiple epochs in the propagator's native coordinate frame
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * Vector of 6-element vectors containing states in the propagator's native output format
    fn states(&self, epochs: &[Epoch]) -> Vec<Vector6<f64>> {
        epochs.iter().map(|&epoch| self.state(epoch)).collect()
    }

    /// Returns states at multiple epochs in Earth-Centered Inertial (ECI)
    /// Cartesian coordinates as a STrajectory6.
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * Vector of 6-element vectors containing position (m) and velocity (m/s) components
    fn states_eci(&self, epochs: &[Epoch]) -> Vec<Vector6<f64>> {
        epochs.iter().map(|&epoch| self.state_eci(epoch)).collect()
    }

    /// Returns states at multiple epochs in Earth-Centered Earth-Fixed (ECEF)
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * Vector of 6-element vectors containing position (m) and velocity (m/s) components
    fn states_ecef(&self, epochs: &[Epoch]) -> Vec<Vector6<f64>> {
        epochs.iter().map(|&epoch| self.state_ecef(epoch)).collect()
    }

    /// Returns states at multiple epochs as osculating orbital elements.
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    /// * `angle_format` - Angle format for angular elements (Degrees or Radians)
    ///
    /// # Returns
    /// * Vector of 6-element vectors containing osculating Keplerian elements
    fn states_as_osculating_elements(
        &self,
        epochs: &[Epoch],
        angle_format: AngleFormat,
    ) -> Vec<Vector6<f64>> {
        epochs
            .iter()
            .map(|&epoch| self.state_as_osculating_elements(epoch, angle_format))
            .collect()
    }
}

/// Combined trait for state providers with identity tracking.
///
/// This supertrait combines `StateProvider` and `Identifiable`, used primarily
/// in access computation where satellite identity needs to be tracked alongside
/// orbital state computation.
///
/// # Automatic Implementation
///
/// This trait is automatically implemented for any type that implements both
/// `StateProvider` and `Identifiable` via a blanket implementation.
///
/// # Examples
///
/// ```
/// use brahe::orbits::{KeplerianPropagator, SGPPropagator};
/// use brahe::traits::IdentifiableStateProvider;
///
/// // Both propagators implement IdentifiableStateProvider automatically
/// fn accepts_identified_provider<P: IdentifiableStateProvider>(provider: &P) {
///     // Can use both StateProvider and Identifiable methods
/// }
/// ```
pub trait IdentifiableStateProvider: StateProvider + Identifiable {}

// Blanket implementation for any type implementing both traits
impl<T: StateProvider + Identifiable> IdentifiableStateProvider for T {}
