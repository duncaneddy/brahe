/*!
 * Propagator traits with clean interfaces and vector-based operations
 */

use nalgebra::Vector6;

use crate::time::Epoch;
use crate::utils::BraheError;
use crate::trajectories::{OrbitFrame, OrbitRepresentation, AngleFormat, Trajectory6};

/// Core trait for orbit propagators with clean interface
pub trait OrbitPropagator {
    /// Step forward by the default step size
    /// Returns Result indicating success/failure, use getters to access state
    fn step(&mut self) -> Result<(), BraheError>;

    /// Step forward by a specified time duration
    ///
    /// # Arguments
    /// * `step_size` - Time step in seconds
    fn step_by(&mut self, step_size: f64) -> Result<(), BraheError>;

    /// Step forward by default step size for a specified number of steps
    ///
    /// # Arguments
    /// * `num_steps` - Number of steps to take
    fn propagate_steps(&mut self, num_steps: usize) -> Result<(), BraheError>;

    /// Propagate to a specific target epoch
    ///
    /// # Arguments
    /// * `target_epoch` - The epoch to propagate to
    fn propagate_to(&mut self, target_epoch: Epoch) -> Result<(), BraheError>;

    // Getter methods for accessing state
    /// Get current state as a 6D vector
    fn current_state(&self) -> Vector6<f64>;

    /// Get current epoch
    fn current_epoch(&self) -> Epoch;

    /// Get initial state as a 6D vector
    fn initial_state(&self) -> Vector6<f64>;

    /// Get initial epoch
    fn initial_epoch(&self) -> Epoch;

    // Configuration methods
    /// Get step size in seconds
    fn step_size(&self) -> f64;

    /// Set step size in seconds
    fn set_step_size(&mut self, step_size: f64);

    /// Reset propagator to initial conditions
    fn reset(&mut self) -> Result<(), BraheError>;

    /// Set initial conditions from components
    ///
    /// # Arguments
    /// * `epoch` - Initial epoch
    /// * `state` - 6-element state vector
    /// * `frame` - Reference frame
    /// * `representation` - Type of orbital representation
    /// * `angle_format` - Format for angular elements
    fn set_initial_conditions(
        &mut self,
        epoch: Epoch,
        state: Vector6<f64>,
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: AngleFormat,
    ) -> Result<(), BraheError>;

    // Trajectory access
    /// Get reference to accumulated trajectory
    fn trajectory(&self) -> &Trajectory6;

    /// Get mutable reference to accumulated trajectory
    fn trajectory_mut(&mut self) -> &mut Trajectory6;

    /// Propagate and populate trajectory at multiple epochs
    ///
    /// # Arguments
    /// * `epochs` - Epochs to propagate to and add to trajectory
    fn propagate_trajectory(&mut self, epochs: &[Epoch]) -> Result<(), BraheError> {
        for &epoch in epochs {
            self.propagate_to(epoch)?;
        }
        Ok(())
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
pub trait AnalyticPropagator {
    /// Returns the state at the given epoch as a 6-element vector in the propagator's
    /// native coordinate frame and representation.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// A 6-element vector containing the state in the propagator's native format
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
    ///
    /// # Returns
    /// A 6-element vector containing osculating Keplerian elements:
    /// [semi-major axis (km), eccentricity, inclination (rad), RAAN (rad),
    ///  argument of perigee (rad), mean anomaly (rad)]
    fn state_osculating_elements(&self, epoch: Epoch) -> Vector6<f64>;

    /// Returns states at multiple epochs as an OrbitalTrajectory in the propagator's
    /// native coordinate frame and representation.
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// Trajectory6 containing states in the propagator's native format
    fn states(&self, epochs: &[Epoch]) -> Trajectory6;

    /// Returns states at multiple epochs in Earth-Centered Inertial (ECI)
    /// Cartesian coordinates as a Trajectory6.
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// Trajectory6 containing states in the ECI frame
    fn states_eci(&self, epochs: &[Epoch]) -> Trajectory6;

    /// Returns states at multiple epochs in Earth-Centered Earth-Fixed (ECEF)
    /// Cartesian coordinates as a Trajectory6.
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// Trajectory6 containing states in the ECEF frame
    fn states_ecef(&self, epochs: &[Epoch]) -> Trajectory6;

    /// Returns states at multiple epochs as osculating orbital elements as a Trajectory6.
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// Trajectory6 containing states as osculating Keplerian elements
    fn states_osculating_elements(&self, epochs: &[Epoch]) -> Trajectory6;
}