/*!
 * Common traits for trajectory implementations.
 *
 * This module defines the core traits that both static (compile-time sized) and dynamic
 * (runtime sized) trajectory implementations must implement to ensure a consistent interface.
 */

use crate::time::Epoch;
use crate::utils::BraheError;

/// Core trajectory functionality that all trajectory implementations must provide.
///
/// This trait defines the essential operations for storing, retrieving, and managing
/// trajectory state data over time, regardless of the underlying storage mechanism
/// (compile-time vs runtime sized vectors).
pub trait TrajectoryCore {
    /// The type used to represent state vectors
    type StateVector;

    /// Create a new empty trajectory with default settings
    fn new() -> Self;

    /// Add a state vector at a specific epoch
    ///
    /// # Arguments
    /// * `epoch` - Time epoch for the state
    /// * `state` - State vector to add
    ///
    /// # Returns
    /// * `Ok(())` - State successfully added
    /// * `Err(BraheError)` - If addition fails (e.g., dimension mismatch)
    fn add_state(&mut self, epoch: Epoch, state: Self::StateVector) -> Result<(), BraheError>;

    /// Get the state vector at a specific epoch using interpolation
    ///
    /// # Arguments
    /// * `epoch` - Target epoch for state retrieval
    ///
    /// # Returns
    /// * `Ok(state)` - Interpolated state vector at the epoch
    /// * `Err(BraheError)` - If interpolation fails or epoch is out of range
    fn state_at_epoch(&self, epoch: &Epoch) -> Result<Self::StateVector, BraheError>;

    /// Get the state vector at a specific index
    ///
    /// # Arguments
    /// * `index` - Index of the state to retrieve
    ///
    /// # Returns
    /// * `Ok(state)` - State vector at the index
    /// * `Err(BraheError)` - If index is out of bounds
    fn state_at_index(&self, index: usize) -> Result<Self::StateVector, BraheError>;

    /// Get the epoch at a specific index
    ///
    /// # Arguments
    /// * `index` - Index of the epoch to retrieve
    ///
    /// # Returns
    /// * `Ok(epoch)` - Epoch at the index
    /// * `Err(BraheError)` - If index is out of bounds
    fn epoch_at_index(&self, index: usize) -> Result<Epoch, BraheError>;

    /// Find the nearest state to a given epoch
    ///
    /// # Arguments
    /// * `epoch` - Target epoch to find nearest state for
    ///
    /// # Returns
    /// * `Ok((epoch, state))` - Nearest epoch and corresponding state
    /// * `Err(BraheError)` - If trajectory is empty
    fn nearest_state(&self, epoch: &Epoch) -> Result<(Epoch, Self::StateVector), BraheError>;

    /// Get the number of states in the trajectory
    fn len(&self) -> usize;

    /// Check if the trajectory is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the first epoch in the trajectory, if any
    fn start_epoch(&self) -> Option<Epoch>;

    /// Get the last epoch in the trajectory, if any
    fn end_epoch(&self) -> Option<Epoch>;

    /// Get the time span covered by the trajectory in seconds
    fn timespan(&self) -> Option<f64>;

    /// Get the first (epoch, state) pair in the trajectory, if any
    fn first(&self) -> Option<(Epoch, Self::StateVector)>;

    /// Get the last (epoch, state) pair in the trajectory, if any
    fn last(&self) -> Option<(Epoch, Self::StateVector)>;

    /// Clear all states from the trajectory
    fn clear(&mut self);
}

/// Extended trajectory functionality for more advanced operations.
///
/// This trait provides additional methods that may not be implemented by all
/// trajectory types, such as removal operations and bulk data operations.
pub trait TrajectoryExtended: TrajectoryCore {
    /// Remove a state at a specific epoch
    ///
    /// # Arguments
    /// * `epoch` - Epoch of the state to remove
    ///
    /// # Returns
    /// * `Ok(state)` - The removed state vector
    /// * `Err(BraheError)` - If epoch not found
    fn remove_state(&mut self, epoch: &Epoch) -> Result<Self::StateVector, BraheError>;

    /// Remove a state at a specific index
    ///
    /// # Arguments
    /// * `index` - Index of the state to remove
    ///
    /// # Returns
    /// * `Ok((epoch, state))` - The removed epoch and state
    /// * `Err(BraheError)` - If index is out of bounds
    fn remove_state_at_index(&mut self, index: usize) -> Result<(Epoch, Self::StateVector), BraheError>;

    /// Get both epoch and state at a specific index
    ///
    /// # Arguments
    /// * `index` - Index to retrieve
    ///
    /// # Returns
    /// * `Ok((epoch, state))` - Epoch and state at the index
    /// * `Err(BraheError)` - If index is out of bounds
    fn get(&self, index: usize) -> Result<(Epoch, Self::StateVector), BraheError>;
}