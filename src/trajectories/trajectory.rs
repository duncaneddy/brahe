/*!
 * Base trajectory implementation for storing and interpolating 6-dimensional state vectors over time.
 *
 * This module provides a frame-agnostic trajectory container that stores epochs and corresponding
 * 6-dimensional state vectors. The trajectory supports various interpolation methods, memory management
 * policies, and efficient access patterns for orbital mechanics applications.
 *
 * # Key Features
 * - Frame-agnostic storage (no assumptions about coordinate frames)
 * - Multiple interpolation methods (linear, cubic spline, Lagrange, etc.)
 * - Memory management with configurable eviction policies
 * - Efficient nearest-state and exact-epoch lookups
 * - Serialization support for persistence
 *
 * # Examples
 * ```rust
 * use brahe::trajectories::{Trajectory6, InterpolationMethod};
 * use brahe::time::{Epoch, TimeSystem};
 * use nalgebra::Vector6;
 *
 * let mut traj = Trajectory6::new();
 * let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
 * let state = Vector6::new(6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0);
 * traj.add_state(epoch, state).unwrap();
 * ```
 */

use nalgebra::SVector;
use serde::{Deserialize, Serialize};
use std::ops::Index;

use crate::time::Epoch;
use crate::utils::BraheError;

/// Type alias for a 3-dimensional trajectory (e.g., position only)
pub type Trajectory3 = Trajectory<3>;

/// Type alias for a 4-dimensional trajectory (e.g., quaternion)
pub type Trajectory4 = Trajectory<4>;

/// Type alias for a 6-dimensional trajectory (commonly used for orbital mechanics)
pub type Trajectory6 = Trajectory<6>;

/// Interpolation methods for retrieving trajectory states at arbitrary epochs.
///
/// Different methods provide varying trade-offs between computational cost and accuracy.
/// For most applications, linear interpolation provides sufficient accuracy with minimal
/// computational overhead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterpolationMethod {
    /// Linear interpolation between adjacent states.
    /// Good balance of speed and accuracy for smooth trajectories.
    Linear,
    /// Cubic spline interpolation using natural boundary conditions.
    /// Higher accuracy for smooth trajectories but requires more computation.
    CubicSpline,
    /// Lagrange polynomial interpolation using nearby points.
    /// High accuracy but can exhibit oscillatory behavior with noisy data.
    Lagrange,
    /// Hermite interpolation preserving first derivatives.
    /// Excellent for smooth trajectories with known velocity information.
    Hermite,
}

impl Default for InterpolationMethod {
    fn default() -> Self {
        InterpolationMethod::Linear
    }
}

/// Enumeration of trajectory eviction policies for memory management
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrajectoryEvictionPolicy {
    /// No eviction - trajectory grows unbounded
    None,
    /// Keep most recent states, evict oldest when limit reached
    KeepCount,
    /// Keep states within a time duration from current epoch
    KeepWithinDuration,
}

impl Default for TrajectoryEvictionPolicy {
    fn default() -> Self {
        TrajectoryEvictionPolicy::None
    }
}

/// Frame-agnostic trajectory container for N-dimensional state vectors over time.
///
/// The trajectory maintains a chronologically sorted collection of epochs and corresponding
/// state vectors. State vectors can be of any dimension N, typically either Cartesian
/// position/velocity (6D: x, y, z, vx, vy, vz) or orbital elements (6D: a, e, i, Ω, ω, M),
/// but the interpretation is left to higher-level containers like `OrbitalTrajectory`.
///
/// # Memory Management
/// The trajectory supports automatic memory management through configurable policies:
/// - Maximum state count (oldest states evicted first)
/// - Maximum age (states older than threshold evicted)
/// - Custom eviction policies for specialized use cases
///
/// # Performance Characteristics
/// - State insertion: O(log n) due to maintaining sorted order
/// - Nearest state lookup: O(log n) binary search
/// - Interpolation: O(1) for linear, O(k) for polynomial methods
///
/// # Thread Safety
/// This structure is not thread-safe. Use appropriate synchronization for concurrent access.
#[derive(Debug, Clone, PartialEq)]
pub struct Trajectory<const R: usize = 6>
{
    /// Time epochs for each state, maintained in chronological order.
    /// All epochs should use consistent time systems for meaningful interpolation.
    pub epochs: Vec<Epoch>,

    /// R-dimensional state vectors corresponding to epochs.
    /// Units and interpretation depend on the specific use case:
    /// - Cartesian: [m, m, m, m/s, m/s, m/s]
    /// - Keplerian: [m, dimensionless, rad, rad, rad, rad]
    pub states: Vec<SVector<f64, R>>,

    /// Interpolation method for state retrieval at arbitrary epochs.
    /// Default is linear interpolation for optimal performance/accuracy balance.
    pub interpolation_method: InterpolationMethod,

    /// Memory management policy for automatic state eviction.
    /// Controls how states are removed when limits are exceeded.
    pub eviction_policy: TrajectoryEvictionPolicy,

    /// Maximum number of states to retain (for KeepCount policy).
    max_size: Option<usize>,

    /// Maximum age of states to retain in seconds (for KeepWithinDuration policy).
    max_age: Option<f64>,
}

impl<const R: usize> Default for Trajectory<R>
{
    /// Creates a trajectory with default settings (linear interpolation, no memory limits).
    fn default() -> Self {
        Self::new()
    }
}

impl<const R: usize> Trajectory<R>
{
    /// Creates a new empty trajectory with default linear interpolation.
    ///
    /// This is the most convenient method for creating trajectories. The interpolation
    /// method can be changed later using `set_interpolation_method()`.
    ///
    /// # Returns
    /// A new empty trajectory with linear interpolation
    ///
    /// # Examples
    /// ```rust
    /// use brahe::trajectories::{Trajectory, Trajectory6};
    /// let traj = Trajectory6::new(); // 6-dimensional trajectory
    /// assert_eq!(traj.len(), 0);
    ///
    /// // Or specify dimension explicitly
    /// let traj: Trajectory<3> = Trajectory::new(); // 3-dimensional trajectory
    /// assert_eq!(traj.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            epochs: Vec::new(),
            states: Vec::new(),
            interpolation_method: InterpolationMethod::Linear,
            eviction_policy: TrajectoryEvictionPolicy::None,
            max_size: None,
            max_age: None,
        }
    }

    /// Creates a new empty trajectory with the specified interpolation method.
    ///
    /// # Arguments
    /// * `interpolation_method` - Method to use for state interpolation between epochs
    ///
    /// # Returns
    /// A new empty trajectory ready for state addition
    ///
    /// # Examples
    /// ```rust
    /// use brahe::trajectories::{Trajectory6, InterpolationMethod};
    /// let traj = Trajectory6::with_interpolation(InterpolationMethod::CubicSpline);
    /// assert_eq!(traj.len(), 0);
    /// ```
    pub fn with_interpolation(interpolation_method: InterpolationMethod) -> Self {
        Self {
            epochs: Vec::new(),
            states: Vec::new(),
            interpolation_method,
            eviction_policy: TrajectoryEvictionPolicy::None,
            max_size: None,
            max_age: None,
        }
    }

    /// Creates a trajectory from existing epoch and state data.
    ///
    /// The input data is validated and sorted chronologically. This method is efficient
    /// for bulk trajectory construction from pre-existing datasets.
    ///
    /// # Arguments
    /// * `epochs` - Vector of time epochs (must be non-empty and same length as states)
    /// * `states` - Vector of 6D state vectors corresponding to epochs
    /// * `interpolation_method` - Method to use for state interpolation
    ///
    /// # Returns
    /// * `Ok(Trajectory)` - Successfully created trajectory with sorted data
    /// * `Err(BraheError)` - If input validation fails
    ///
    /// # Errors
    /// * `BraheError::Error` - If epochs and states have different lengths or are empty
    ///
    /// # Examples
    /// ```rust
    /// use brahe::trajectories::{Trajectory6, InterpolationMethod};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use nalgebra::Vector6;
    ///
    /// let epochs = vec![Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC)];
    /// let states = vec![Vector6::new(6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0)];
    /// let traj = Trajectory6::from_data(epochs, states, InterpolationMethod::Linear).unwrap();
    /// ```
    pub fn from_data(
        epochs: Vec<Epoch>,
        states: Vec<SVector<f64, R>>,
        interpolation_method: InterpolationMethod,
    ) -> Result<Self, BraheError> {
        if epochs.len() != states.len() {
            return Err(BraheError::Error(
                "Epochs and states vectors must have the same length".to_string(),
            ));
        }

        if epochs.is_empty() {
            return Err(BraheError::Error(
                "Cannot create trajectory from empty data".to_string(),
            ));
        }

        // Ensure epochs are sorted
        let mut indices: Vec<usize> = (0..epochs.len()).collect();
        indices.sort_by(|&i, &j| epochs[i].partial_cmp(&epochs[j]).unwrap());

        let sorted_epochs: Vec<Epoch> = indices.iter().map(|&i| epochs[i]).collect();
        let sorted_states: Vec<SVector<f64, R>> = indices.iter().map(|&i| states[i].clone()).collect();

        Ok(Self {
            epochs: sorted_epochs,
            states: sorted_states,
            interpolation_method,
            eviction_policy: TrajectoryEvictionPolicy::None,
            max_size: None,
            max_age: None,
        })
    }

    /// Set the interpolation method for state retrieval.
    ///
    /// This allows changing the interpolation behavior after trajectory creation.
    /// The change affects all future calls to `state_at_epoch()` and related methods.
    ///
    /// # Arguments
    /// * `method` - New interpolation method to use
    ///
    /// # Examples
    /// ```rust
    /// use brahe::trajectories::{Trajectory6, InterpolationMethod};
    /// let mut traj = Trajectory6::new(); // defaults to Linear
    /// traj.set_interpolation_method(InterpolationMethod::CubicSpline);
    /// ```
    pub fn set_interpolation_method(&mut self, method: InterpolationMethod) {
        self.interpolation_method = method;
    }

    /// Set eviction policy to keep a maximum number of states.
    ///
    /// # Arguments
    /// * `max_size` - Maximum number of states to retain (must be >= 1)
    ///
    /// # Errors
    /// * `BraheError::Error` - If max_size is less than 1
    pub fn set_eviction_policy_max_size(&mut self, max_size: usize) -> Result<(), BraheError> {
        if max_size < 1 {
            return Err(BraheError::Error(
                "Maximum size must be >= 1".to_string(),
            ));
        }
        self.eviction_policy = TrajectoryEvictionPolicy::KeepCount;
        self.max_size = Some(max_size);
        self.max_age = None;
        self.apply_eviction_policy()?;
        Ok(())
    }

    /// Set eviction policy to keep states within a maximum age.
    ///
    /// # Arguments
    /// * `max_age` - Maximum age of states to retain in seconds (must be > 0.0)
    ///
    /// # Errors
    /// * `BraheError::Error` - If max_age is not positive
    pub fn set_eviction_policy_max_age(&mut self, max_age: f64) -> Result<(), BraheError> {
        if max_age <= 0.0 {
            return Err(BraheError::Error(
                "Maximum age must be > 0.0".to_string(),
            ));
        }
        self.eviction_policy = TrajectoryEvictionPolicy::KeepWithinDuration;
        self.max_age = Some(max_age);
        self.max_size = None;
        self.apply_eviction_policy()?;
        Ok(())
    }

    /// Add a state to the trajectory
    pub fn add_state(&mut self, epoch: Epoch, state: SVector<f64, R>) -> Result<(), BraheError> {
        // Find the correct position to insert based on epoch
        let mut insert_idx = self.epochs.len();
        for (i, existing_epoch) in self.epochs.iter().enumerate() {
            if epoch < *existing_epoch {
                insert_idx = i;
                break;
            } else if epoch == *existing_epoch {
                // Replace state if epochs are equal
                self.states[i] = state;
                self.apply_eviction_policy()?;
                return Ok(());
            }
        }

        // Insert at the correct position
        self.epochs.insert(insert_idx, epoch);
        self.states.insert(insert_idx, state);

        // Apply eviction policy after adding state
        self.apply_eviction_policy()?;
        Ok(())
    }

    /// Apply eviction policy to manage trajectory memory
    fn apply_eviction_policy(&mut self) -> Result<(), BraheError> {
        match self.eviction_policy {
            TrajectoryEvictionPolicy::None => {
                // No eviction
            },
            TrajectoryEvictionPolicy::KeepCount => {
                if let Some(max_size) = self.max_size {
                    if self.epochs.len() > max_size {
                        let to_remove = self.epochs.len() - max_size;
                        self.epochs.drain(0..to_remove);
                        self.states.drain(0..to_remove);
                    }
                }
            },
            TrajectoryEvictionPolicy::KeepWithinDuration => {
                if let Some(max_age) = self.max_age {
                    if let Some(&last_epoch) = self.epochs.last() {
                        let mut indices_to_keep = Vec::new();
                        for (i, &epoch) in self.epochs.iter().enumerate() {
                            if (last_epoch - epoch).abs() <= max_age {
                                indices_to_keep.push(i);
                            }
                        }

                        let new_epochs: Vec<Epoch> = indices_to_keep.iter().map(|&i| self.epochs[i]).collect();
                        let new_states: Vec<SVector<f64, R>> = indices_to_keep.iter().map(|&i| self.states[i].clone()).collect();

                        self.epochs = new_epochs;
                        self.states = new_states;
                    }
                }
            },
        }
        Ok(())
    }

    /// Get the state at a specific epoch using interpolation
    pub fn state_at_epoch(&self, epoch: &Epoch) -> Result<SVector<f64, R>, BraheError> {
        if self.epochs.is_empty() {
            return Err(BraheError::Error(
                "Cannot interpolate state from empty trajectory".to_string(),
            ));
        }

        // If only one state, return it
        if self.epochs.len() == 1 {
            return Ok(self.states[0].clone());
        }

        // If epoch is before the first state or after the last state
        if epoch < &self.epochs[0] {
            return Err(BraheError::Error(
                "Requested epoch is before the first state in trajectory".to_string(),
            ));
        }
        if epoch > self.epochs.last().unwrap() {
            return Err(BraheError::Error(
                "Requested epoch is after the last state in trajectory".to_string(),
            ));
        }

        // Find the exact state if it exists
        for (i, existing_epoch) in self.epochs.iter().enumerate() {
            if epoch == existing_epoch {
                return Ok(self.states[i].clone());
            }
        }

        // Interpolate based on method
        match self.interpolation_method {
            InterpolationMethod::Linear => self.interpolate_linear(epoch),
            InterpolationMethod::CubicSpline => Err(BraheError::Error(
                "Cubic spline interpolation not yet implemented".to_string(),
            )),
            InterpolationMethod::Lagrange => Err(BraheError::Error(
                "Lagrange interpolation not yet implemented".to_string(),
            )),
            InterpolationMethod::Hermite => Err(BraheError::Error(
                "Hermite interpolation not yet implemented".to_string(),
            )),
        }
    }

    /// Find the nearest state to the specified epoch
    /// Returns (epoch, state) tuple
    pub fn nearest_state(&self, epoch: &Epoch) -> Result<(Epoch, SVector<f64, R>), BraheError> {
        if self.epochs.is_empty() {
            return Err(BraheError::Error(
                "Cannot find nearest state in empty trajectory".to_string(),
            ));
        }

        let mut nearest_idx = 0;
        let mut min_diff = f64::MAX;

        for (i, existing_epoch) in self.epochs.iter().enumerate() {
            let diff = (*epoch - *existing_epoch).abs();
            if diff < min_diff {
                min_diff = diff;
                nearest_idx = i;
            }

            // Optimization: if we're past the epoch and moving away, we can stop
            if i > 0 && existing_epoch > epoch && diff > min_diff {
                break;
            }
        }

        Ok((self.epochs[nearest_idx], self.states[nearest_idx].clone()))
    }

    /// Interpolate between states using linear interpolation
    fn interpolate_linear(&self, epoch: &Epoch) -> Result<SVector<f64, R>, BraheError> {
        if self.epochs.is_empty() {
            return Err(BraheError::Error(
                "Cannot interpolate state from empty trajectory".to_string(),
            ));
        }

        // If only one state, return it
        if self.epochs.len() == 1 {
            return Ok(self.states[0].clone());
        }

        // Find the two states that bracket the requested epoch
        for i in 0..self.epochs.len() - 1 {
            let epoch1 = self.epochs[i];
            let epoch2 = self.epochs[i + 1];

            // Check if the requested epoch is between these two states
            if epoch >= &epoch1 && epoch <= &epoch2 {
                let state1 = &self.states[i];
                let state2 = &self.states[i + 1];

                // Calculate interpolation factor (t)
                let t1 = epoch1;
                let t2 = epoch2;
                let t = *epoch;

                // This computes the normalized interpolation factor (0 to 1)
                let alpha = (t - t1) / (t2 - t1);

                // Linear interpolation for each element
                let mut interpolated_state = SVector::<f64, R>::zeros();
                for j in 0..interpolated_state.len() {
                    interpolated_state[j] = state1[j] * (1.0 - alpha) + state2[j] * alpha;
                }

                return Ok(interpolated_state);
            }
        }

        // If we reach here, something went wrong with our epoch comparison logic
        Err(BraheError::Error(
            "Failed to find bracketing states for interpolation".to_string(),
        ))
    }

    /// Get the state at the specified index
    pub fn state_at_index(&self, index: usize) -> Result<SVector<f64, R>, BraheError> {
        if index >= self.states.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            )));
        }

        Ok(self.states[index].clone())
    }

    /// Get the epoch at the specified index
    pub fn epoch_at_index(&self, index: usize) -> Result<Epoch, BraheError> {
        if index >= self.epochs.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} epochs",
                index,
                self.epochs.len()
            )));
        }

        Ok(self.epochs[index])
    }

    /// Remove a state at the specified epoch
    /// Returns the removed state if found
    pub fn remove_state(&mut self, epoch: &Epoch) -> Result<SVector<f64, R>, BraheError> {
        if let Some(index) = self.epochs.iter().position(|e| e == epoch) {
            let removed_state = self.states.remove(index);
            self.epochs.remove(index);
            Ok(removed_state)
        } else {
            Err(BraheError::Error(
                "Epoch not found in trajectory".to_string(),
            ))
        }
    }

    /// Remove a state at the specified index
    /// Returns (epoch, state) tuple of the removed entry
    pub fn remove_state_at_index(&mut self, index: usize) -> Result<(Epoch, SVector<f64, R>), BraheError> {
        if index >= self.states.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            )));
        }

        let removed_epoch = self.epochs.remove(index);
        let removed_state = self.states.remove(index);
        Ok((removed_epoch, removed_state))
    }

    /// Returns the number of states in the trajectory
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Returns true if the trajectory is empty
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Get the first epoch in the trajectory
    pub fn start_epoch(&self) -> Option<Epoch> {
        self.epochs.first().copied()
    }

    /// Get the last epoch in the trajectory
    pub fn end_epoch(&self) -> Option<Epoch> {
        self.epochs.last().copied()
    }

    /// Get the time span covered by the trajectory
    pub fn timespan(&self) -> Option<f64> {
        if self.epochs.len() < 2 {
            None
        } else {
            Some(*self.epochs.last().unwrap() - *self.epochs.first().unwrap())
        }
    }

    /// Get both epoch and state at the specified index as a tuple
    pub fn get(&self, index: usize) -> Result<(Epoch, SVector<f64, R>), BraheError> {
        if index >= self.states.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            )));
        }

        Ok((self.epochs[index], self.states[index].clone()))
    }

    /// Convert the trajectory to a matrix representation
    /// Returns a matrix where columns are time points and rows are state elements
    /// The matrix has shape (6, n_epochs) for a 6-element state vector
    pub fn to_matrix(&self) -> Result<nalgebra::DMatrix<f64>, BraheError> {
        if self.states.is_empty() {
            return Err(BraheError::Error(
                "Cannot convert empty trajectory to matrix".to_string(),
            ));
        }

        let n_epochs = self.states.len();
        let n_elements = 6;

        let mut matrix = nalgebra::DMatrix::<f64>::zeros(n_elements, n_epochs);

        for (col_idx, state) in self.states.iter().enumerate() {
            for row_idx in 0..n_elements {
                matrix[(row_idx, col_idx)] = state[row_idx];
            }
        }

        Ok(matrix)
    }

    /// Clear all states from the trajectory
    pub fn clear(&mut self) {
        self.epochs.clear();
        self.states.clear();
    }

    /// Get the first (epoch, state) tuple in the trajectory, if any exists.
    ///
    /// # Returns
    /// * `Some((epoch, state))` - If the trajectory contains at least one state
    /// * `None` - If the trajectory is empty
    ///
    /// # Examples
    /// ```rust
    /// use brahe::trajectories::Trajectory6;
    /// use brahe::time::{Epoch, TimeSystem};
    /// use nalgebra::Vector6;
    ///
    /// let mut traj = Trajectory6::new();
    /// assert!(traj.first().is_none());
    ///
    /// let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state = Vector6::new(6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0);
    /// traj.add_state(epoch, state).unwrap();
    ///
    /// let (first_epoch, first_state) = traj.first().unwrap();
    /// assert_eq!(first_epoch, epoch);
    /// assert_eq!(first_state, state);
    /// ```
    pub fn first(&self) -> Option<(Epoch, SVector<f64, R>)> {
        if self.epochs.is_empty() {
            None
        } else {
            Some((self.epochs[0], self.states[0].clone()))
        }
    }

    /// Get the last (epoch, state) tuple in the trajectory, if any exists.
    ///
    /// # Returns
    /// * `Some((epoch, state))` - If the trajectory contains at least one state
    /// * `None` - If the trajectory is empty
    ///
    /// # Examples
    /// ```rust
    /// use brahe::trajectories::Trajectory6;
    /// use brahe::time::{Epoch, TimeSystem};
    /// use nalgebra::Vector6;
    ///
    /// let mut traj = Trajectory6::new();
    /// assert!(traj.last().is_none());
    ///
    /// let epoch1 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state1 = Vector6::new(6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0);
    /// traj.add_state(epoch1, state1).unwrap();
    ///
    /// let epoch2 = Epoch::from_datetime(2023, 1, 1, 13, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let state2 = Vector6::new(6.778e6, 0.0, 0.0, 0.0, 7.626e3, 0.0);
    /// traj.add_state(epoch2, state2).unwrap();
    ///
    /// let (last_epoch, last_state) = traj.last().unwrap();
    /// assert_eq!(last_epoch, epoch2);
    /// assert_eq!(last_state, state2);
    /// ```
    pub fn last(&self) -> Option<(Epoch, SVector<f64, R>)> {
        if self.epochs.is_empty() {
            None
        } else {
            let last_index = self.epochs.len() - 1;
            Some((self.epochs[last_index], self.states[last_index].clone()))
        }
    }
}

// Allow indexing into the trajectory directly - returns state vector only
// For (epoch, state) tuples, use the get() method instead
impl<const R: usize> Index<usize> for Trajectory<R>
{
    type Output = SVector<f64, R>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.states[index]
    }
}

// Iterator implementation will be added later once trait bounds are resolved

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{Epoch, TimeSystem};
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector6;

    fn create_test_trajectory() -> Trajectory6 {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
            Epoch::from_jd(2451545.2, TimeSystem::UTC),
        ];

        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
            Vector6::new(7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0),
        ];

        Trajectory6::from_data(epochs, states, InterpolationMethod::Linear).unwrap()
    }

    #[test]
    fn test_trajectory_creation() {
        let trajectory = create_test_trajectory();

        assert_eq!(trajectory.len(), 3);
        assert_eq!(trajectory.interpolation_method, InterpolationMethod::Linear);
        assert!(!trajectory.is_empty());
    }

    #[test]
    fn test_trajectory_add_state() {
        let mut trajectory = Trajectory6::new();

        // Add states in order
        let epoch1 = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state1 = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        trajectory.add_state(epoch1, state1).unwrap();

        let epoch3 = Epoch::from_jd(2451545.2, TimeSystem::UTC);
        let state3 = Vector6::new(7200e3, 0.0, 0.0, 0.0, 7.7e3, 0.0);
        trajectory.add_state(epoch3, state3).unwrap();

        // Add a state in between
        let epoch2 = Epoch::from_jd(2451545.1, TimeSystem::UTC);
        let state2 = Vector6::new(7100e3, 0.0, 0.0, 0.0, 7.6e3, 0.0);
        trajectory.add_state(epoch2, state2).unwrap();

        assert_eq!(trajectory.len(), 3);
        assert_eq!(trajectory.epochs[0].jd(), 2451545.0);
        assert_eq!(trajectory.epochs[1].jd(), 2451545.1);
        assert_eq!(trajectory.epochs[2].jd(), 2451545.2);
    }

    #[test]
    fn test_trajectory_indexing() {
        let trajectory = create_test_trajectory();

        assert_eq!(trajectory[0][0], 7000e3);
        assert_eq!(trajectory[1][0], 7100e3);
        assert_eq!(trajectory[2][0], 7200e3);
    }

    #[test]
    fn test_trajectory_nearest_state() {
        let trajectory = create_test_trajectory();

        // Request a time exactly at a state
        let (epoch, state) = trajectory
            .nearest_state(&Epoch::from_jd(2451545.0, TimeSystem::UTC))
            .unwrap();
        assert_eq!(epoch.jd(), 2451545.0);
        assert_eq!(state[0], 7000e3);

        // Request a time halfway between two states
        let (epoch, state) = trajectory
            .nearest_state(&Epoch::from_jd(2451545.05, TimeSystem::UTC))
            .unwrap();

        // Should return the closest state (first one)
        assert_eq!(epoch.jd(), 2451545.0);
        assert_eq!(state[0], 7000e3);

        // Request a time after the last state
        let (epoch, state) = trajectory
            .nearest_state(&Epoch::from_jd(2451545.3, TimeSystem::UTC))
            .unwrap();
        assert_eq!(epoch.jd(), 2451545.2);
        assert_eq!(state[0], 7200e3);

        // Request a time closer to the second state
        let (epoch, state) = trajectory
            .nearest_state(&Epoch::from_jd(2451545.09, TimeSystem::UTC))
            .unwrap();
        assert_eq!(epoch.jd(), 2451545.1);
        assert_eq!(state[0], 7100e3);
    }

    #[test]
    fn test_trajectory_linear_interpolation() {
        let trajectory = create_test_trajectory();

        // Test interpolation at 50% between first two states
        let epoch_50 = Epoch::from_jd(2451545.05, TimeSystem::UTC);
        let state_at_50 = trajectory.state_at_epoch(&epoch_50).unwrap();

        // The interpolated values should be 50% between the first two states
        assert_abs_diff_eq!(state_at_50[0], 7050e3, epsilon = 1.0); // 7000 + 0.5*(7100-7000)
        assert_abs_diff_eq!(state_at_50[1], 500e3, epsilon = 1.0); // 0 + 0.5*(1000-0)
        assert_abs_diff_eq!(state_at_50[4], 7.55e3, epsilon = 0.01); // 7.5 + 0.5*(7.6-7.5)
    }

    #[test]
    fn test_trajectory_to_matrix() {
        let trajectory = create_test_trajectory();

        let matrix = trajectory.to_matrix().unwrap();
        assert_eq!(matrix.nrows(), 6);
        assert_eq!(matrix.ncols(), 3);

        // Check first column
        assert_eq!(matrix[(0, 0)], 7000e3);
        assert_eq!(matrix[(1, 0)], 0.0);
        assert_eq!(matrix[(2, 0)], 0.0);
        assert_eq!(matrix[(3, 0)], 0.0);
        assert_eq!(matrix[(4, 0)], 7.5e3);
        assert_eq!(matrix[(5, 0)], 0.0);
        assert_eq!(matrix[(0, 1)], 7100e3);

        // Check second column
        assert_eq!(matrix[(0, 1)], 7100e3);
        assert_eq!(matrix[(1, 1)], 1000e3);
        assert_eq!(matrix[(2, 1)], 500e3);
        assert_eq!(matrix[(3, 1)], 100.0);
        assert_eq!(matrix[(4, 1)], 7.6e3);
        assert_eq!(matrix[(5, 1)], 50.0);

        // Check third column
        assert_eq!(matrix[(0, 2)], 7200e3);
        assert_eq!(matrix[(1, 2)], 2000e3);
        assert_eq!(matrix[(2, 2)], 1000e3);
        assert_eq!(matrix[(3, 2)], 200.0);
        assert_eq!(matrix[(4, 2)], 7.7e3);
        assert_eq!(matrix[(5, 2)], 100.0);
    }

    #[test]
    fn test_trajectory_eviction_policy_max_size() {
        let mut trajectory = Trajectory6::new();
        trajectory.set_eviction_policy_max_size(2).unwrap();

        // Add three states
        trajectory.add_state(
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Vector6::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        ).unwrap();

        trajectory.add_state(
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
            Vector6::new(2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        ).unwrap();

        trajectory.add_state(
            Epoch::from_jd(2451545.2, TimeSystem::UTC),
            Vector6::new(3.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        ).unwrap();

        // Should only keep the last 2 states
        assert_eq!(trajectory.len(), 2);
        assert_eq!(trajectory.states[0][0], 2.0); // Second state
        assert_eq!(trajectory.states[1][0], 3.0); // Third state
    }

    #[test]
    fn test_trajectory_eviction_policy_max_age() {
        let mut trajectory = Trajectory6::new();
        trajectory.set_eviction_policy_max_age(86400.0).unwrap(); // 1 day in seconds
        let base_epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);

        // Add states at 0, 0.5, 0.99
        trajectory.add_state(
            base_epoch,
            Vector6::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        ).unwrap(); // Day 0

        trajectory.add_state(
            base_epoch + 0.5 * 86400.0,
            Vector6::new(2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        ).unwrap(); // Day 0.5

        trajectory.add_state(
            base_epoch + 0.99 * 86400.0,
            Vector6::new(3.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        ).unwrap(); // Day 0.99

        // Confirm all three states are present
        assert_eq!(trajectory.len(), 3);
        assert_eq!(trajectory.states[0][0], 1.0);
        assert_eq!(trajectory.states[1][0], 2.0);
        assert_eq!(trajectory.states[2][0], 3.0);

        // Add a state at day 2.0, which should evict the first three states
        trajectory.add_state(
            base_epoch + 2.0 * 86400.0,
            Vector6::new(4.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        ).unwrap(); // Day 2.0

        // Should only keep the last state
        assert_eq!(trajectory.len(), 1);
        assert_eq!(trajectory.states[0][0], 4.0);
    }

    #[test]
    fn test_trajectory_new_eviction_methods() {
        let mut trajectory = Trajectory6::new();

        // Test validation for max_size
        assert!(trajectory.set_eviction_policy_max_size(0).is_err());
        assert!(trajectory.set_eviction_policy_max_size(1).is_ok());

        // Test validation for max_age
        assert!(trajectory.set_eviction_policy_max_age(0.0).is_err());
        assert!(trajectory.set_eviction_policy_max_age(-1.0).is_err());
        assert!(trajectory.set_eviction_policy_max_age(60.0).is_ok());
    }

    #[test]
    fn test_trajectory_remove_state_at_index() {
        let mut trajectory = create_test_trajectory();

        // Test remove_state_at_index
        let (removed_epoch, removed_state) = trajectory.remove_state_at_index(1).unwrap();
        assert_eq!(removed_epoch.jd(), 2451545.1);
        assert_eq!(removed_state[0], 7100e3);
        assert_eq!(trajectory.len(), 2);

        // Test error cases
        assert!(trajectory.remove_state_at_index(10).is_err());
        let non_existent_epoch = Epoch::from_jd(2451546.0, TimeSystem::UTC);
        assert!(trajectory.remove_state(&non_existent_epoch).is_err());
    }

    #[test]
    fn test_trajectory_remove_state() {
        let mut trajectory = create_test_trajectory();

        // Test remove_state
        let epoch_to_remove = Epoch::from_jd(2451545.1, TimeSystem::UTC);
        let removed_state = trajectory.remove_state(&epoch_to_remove).unwrap();
        assert_eq!(removed_state[0], 7100e3);
        assert_eq!(trajectory.len(), 2);

        // Test error case
        let non_existent_epoch = Epoch::from_jd(2451546.0, TimeSystem::UTC);
        assert!(trajectory.remove_state(&non_existent_epoch).is_err());
    }

    #[test]
    fn test_trajectory_get_method() {
        let trajectory = create_test_trajectory();

        // Test get method
        let (epoch, state) = trajectory.get(0).unwrap();
        assert_eq!(epoch.jd(), 2451545.0);
        assert_eq!(state[0], 7000e3);

        // Test bounds checking
        assert!(trajectory.get(10).is_err());
    }

    #[test]
    fn test_trajectory_iterator() {
        let trajectory = create_test_trajectory();

        // Test manual iteration using get method
        for i in 0..trajectory.len() {
            let (epoch, state) = trajectory.get(i).unwrap();
            match i {
                0 => {
                    assert_eq!(epoch.jd(), 2451545.0);
                    assert_eq!(state[0], 7000e3);
                }
                1 => {
                    assert_eq!(epoch.jd(), 2451545.1);
                    assert_eq!(state[0], 7100e3);
                }
                2 => {
                    assert_eq!(epoch.jd(), 2451545.2);
                    assert_eq!(state[0], 7200e3);
                }
                _ => panic!("Unexpected iteration"),
            }
        }
        assert_eq!(trajectory.len(), 3);
    }

    #[test]
    fn test_trajectory_timespan_rename() {
        let trajectory = create_test_trajectory();

        // Test renamed method
        let span = trajectory.timespan().unwrap();
        assert_abs_diff_eq!(span, 0.2 * 86400.0, epsilon = 1.0); // 0.2 days in seconds
    }

    #[test]
    fn test_trajectory_set_interpolation_method() {
        let mut trajectory = Trajectory6::new();
        assert_eq!(trajectory.interpolation_method, InterpolationMethod::Linear);

        trajectory.set_interpolation_method(InterpolationMethod::CubicSpline);
        assert_eq!(trajectory.interpolation_method, InterpolationMethod::CubicSpline);

        trajectory.set_interpolation_method(InterpolationMethod::Lagrange);
        assert_eq!(trajectory.interpolation_method, InterpolationMethod::Lagrange);
    }

    #[test]
    fn test_trajectory_state_at_index() {
        let trajectory = create_test_trajectory();

        // Test valid indices
        let state0 = trajectory.state_at_index(0).unwrap();
        assert_eq!(state0[0], 7000e3);

        let state1 = trajectory.state_at_index(1).unwrap();
        assert_eq!(state1[0], 7100e3);

        let state2 = trajectory.state_at_index(2).unwrap();
        assert_eq!(state2[0], 7200e3);

        // Test invalid index
        assert!(trajectory.state_at_index(10).is_err());
    }

    #[test]
    fn test_trajectory_epoch_at_index() {
        let trajectory = create_test_trajectory();

        // Test valid indices
        let epoch0 = trajectory.epoch_at_index(0).unwrap();
        assert_eq!(epoch0.jd(), 2451545.0);

        let epoch1 = trajectory.epoch_at_index(1).unwrap();
        assert_eq!(epoch1.jd(), 2451545.1);

        let epoch2 = trajectory.epoch_at_index(2).unwrap();
        assert_eq!(epoch2.jd(), 2451545.2);

        // Test invalid index
        assert!(trajectory.epoch_at_index(10).is_err());
    }

    #[test]
    fn test_trajectory_start_and_end_epoch() {
        let trajectory = create_test_trajectory();

        // Test start_epoch
        let start = trajectory.start_epoch().unwrap();
        assert_eq!(start.jd(), 2451545.0);

        // Test end_epoch
        let end = trajectory.end_epoch().unwrap();
        assert_eq!(end.jd(), 2451545.2);

        // Test empty trajectory
        let empty_trajectory = Trajectory6::new();
        assert!(empty_trajectory.start_epoch().is_none());
        assert!(empty_trajectory.end_epoch().is_none());
    }

    #[test]
    fn test_trajectory_clear() {
        let mut trajectory = create_test_trajectory();
        assert_eq!(trajectory.len(), 3);
        assert!(!trajectory.is_empty());

        trajectory.clear();
        assert_eq!(trajectory.len(), 0);
        assert!(trajectory.is_empty());
        assert!(trajectory.start_epoch().is_none());
        assert!(trajectory.end_epoch().is_none());
    }

    #[test]
    fn test_trajectory_first_and_last() {
        // Test empty trajectory
        let empty_trajectory = Trajectory6::new();
        assert!(empty_trajectory.first().is_none());
        assert!(empty_trajectory.last().is_none());

        // Test single state trajectory
        let mut single_trajectory = Trajectory6::new();
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        single_trajectory.add_state(epoch, state).unwrap();

        let (first_epoch, first_state) = single_trajectory.first().unwrap();
        assert_eq!(first_epoch.jd(), 2451545.0);
        assert_eq!(first_state[0], 7000e3);

        let (last_epoch, last_state) = single_trajectory.last().unwrap();
        assert_eq!(last_epoch.jd(), 2451545.0);
        assert_eq!(last_state[0], 7000e3);

        // Test multi-state trajectory
        let trajectory = create_test_trajectory();

        let (first_epoch, first_state) = trajectory.first().unwrap();
        assert_eq!(first_epoch.jd(), 2451545.0);
        assert_eq!(first_state[0], 7000e3);

        let (last_epoch, last_state) = trajectory.last().unwrap();
        assert_eq!(last_epoch.jd(), 2451545.2);
        assert_eq!(last_state[0], 7200e3);
    }

    #[test]
    fn test_trajectory_from_data_errors() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
        ];

        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
        ];

        // Test mismatched lengths
        let result = Trajectory6::from_data(epochs, states, InterpolationMethod::Linear);
        assert!(result.is_err());

        // Test empty data
        let empty_epochs = vec![];
        let empty_states = vec![];
        let result = Trajectory6::from_data(empty_epochs, empty_states, InterpolationMethod::Linear);
        assert!(result.is_err());
    }

    #[test]
    fn test_trajectory_state_at_epoch_errors() {
        let trajectory = create_test_trajectory();

        // Test epoch before first state
        let early_epoch = Epoch::from_jd(2451544.5, TimeSystem::UTC);
        assert!(trajectory.state_at_epoch(&early_epoch).is_err());

        // Test epoch after last state
        let late_epoch = Epoch::from_jd(2451545.5, TimeSystem::UTC);
        assert!(trajectory.state_at_epoch(&late_epoch).is_err());

        // Test empty trajectory
        let empty_trajectory = Trajectory6::new();
        let any_epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        assert!(empty_trajectory.state_at_epoch(&any_epoch).is_err());
    }

    #[test]
    fn test_trajectory_unimplemented_interpolation_methods() {
        let mut trajectory = create_test_trajectory();
        let mid_epoch = Epoch::from_jd(2451545.05, TimeSystem::UTC);

        // Test CubicSpline (not implemented)
        trajectory.set_interpolation_method(InterpolationMethod::CubicSpline);
        assert!(trajectory.state_at_epoch(&mid_epoch).is_err());

        // Test Lagrange (not implemented)
        trajectory.set_interpolation_method(InterpolationMethod::Lagrange);
        assert!(trajectory.state_at_epoch(&mid_epoch).is_err());

        // Test Hermite (not implemented)
        trajectory.set_interpolation_method(InterpolationMethod::Hermite);
        assert!(trajectory.state_at_epoch(&mid_epoch).is_err());
    }

    #[test]
    fn test_trajectory_timespan_edge_cases() {
        // Test single state trajectory
        let mut single_trajectory = Trajectory6::new();
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        single_trajectory.add_state(epoch, state).unwrap();
        assert!(single_trajectory.timespan().is_none());

        // Test empty trajectory
        let empty_trajectory = Trajectory6::new();
        assert!(empty_trajectory.timespan().is_none());
    }
}