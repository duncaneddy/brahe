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
 * use brahe::trajectories::{Trajectory, InterpolationMethod};
 * use brahe::time::{Epoch, TimeSystem};
 * use nalgebra::Vector6;
 *
 * let mut traj = Trajectory::new();
 * let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
 * let state = Vector6::new(6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0);
 * traj.add_state(epoch, state).unwrap();
 * ```
 */

use nalgebra::Vector6;
use serde::{Deserialize, Serialize};
use std::ops::Index;

use crate::time::Epoch;
use crate::utils::BraheError;

/// Interpolation methods for retrieving trajectory states at arbitrary epochs.
///
/// Different methods provide varying trade-offs between computational cost and accuracy.
/// For most applications, linear interpolation provides sufficient accuracy with minimal
/// computational overhead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterpolationMethod {
    /// No interpolation - returns the nearest state by time.
    /// Fastest method but can introduce discontinuities.
    None,
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

/// Frame-agnostic trajectory container for 6-dimensional state vectors over time.
///
/// The trajectory maintains a chronologically sorted collection of epochs and corresponding
/// state vectors. State vectors are typically either Cartesian position/velocity (x, y, z, vx, vy, vz)
/// or orbital elements (a, e, i, Ω, ω, M), but the interpretation is left to higher-level containers
/// like `OrbitalTrajectory`.
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trajectory {
    /// Time epochs for each state, maintained in chronological order.
    /// All epochs should use consistent time systems for meaningful interpolation.
    pub epochs: Vec<Epoch>,

    /// 6-dimensional state vectors corresponding to epochs.
    /// Units and interpretation depend on the specific use case:
    /// - Cartesian: [m, m, m, m/s, m/s, m/s]
    /// - Keplerian: [m, dimensionless, rad, rad, rad, rad]
    pub states: Vec<Vector6<f64>>,

    /// Interpolation method for state retrieval at arbitrary epochs.
    /// Default is linear interpolation for optimal performance/accuracy balance.
    pub interpolation_method: InterpolationMethod,

    /// Maximum number of states to retain (None = unlimited).
    /// When exceeded, oldest states are evicted according to the eviction policy.
    pub max_size: Option<usize>,

    /// Maximum age of states to retain in seconds (None = unlimited).
    /// States older than this duration from the latest epoch are evicted.
    pub max_age: Option<f64>,

    /// Memory management policy for automatic state eviction.
    /// Controls how states are removed when limits are exceeded.
    pub eviction_policy: TrajectoryEvictionPolicy,
}

impl Default for Trajectory {
    /// Creates a trajectory with default settings (linear interpolation, no memory limits).
    fn default() -> Self {
        Self::new()
    }
}

impl Trajectory {
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
    /// use brahe::trajectories::Trajectory;
    /// let traj = Trajectory::new();
    /// assert_eq!(traj.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            epochs: Vec::new(),
            states: Vec::new(),
            interpolation_method: InterpolationMethod::Linear,
            max_size: None,
            max_age: None,
            eviction_policy: TrajectoryEvictionPolicy::None,
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
    /// use brahe::trajectories::{Trajectory, InterpolationMethod};
    /// let traj = Trajectory::with_interpolation(InterpolationMethod::CubicSpline);
    /// assert_eq!(traj.len(), 0);
    /// ```
    pub fn with_interpolation(interpolation_method: InterpolationMethod) -> Self {
        Self {
            epochs: Vec::new(),
            states: Vec::new(),
            interpolation_method,
            max_size: None,
            max_age: None,
            eviction_policy: TrajectoryEvictionPolicy::None,
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
    /// use brahe::trajectories::{Trajectory, InterpolationMethod};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use nalgebra::Vector6;
    ///
    /// let epochs = vec![Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC)];
    /// let states = vec![Vector6::new(6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0)];
    /// let traj = Trajectory::from_data(epochs, states, InterpolationMethod::Linear).unwrap();
    /// ```
    pub fn from_data(
        epochs: Vec<Epoch>,
        states: Vec<Vector6<f64>>,
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
        let sorted_states: Vec<Vector6<f64>> = indices.iter().map(|&i| states[i]).collect();

        Ok(Self {
            epochs: sorted_epochs,
            states: sorted_states,
            interpolation_method,
            max_size: None,
            max_age: None,
            eviction_policy: TrajectoryEvictionPolicy::None,
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
    /// use brahe::trajectories::{Trajectory, InterpolationMethod};
    /// let mut traj = Trajectory::new(); // defaults to Linear
    /// traj.set_interpolation_method(InterpolationMethod::CubicSpline);
    /// ```
    pub fn set_interpolation_method(&mut self, method: InterpolationMethod) {
        self.interpolation_method = method;
    }

    /// Set maximum trajectory size for memory management
    pub fn set_max_size(&mut self, max_size: Option<usize>) {
        self.max_size = max_size;
    }

    /// Set maximum age of states to keep (in seconds) for time-based eviction
    pub fn set_max_age(&mut self, max_age: Option<f64>) {
        self.max_age = max_age;
    }

    /// Set eviction policy for trajectory memory management
    pub fn set_eviction_policy(&mut self, policy: TrajectoryEvictionPolicy) {
        self.eviction_policy = policy;
    }

    /// Add a state to the trajectory
    pub fn add_state(&mut self, epoch: Epoch, state: Vector6<f64>) -> Result<(), BraheError> {
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
                        let new_states: Vec<Vector6<f64>> = indices_to_keep.iter().map(|&i| self.states[i]).collect();

                        self.epochs = new_epochs;
                        self.states = new_states;
                    }
                }
            },
        }
        Ok(())
    }

    /// Get the state at a specific epoch using interpolation
    pub fn state_at_epoch(&self, epoch: &Epoch) -> Result<Vector6<f64>, BraheError> {
        if self.epochs.is_empty() {
            return Err(BraheError::Error(
                "Cannot interpolate state from empty trajectory".to_string(),
            ));
        }

        // If only one state, return it
        if self.epochs.len() == 1 {
            return Ok(self.states[0]);
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
                return Ok(self.states[i]);
            }
        }

        // Interpolate based on method
        match self.interpolation_method {
            InterpolationMethod::None => {
                let (_, state) = self.nearest_state(epoch)?;
                Ok(state)
            },
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
    pub fn nearest_state(&self, epoch: &Epoch) -> Result<(Epoch, Vector6<f64>), BraheError> {
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

        Ok((self.epochs[nearest_idx], self.states[nearest_idx]))
    }

    /// Interpolate between states using linear interpolation
    fn interpolate_linear(&self, epoch: &Epoch) -> Result<Vector6<f64>, BraheError> {
        if self.epochs.is_empty() {
            return Err(BraheError::Error(
                "Cannot interpolate state from empty trajectory".to_string(),
            ));
        }

        // If only one state, return it
        if self.epochs.len() == 1 {
            return Ok(self.states[0]);
        }

        // Find the two states that bracket the requested epoch
        for i in 0..self.epochs.len() - 1 {
            let epoch1 = self.epochs[i];
            let epoch2 = self.epochs[i + 1];

            // Check if the requested epoch is between these two states
            if epoch >= &epoch1 && epoch <= &epoch2 {
                let state1 = self.states[i];
                let state2 = self.states[i + 1];

                // Calculate interpolation factor (t)
                let t1 = epoch1;
                let t2 = epoch2;
                let t = *epoch;

                // This computes the normalized interpolation factor (0 to 1)
                let alpha = (t - t1) / (t2 - t1);

                // Linear interpolation for each element
                let mut interpolated_state = Vector6::zeros();
                for j in 0..6 {
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
    pub fn state_at_index(&self, index: usize) -> Result<Vector6<f64>, BraheError> {
        if index >= self.states.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            )));
        }

        Ok(self.states[index])
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
    pub fn time_span(&self) -> Option<f64> {
        if self.epochs.len() < 2 {
            None
        } else {
            Some(*self.epochs.last().unwrap() - *self.epochs.first().unwrap())
        }
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
}

// Allow indexing into the trajectory directly
impl Index<usize> for Trajectory {
    type Output = Vector6<f64>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.states[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{Epoch, TimeSystem};
    use approx::assert_abs_diff_eq;

    fn create_test_trajectory() -> Trajectory {
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

        Trajectory::from_data(epochs, states, InterpolationMethod::Linear).unwrap()
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
        let mut trajectory = Trajectory::new();

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
        assert_eq!(matrix[(4, 0)], 7.5e3);

        // Check second column
        assert_eq!(matrix[(0, 1)], 7100e3);
        assert_eq!(matrix[(1, 1)], 1000e3);
    }

    #[test]
    fn test_trajectory_eviction_policy() {
        let mut trajectory = Trajectory::new();
        trajectory.set_max_size(Some(2));
        trajectory.set_eviction_policy(TrajectoryEvictionPolicy::KeepCount);

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
}