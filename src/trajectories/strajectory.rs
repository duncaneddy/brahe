/*!
 * Static trajectory implementation for storing and interpolating N-dimensional state vectors over time.
 *
 * This module provides a compile-time sized, frame-agnostic trajectory container that stores epochs and
 * corresponding N-dimensional state vectors. The trajectory supports various interpolation methods, memory
 * management policies, and efficient access patterns for high-performance applications.
 *
 * # Key Features
 * - Compile-time sized vectors for maximum performance
 * - Frame-agnostic storage (no assumptions about coordinate frames)
 * - Multiple interpolation methods (linear, cubic spline, Lagrange, etc.)
 * - Memory management with configurable eviction policies
 * - Efficient nearest-state and exact-epoch lookups
 * - Serialization support for persistence
 *
 * # Examples
 * ```rust
 * use brahe::trajectories::{STrajectory6, InterpolationMethod, Trajectory};
 * use brahe::time::{Epoch, TimeSystem};
 * use nalgebra::Vector6;
 *
 * let mut traj = STrajectory6::new();
 * let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
 * let state = Vector6::new(6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0);
 * traj.add(epoch, state).unwrap();
 * ```
 */

use nalgebra::SVector;
use serde_json::Value;
use std::collections::HashMap;
use std::ops::Index;

use crate::time::Epoch;
use crate::utils::BraheError;

use super::traits::{Trajectory, Interpolatable, InterpolationMethod, TrajectoryEvictionPolicy};

/// Type alias for a 3-dimensional static trajectory (e.g., position only)
pub type STrajectory3 = STrajectory<3>;

/// Type alias for a 4-dimensional static trajectory (e.g., quaternion)
pub type STrajectory4 = STrajectory<4>;

/// Type alias for a 6-dimensional static trajectory (commonly used for orbital mechanics)
pub type STrajectory6 = STrajectory<6>;

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
pub struct STrajectory<const R: usize>
{
    /// Time epochs for each state, maintained in chronological order.
    /// All epochs should use consistent time systems for meaningful interpolation.
    pub epochs: Vec<Epoch>,

    /// R-dimensional state vectors corresponding to epochs.
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

    /// Generic metadata storage supporting arbitrary key-value pairs.
    /// Can store any JSON-serializable data including strings, numbers, booleans,
    /// arrays, and nested objects. For orbital trajectories, use ORBITAL_*_KEY constants.
    pub metadata: HashMap<String, Value>,
}

impl<const R: usize> STrajectory<R>
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
    /// use brahe::trajectories::{STrajectory, STrajectory6, Trajectory};
    /// let traj = STrajectory6::new(); // 6-dimensional static trajectory
    /// assert_eq!(traj.len(), 0);
    ///
    /// // Or specify dimension explicitly
    /// let traj: STrajectory<3> = STrajectory::new(); // 3-dimensional static trajectory
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
            metadata: HashMap::new(),
        }
    }

    /// Sets the interpolation method using builder pattern.
    ///
    /// This method consumes self and returns a new trajectory with the specified
    /// interpolation method, allowing for method chaining.
    ///
    /// # Arguments
    /// * `interpolation_method` - Method to use for state interpolation between epochs
    ///
    /// # Returns
    /// Self with updated interpolation method
    ///
    /// # Examples
    /// ```rust
    /// use brahe::trajectories::{STrajectory6, InterpolationMethod, TrajectoryEvictionPolicy};
    /// let traj = STrajectory6::new()
    ///     .with_interpolation_method(InterpolationMethod::Linear);
    /// ```
    pub fn with_interpolation_method(mut self, interpolation_method: InterpolationMethod) -> Self {
        self.interpolation_method = interpolation_method;
        self
    }

    /// Sets the eviction policy to keep a maximum number of states using builder pattern.
    ///
    /// This method consumes self and returns a new trajectory with the specified
    /// eviction policy, allowing for method chaining.
    ///
    /// # Arguments
    /// * `max_size` - Maximum number of states to retain (must be >= 1)
    ///
    /// # Returns
    /// Self with updated eviction policy
    ///
    /// # Panics
    /// Panics if max_size is less than 1
    ///
    /// # Examples
    /// ```rust
    /// use brahe::trajectories::{STrajectory6, InterpolationMethod};
    /// let traj = STrajectory6::new()
    ///     .with_eviction_policy_max_size(100);
    /// ```
    pub fn with_eviction_policy_max_size(mut self, max_size: usize) -> Self {
        if max_size < 1 {
            panic!("Maximum size must be >= 1");
        }
        self.eviction_policy = TrajectoryEvictionPolicy::KeepCount;
        self.max_size = Some(max_size);
        self.max_age = None;
        self
    }

    /// Sets the eviction policy to keep states within a maximum age using builder pattern.
    ///
    /// This method consumes self and returns a new trajectory with the specified
    /// eviction policy, allowing for method chaining.
    ///
    /// # Arguments
    /// * `max_age` - Maximum age of states to retain in seconds (must be > 0.0)
    ///
    /// # Returns
    /// Self with updated eviction policy
    ///
    /// # Panics
    /// Panics if max_age is not positive
    ///
    /// # Examples
    /// ```rust
    /// use brahe::trajectories::{STrajectory6, InterpolationMethod};
    /// let traj = STrajectory6::new()
    ///     .with_eviction_policy_max_age(3600.0);
    /// ```
    pub fn with_eviction_policy_max_age(mut self, max_age: f64) -> Self {
        if max_age <= 0.0 {
            panic!("Maximum age must be > 0.0");
        }
        self.eviction_policy = TrajectoryEvictionPolicy::KeepWithinDuration;
        self.max_age = Some(max_age);
        self.max_size = None;
        self
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

    pub fn dimension(&self) -> usize {
        R
    }

    /// Convert the trajectory to a matrix representation
    /// Returns a matrix where rows are time points (epochs) and columns are state elements
    /// The matrix has shape (n_epochs, 6) for a 6-element state vector
    pub fn to_matrix(&self) -> Result<nalgebra::DMatrix<f64>, BraheError> {
        if self.states.is_empty() {
            return Err(BraheError::Error(
                "Cannot convert empty trajectory to matrix".to_string(),
            ));
        }

        let n_epochs = self.states.len();
        let n_elements = 6;

        let mut matrix = nalgebra::DMatrix::<f64>::zeros(n_epochs, n_elements);

        for (row_idx, state) in self.states.iter().enumerate() {
            for col_idx in 0..n_elements {
                matrix[(row_idx, col_idx)] = state[col_idx];
            }
        }

        Ok(matrix)
    }

}

impl<const R: usize> Default for STrajectory<R>
{
    /// Creates a trajectory with default settings (linear interpolation, no memory limits).
    fn default() -> Self {
        Self::new()
    }
}

/// Index implementation returns state vector at given index
///
/// This provides array-like access: `traj[i]` returns the state at index i.
/// For accessing both epoch and state together, use `.get(i)` method instead.
///
/// # Panics
/// Panics if index is out of bounds
impl<const R: usize> Index<usize> for STrajectory<R>
{
    type Output = SVector<f64, R>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.states[index]
    }
}

/// Iterator over trajectory (epoch, state) pairs
pub struct STrajectoryIterator<'a, const R: usize> {
    trajectory: &'a STrajectory<R>,
    index: usize,
}

impl<'a, const R: usize> Iterator for STrajectoryIterator<'a, R> {
    type Item = (Epoch, SVector<f64, R>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.trajectory.len() {
            let result = self.trajectory.get(self.index).ok();
            self.index += 1;
            result
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.trajectory.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, const R: usize> ExactSizeIterator for STrajectoryIterator<'a, R> {
    fn len(&self) -> usize {
        self.trajectory.len() - self.index
    }
}

/// IntoIterator implementation for iterating over (epoch, state) pairs
impl<'a, const R: usize> IntoIterator for &'a STrajectory<R> {
    type Item = (Epoch, SVector<f64, R>);
    type IntoIter = STrajectoryIterator<'a, R>;

    fn into_iter(self) -> Self::IntoIter {
        STrajectoryIterator {
            trajectory: self,
            index: 0,
        }
    }
}

impl<const R: usize> Trajectory for STrajectory<R> {
    type StateVector = SVector<f64, R>;

    fn from_data(
        epochs: Vec<Epoch>,
        states: Vec<Self::StateVector>,
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
            interpolation_method: InterpolationMethod::Linear,  // Default to Linear
            eviction_policy: TrajectoryEvictionPolicy::None,
            max_size: None,
            max_age: None,
            metadata: HashMap::new(),
        })
    }

    fn add(&mut self, epoch: Epoch, state: Self::StateVector) -> Result<(), BraheError> {
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

    fn epoch(&self, index: usize) -> Result<Epoch, BraheError> {
        if index >= self.epochs.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} epochs",
                index,
                self.epochs.len()
            )));
        }

        Ok(self.epochs[index])
    }

    fn state(&self, index: usize) -> Result<Self::StateVector, BraheError> {
        if index >= self.states.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            )));
        }

        Ok(self.states[index].clone())
    }

    fn nearest_state(&self, epoch: &Epoch) -> Result<(Epoch, Self::StateVector), BraheError> {
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

    fn len(&self) -> usize {
        self.states.len()
    }

    /// Returns true if the trajectory is empty
    fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    fn start_epoch(&self) -> Option<Epoch> {
        self.epochs.first().copied()
    }

    fn end_epoch(&self) -> Option<Epoch> {
        self.epochs.last().copied()
    }

    fn timespan(&self) -> Option<f64> {
        if self.epochs.len() < 2 {
            None
        } else {
            Some(*self.epochs.last().unwrap() - *self.epochs.first().unwrap())
        }
    }

    fn first(&self) -> Option<(Epoch, Self::StateVector)> {
        if self.epochs.is_empty() {
            None
        } else {
            Some((self.epochs[0], self.states[0].clone()))
        }
    }

    fn last(&self) -> Option<(Epoch, Self::StateVector)> {
        if self.epochs.is_empty() {
            None
        } else {
            let last_index = self.epochs.len() - 1;
            Some((self.epochs[last_index], self.states[last_index].clone()))
        }
    }

    fn clear(&mut self) {
        self.epochs.clear();
        self.states.clear();
    }

    fn remove_epoch(&mut self, epoch: &Epoch) -> Result<Self::StateVector, BraheError> {
        // This could be improved with binary search since epochs are sorted
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

    fn remove(&mut self, index: usize) -> Result<(Epoch, Self::StateVector), BraheError> {
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

    fn get(&self, index: usize) -> Result<(Epoch, Self::StateVector), BraheError> {
        if index >= self.states.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            )));
        }

        Ok((self.epochs[index], self.states[index].clone()))
    }

    fn index_before_epoch(&self, epoch: &Epoch) -> Result<usize, BraheError> {
        if self.epochs.is_empty() {
            return Err(BraheError::Error(
                "Cannot get index from empty trajectory".to_string(),
            ));
        }

        // If epoch is before the first state, error
        if epoch < &self.epochs[0] {
            return Err(BraheError::Error(
                "Epoch is before all states in trajectory".to_string(),
            ));
        }

        // Find the index at or before the epoch
        for i in (0..self.epochs.len()).rev() {
            if &self.epochs[i] <= epoch {
                return Ok(i);
            }
        }

        // Should never reach here given the checks above
        Err(BraheError::Error(
            "Failed to find index before epoch".to_string(),
        ))
    }

    fn index_after_epoch(&self, epoch: &Epoch) -> Result<usize, BraheError> {
        if self.epochs.is_empty() {
            return Err(BraheError::Error(
                "Cannot get index from empty trajectory".to_string(),
            ));
        }

        // If epoch is after the last state, error
        if epoch > self.epochs.last().unwrap() {
            return Err(BraheError::Error(
                "Epoch is after all states in trajectory".to_string(),
            ));
        }

        // Find the index at or after the epoch
        for i in 0..self.epochs.len() {
            if &self.epochs[i] >= epoch {
                return Ok(i);
            }
        }

        // Should never reach here given the checks above
        Err(BraheError::Error(
            "Failed to find index after epoch".to_string(),
        ))
    }

    fn set_eviction_policy_max_size(&mut self, max_size: usize) -> Result<(), BraheError> {
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

    fn set_eviction_policy_max_age(&mut self, max_age: f64) -> Result<(), BraheError> {
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

    fn get_eviction_policy(&self) -> TrajectoryEvictionPolicy {
        self.eviction_policy
    }
}

impl<const R: usize> Interpolatable for STrajectory<R> {
    fn set_interpolation_method(&mut self, method: InterpolationMethod) {
        self.interpolation_method = method;
    }

    fn get_interpolation_method(&self) -> InterpolationMethod {
        self.interpolation_method
    }
}

// Iterator implementation will be added later once trait bounds are resolved

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{Epoch, TimeSystem};
    use crate::utils::testing::setup_global_test_eop;
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector6;

    fn create_test_trajectory() -> STrajectory6 {
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

        STrajectory6::from_data(epochs, states).unwrap()
    }

    // STrajectory Trait Tests

    #[test]
    fn test_strajectory_strajectory_new() {
        let trajectory = STrajectory6::new();

        assert_eq!(trajectory.len(), 0);
        assert_eq!(trajectory.interpolation_method, InterpolationMethod::Linear);
        assert!(trajectory.is_empty());
    }

    #[test]
    fn test_strajectory_with_interpolation_method() {
        // Test creating trajectory with specific interpolation method using builder pattern
        let traj = STrajectory6::new()
            .with_interpolation_method(InterpolationMethod::Linear);
        assert_eq!(traj.get_interpolation_method(), InterpolationMethod::Linear);
        assert_eq!(traj.len(), 0);

        // Verify it works with adding states
        let mut traj = STrajectory6::new()
            .with_interpolation_method(InterpolationMethod::Linear);
        let t0 = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        traj.add(t0, state).unwrap();
        assert_eq!(traj.len(), 1);
        assert_eq!(traj.get_interpolation_method(), InterpolationMethod::Linear);
    }

    #[test]
    fn test_strajectory_with_eviction_policy_max_size_builder() {
        // Test builder pattern for max size eviction policy
        let traj = STrajectory6::new()
            .with_eviction_policy_max_size(5);

        assert_eq!(traj.get_eviction_policy(), TrajectoryEvictionPolicy::KeepCount);
        assert_eq!(traj.len(), 0);
    }

    #[test]
    fn test_strajectory_with_eviction_policy_max_age_builder() {
        // Test builder pattern for max age eviction policy
        let traj = STrajectory6::new()
            .with_eviction_policy_max_age(300.0);

        assert_eq!(traj.get_eviction_policy(), TrajectoryEvictionPolicy::KeepWithinDuration);
        assert_eq!(traj.len(), 0);
    }

    #[test]
    fn test_strajectory_builder_pattern_chaining() {
        // Test chaining multiple builder methods
        let mut traj = STrajectory6::new()
            .with_interpolation_method(InterpolationMethod::Linear)
            .with_eviction_policy_max_size(10);

        assert_eq!(traj.get_interpolation_method(), InterpolationMethod::Linear);
        assert_eq!(traj.get_eviction_policy(), TrajectoryEvictionPolicy::KeepCount);

        // Add states and verify eviction policy works
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        for i in 0..15 {
            let epoch = t0 + (i as f64 * 60.0);
            let state = Vector6::new(7000e3 + i as f64 * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0);
            traj.add(epoch, state).unwrap();
        }

        // Should only have 10 states due to eviction policy
        assert_eq!(traj.len(), 10);
    }

    #[test]
    fn test_strajectory_dimension() {
        let traj = STrajectory6::new();
        assert_eq!(traj.dimension(), 6);

        let traj3: STrajectory<3> = STrajectory::new();
        assert_eq!(traj3.dimension(), 3);

        let traj4: STrajectory<4> = STrajectory::new();
        assert_eq!(traj4.dimension(), 4);

        let traj12: STrajectory<12> = STrajectory::new();
        assert_eq!(traj12.dimension(), 12);
    }

    #[test]
    fn test_strajectory_to_matrix() {
        let t0 = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let epochs = vec![
            t0,
            t0 + 60.0,
            t0 + 120.0,
        ];
        let states = vec![
            Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            Vector6::new(11.0, 12.0, 13.0, 14.0, 15.0, 16.0),
            Vector6::new(21.0, 22.0, 23.0, 24.0, 25.0, 26.0),
        ];

        let traj = STrajectory6::from_data(epochs, states).unwrap();

        let matrix = traj.to_matrix().unwrap();

        // Matrix should be 3 rows (time points) x 6 columns (state elements)
        assert_eq!(matrix.nrows(), 3);
        assert_eq!(matrix.ncols(), 6);

        // Check first row (first state at t0)
        assert_eq!(matrix[(0, 0)], 1.0);
        assert_eq!(matrix[(0, 1)], 2.0);
        assert_eq!(matrix[(0, 2)], 3.0);
        assert_eq!(matrix[(0, 3)], 4.0);
        assert_eq!(matrix[(0, 4)], 5.0);
        assert_eq!(matrix[(0, 5)], 6.0);

        // Check second row (second state at t1)
        assert_eq!(matrix[(1, 0)], 11.0);
        assert_eq!(matrix[(1, 1)], 12.0);
        assert_eq!(matrix[(1, 2)], 13.0);
        assert_eq!(matrix[(1, 3)], 14.0);
        assert_eq!(matrix[(1, 4)], 15.0);
        assert_eq!(matrix[(1, 5)], 16.0);

        // Check third row (third state at t2)
        assert_eq!(matrix[(2, 0)], 21.0);
        assert_eq!(matrix[(2, 1)], 22.0);
        assert_eq!(matrix[(2, 2)], 23.0);
        assert_eq!(matrix[(2, 3)], 24.0);
        assert_eq!(matrix[(2, 4)], 25.0);
        assert_eq!(matrix[(2, 5)], 26.0);

        // Check first column (first element of each state over time)
        assert_eq!(matrix[(0, 0)], 1.0);
        assert_eq!(matrix[(1, 0)], 11.0);
        assert_eq!(matrix[(2, 0)], 21.0);
    }

    // Default Trait Tests

    #[test]
    fn test_strajectory_default() {
        let trajectory = STrajectory6::default();
        assert_eq!(trajectory.len(), 0);
        assert_eq!(trajectory.interpolation_method, InterpolationMethod::Linear);
        assert!(trajectory.is_empty());
    }

    // Index Trait Tests

    #[test]
    fn test_strajectory_index_index() {
        let trajectory = create_test_trajectory();

        // Test indexing returns state vectors
        let state0 = &trajectory[0];
        assert_abs_diff_eq!(state0[0], 7000e3, epsilon = 1.0);

        let state1 = &trajectory[1];
        assert_abs_diff_eq!(state1[0], 7100e3, epsilon = 1.0);

        let state2 = &trajectory[2];
        assert_abs_diff_eq!(state2[0], 7200e3, epsilon = 1.0);
    }

    #[test]
    #[should_panic]
    fn test_strajectory_index_index_out_of_bounds() {
        let trajectory = create_test_trajectory();
        let _ = &trajectory[10]; // Should panic
    }

    // IntoIterator Trait Tests

    #[test]
    fn test_strajectory_intoiterator_into_iter() {
        let trajectory = create_test_trajectory();

        let mut count = 0;
        for (epoch, state) in &trajectory {
            match count {
                0 => {
                    assert_eq!(epoch.jd(), 2451545.0);
                    assert_abs_diff_eq!(state[0], 7000e3, epsilon = 1.0);
                }
                1 => {
                    assert_eq!(epoch.jd(), 2451545.1);
                    assert_abs_diff_eq!(state[0], 7100e3, epsilon = 1.0);
                }
                2 => {
                    assert_eq!(epoch.jd(), 2451545.2);
                    assert_abs_diff_eq!(state[0], 7200e3, epsilon = 1.0);
                }
                _ => panic!("Too many iterations"),
            }
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_strajectory_intoiterator_into_iter_empty() {
        let trajectory = STrajectory6::new();

        let mut count = 0;
        for _ in &trajectory {
            count += 1;
        }
        assert_eq!(count, 0);
    }

    #[test]
    fn test_strajectory_iterator_iterator_size_hint() {
        let trajectory = create_test_trajectory();

        let iter = trajectory.into_iter();
        let (lower, upper) = iter.size_hint();
        assert_eq!(lower, 3);
        assert_eq!(upper, Some(3));
    }

    #[test]
    fn test_strajectory_iterator_iterator_len() {
        let trajectory = create_test_trajectory();

        let iter = trajectory.into_iter();
        assert_eq!(iter.len(), 3);
    }

    // Trajectory Trait Tests

    #[test]
    fn test_strajectory_trajectory_add() {
        let mut trajectory = STrajectory6::new();

        // Add states in order
        let epoch1 = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state1 = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        trajectory.add(epoch1, state1).unwrap();

        let epoch3 = Epoch::from_jd(2451545.2, TimeSystem::UTC);
        let state3 = Vector6::new(7200e3, 0.0, 0.0, 0.0, 7.7e3, 0.0);
        trajectory.add(epoch3, state3).unwrap();

        // Add a state in between
        let epoch2 = Epoch::from_jd(2451545.1, TimeSystem::UTC);
        let state2 = Vector6::new(7100e3, 0.0, 0.0, 0.0, 7.6e3, 0.0);
        trajectory.add(epoch2, state2).unwrap();

        assert_eq!(trajectory.len(), 3);
        assert_eq!(trajectory.epochs[0].jd(), 2451545.0);
        assert_eq!(trajectory.epochs[1].jd(), 2451545.1);
        assert_eq!(trajectory.epochs[2].jd(), 2451545.2);
    }

    #[test]
    fn test_strajectory_trajectory_state() {
        let trajectory = create_test_trajectory();

        // Test valid indices
        let state0 = trajectory.state(0).unwrap();
        assert_eq!(state0[0], 7000e3);

        let state1 = trajectory.state(1).unwrap();
        assert_eq!(state1[0], 7100e3);

        let state2 = trajectory.state(2).unwrap();
        assert_eq!(state2[0], 7200e3);

        // Test invalid index
        assert!(trajectory.state(10).is_err());
    }

    #[test]
    fn test_strajectory_trajectory_epoch() {
        let trajectory = create_test_trajectory();

        // Test valid indices
        let epoch0 = trajectory.epoch(0).unwrap();
        assert_eq!(epoch0.jd(), 2451545.0);

        let epoch1 = trajectory.epoch(1).unwrap();
        assert_eq!(epoch1.jd(), 2451545.1);

        let epoch2 = trajectory.epoch(2).unwrap();
        assert_eq!(epoch2.jd(), 2451545.2);

        // Test invalid index
        assert!(trajectory.epoch(10).is_err());
    }

    #[test]
    fn test_strajectory_trajectory_nearest_state() {
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

        // Request a time right before the second state
        let (epoch, state) = trajectory
            .nearest_state(&Epoch::from_jd(2451545.0999, TimeSystem::UTC))
            .unwrap();
        assert_eq!(epoch.jd(), 2451545.1);
        assert_eq!(state[0], 7100e3);

        // Request a time after the last state
        let (epoch, state) = trajectory
            .nearest_state(&Epoch::from_jd(2451545.3, TimeSystem::UTC))
            .unwrap();
        assert_eq!(epoch.jd(), 2451545.2);
        assert_eq!(state[0], 7200e3);
    }

    #[test]
    fn test_strajectory_trajectory_len() {
        let mut trajectory = STrajectory6::new();
        assert_eq!(trajectory.len(), 0);

        trajectory.add(
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0)
        ).unwrap();
        assert_eq!(trajectory.len(), 1);

        trajectory.add(
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
            Vector6::new(7100e3, 0.0, 0.0, 0.0, 7.6e3, 0.0)
        ).unwrap();
        assert_eq!(trajectory.len(), 2);
    }

    #[test]
    fn test_strajectory_trajectory_is_empty() {
        let mut trajectory = STrajectory6::new();
        assert!(trajectory.is_empty());

        trajectory.add(
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0)
        ).unwrap();
        assert!(!trajectory.is_empty());

        trajectory.clear();
        assert!(trajectory.is_empty());
    }

    #[test]
    fn test_strajectory_trajectory_start_epoch() {
        let trajectory = create_test_trajectory();

        let start = trajectory.start_epoch().unwrap();
        assert_eq!(start.jd(), 2451545.0);

        // Test empty trajectory
        let empty_trajectory = STrajectory6::new();
        assert!(empty_trajectory.start_epoch().is_none());
    }

    #[test]
    fn test_strajectory_trajectory_end_epoch() {
        let trajectory = create_test_trajectory();

        let end = trajectory.end_epoch().unwrap();
        assert_eq!(end.jd(), 2451545.2);

        // Test empty trajectory
        let empty_trajectory = STrajectory6::new();
        assert!(empty_trajectory.end_epoch().is_none());
    }

    #[test]
    fn test_strajectory_trajectory_timespan() {
        let trajectory = create_test_trajectory();

        let span = trajectory.timespan().unwrap();
        assert_abs_diff_eq!(span, 0.2 * 86400.0, epsilon = 1.0); // 0.2 days in seconds

        // Test single state trajectory
        let mut single_trajectory = STrajectory6::new();
        single_trajectory.add(
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0)
        ).unwrap();
        assert!(single_trajectory.timespan().is_none());

        // Test empty trajectory
        let empty_trajectory = STrajectory6::new();
        assert!(empty_trajectory.timespan().is_none());
    }

    #[test]
    fn test_strajectory_trajectory_first() {
        // Test empty trajectory
        let empty_trajectory = STrajectory6::new();
        assert!(empty_trajectory.first().is_none());

        // Test single state trajectory
        let mut single_trajectory = STrajectory6::new();
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        single_trajectory.add(epoch, state).unwrap();

        let (first_epoch, first_state) = single_trajectory.first().unwrap();
        assert_eq!(first_epoch.jd(), 2451545.0);
        assert_eq!(first_state[0], 7000e3);

        // Test multi-state trajectory
        let trajectory = create_test_trajectory();
        let (first_epoch, first_state) = trajectory.first().unwrap();
        assert_eq!(first_epoch.jd(), 2451545.0);
        assert_eq!(first_state[0], 7000e3);
    }

    #[test]
    fn test_strajectory_trajectory_last() {
        // Test empty trajectory
        let empty_trajectory = STrajectory6::new();
        assert!(empty_trajectory.last().is_none());

        // Test single state trajectory
        let mut single_trajectory = STrajectory6::new();
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        single_trajectory.add(epoch, state).unwrap();

        let (last_epoch, last_state) = single_trajectory.last().unwrap();
        assert_eq!(last_epoch.jd(), 2451545.0);
        assert_eq!(last_state[0], 7000e3);

        // Test multi-state trajectory
        let trajectory = create_test_trajectory();
        let (last_epoch, last_state) = trajectory.last().unwrap();
        assert_eq!(last_epoch.jd(), 2451545.2);
        assert_eq!(last_state[0], 7200e3);
    }

    #[test]
    fn test_strajectory_trajectory_clear() {
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
    fn test_strajectory_trajectory_remove_epoch() {
        let mut trajectory = create_test_trajectory();

        let epoch_to_remove = Epoch::from_jd(2451545.1, TimeSystem::UTC);
        let removed_state = trajectory.remove_epoch(&epoch_to_remove).unwrap();
        assert_eq!(removed_state[0], 7100e3);
        assert_eq!(trajectory.len(), 2);

        // Test error case
        let non_existent_epoch = Epoch::from_jd(2451546.0, TimeSystem::UTC);
        assert!(trajectory.remove_epoch(&non_existent_epoch).is_err());
    }

    #[test]
    fn test_strajectory_trajectory_remove() {
        let mut trajectory = create_test_trajectory();

        let (removed_epoch, removed_state) = trajectory.remove(1).unwrap();
        assert_eq!(removed_epoch.jd(), 2451545.1);
        assert_eq!(removed_state[0], 7100e3);
        assert_eq!(trajectory.len(), 2);

        // Test error case
        assert!(trajectory.remove(10).is_err());
    }

    #[test]
    fn test_strajectory_trajectory_get() {
        let trajectory = create_test_trajectory();

        let (epoch, state) = trajectory.get(0).unwrap();
        assert_eq!(epoch.jd(), 2451545.0);
        assert_eq!(state[0], 7000e3);

        // Test bounds checking
        assert!(trajectory.get(10).is_err());
    }

    #[test]
    fn test_strajectory_trajectory_index_before_epoch() {
        // Create a trajectory with states at epochs: t0, t0+60s, t0+120s
        let t0 = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let epochs = vec![
            t0,
            t0 + 60.0,
            t0 + 120.0,
        ];
        let states = vec![
            Vector6::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Vector6::new(2.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Vector6::new(3.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ];
        let traj = STrajectory6::from_data(epochs, states).unwrap();

        // Test finding index before t0 (should error - before all states)
        let before_t0 = t0 - 10.0;
        assert!(traj.index_before_epoch(&before_t0).is_err());

        // Test finding index before t0+30s (should return index 0)
        let t0_plus_30 = t0 + 30.0;
        let idx = traj.index_before_epoch(&t0_plus_30).unwrap();
        assert_eq!(idx, 0);

        // Test finding index before t0+60s (should return index 1 - exact match)
        let t0_plus_60 = t0 + 60.0;
        let idx = traj.index_before_epoch(&t0_plus_60).unwrap();
        assert_eq!(idx, 1);

        // Test finding index before t0+90s (should return index 1)
        let t0_plus_90 = t0 + 90.0;
        let idx = traj.index_before_epoch(&t0_plus_90).unwrap();
        assert_eq!(idx, 1);

        // Test finding index before t0+120s (should return index 2 - exact match)
        let t0_plus_120 = t0 + 120.0;
        let idx = traj.index_before_epoch(&t0_plus_120).unwrap();
        assert_eq!(idx, 2);

        // Test finding index before t0+150s (should return index 2)
        let t0_plus_150 = t0 + 150.0;
        let idx = traj.index_before_epoch(&t0_plus_150).unwrap();
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_strajectory_trajectory_index_after_epoch() {
        // Create a trajectory with states at epochs: t0, t0+60s, t0+120s
        let t0 = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let epochs = vec![
            t0,
            t0 + 60.0,
            t0 + 120.0,
        ];
        let states = vec![
            Vector6::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Vector6::new(2.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Vector6::new(3.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ];
        let traj = STrajectory6::from_data(epochs, states).unwrap();

        // Test finding index after t0-30s (should return index 0)
        let before_t0 = t0 - 30.0;
        let idx = traj.index_after_epoch(&before_t0).unwrap();
        assert_eq!(idx, 0);

        // Test finding index after t0 (should return index 0 - exact match)
        let idx = traj.index_after_epoch(&t0).unwrap();
        assert_eq!(idx, 0);

        // Test finding index after t0+30s (should return index 1)
        let t0_plus_30 = t0 + 30.0;
        let idx = traj.index_after_epoch(&t0_plus_30).unwrap();
        assert_eq!(idx, 1);

        // Test finding index after t0+60s (should return index 1 - exact match)
        let t0_plus_60 = t0 + 60.0;
        let idx = traj.index_after_epoch(&t0_plus_60).unwrap();
        assert_eq!(idx, 1);

        // Test finding index after t0+90s (should return index 2)
        let t0_plus_90 = t0 + 90.0;
        let idx = traj.index_after_epoch(&t0_plus_90).unwrap();
        assert_eq!(idx, 2);

        // Test finding index after t0+120s (should return index 2 - exact match)
        let t0_plus_120 = t0 + 120.0;
        let idx = traj.index_after_epoch(&t0_plus_120).unwrap();
        assert_eq!(idx, 2);

        // Test finding index after t0+150s (should error - after all states)
        let t0_plus_150 = t0 + 150.0;
        assert!(traj.index_after_epoch(&t0_plus_150).is_err());
    }

    #[test]
    fn test_strajectory_trajectory_state_before_epoch() {
        // Create a trajectory with distinguishable states at 3 epochs
        let t0 = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let epochs = vec![
            t0,
            t0 + 60.0,
            t0 + 120.0,
        ];
        let states = vec![
            Vector6::new(1000.0, 100.0, 10.0, 1.0, 0.1, 0.01),
            Vector6::new(2000.0, 200.0, 20.0, 2.0, 0.2, 0.02),
            Vector6::new(3000.0, 300.0, 30.0, 3.0, 0.3, 0.03),
        ];
        let traj = STrajectory6::from_data(epochs, states).unwrap();

        // Test error case for epoch before all states
        let before_t0 = t0 - 10.0;
        assert!(traj.state_before_epoch(&before_t0).is_err());

        // Test that state_before_epoch returns correct (epoch, state) tuples
        // Test at t0+30s (should return first state)
        let t0_plus_30 = t0 + 30.0;
        let (epoch, state) = traj.state_before_epoch(&t0_plus_30).unwrap();
        assert_eq!(epoch, t0);
        assert_eq!(state[0], 1000.0);
        assert_eq!(state[1], 100.0);

        // Test at exact match t0+60s (should return second state)
        let t0_plus_60 = t0 + 60.0;
        let (epoch, state) = traj.state_before_epoch(&t0_plus_60).unwrap();
        assert_eq!(epoch, t0 + 60.0);
        assert_eq!(state[0], 2000.0);
        assert_eq!(state[1], 200.0);

        // Test at t0+90s (should return second state)
        let t0_plus_90 = t0 + 90.0;
        let (epoch, state) = traj.state_before_epoch(&t0_plus_90).unwrap();
        assert_eq!(epoch, t0 + 60.0);
        assert_eq!(state[0], 2000.0);
        assert_eq!(state[1], 200.0);

        // Test at t0+150s (should return third state)
        let t0_plus_150 = t0 + 150.0;
        let (epoch, state) = traj.state_before_epoch(&t0_plus_150).unwrap();
        assert_eq!(epoch, t0 + 120.0);
        assert_eq!(state[0], 3000.0);
        assert_eq!(state[1], 300.0);

        // Verify it uses the default trait implementation correctly by checking
        // that it produces the same result as calling index_before_epoch + get
        let t0_plus_45 = t0 + 45.0;
        let idx = traj.index_before_epoch(&t0_plus_45).unwrap();
        let (expected_epoch, expected_state) = traj.get(idx).unwrap();
        let (actual_epoch, actual_state) = traj.state_before_epoch(&t0_plus_45).unwrap();
        assert_eq!(actual_epoch, expected_epoch);
        assert_eq!(actual_state, expected_state);
    }

    #[test]
    fn test_strajectory_trajectory_state_after_epoch() {
        // Create a trajectory with distinguishable states at 3 epochs
        let t0 = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let epochs = vec![
            t0,
            t0 + 60.0,
            t0 + 120.0,
        ];
        let states = vec![
            Vector6::new(1000.0, 100.0, 10.0, 1.0, 0.1, 0.01),
            Vector6::new(2000.0, 200.0, 20.0, 2.0, 0.2, 0.02),
            Vector6::new(3000.0, 300.0, 30.0, 3.0, 0.3, 0.03),
        ];
        let traj = STrajectory6::from_data(epochs, states).unwrap();

        // Test error case for epoch after all states
        let after_t0_120 = t0 + 150.0;
        assert!(traj.state_after_epoch(&after_t0_120).is_err());

        // Test that state_after_epoch returns correct (epoch, state) tuples
        // Test at t0-30s (should return first state)
        let before_t0 = t0 - 30.0;
        let (epoch, state) = traj.state_after_epoch(&before_t0).unwrap();
        assert_eq!(epoch, t0);
        assert_eq!(state[0], 1000.0);
        assert_eq!(state[1], 100.0);

        // Test at exact match t0 (should return first state)
        let (epoch, state) = traj.state_after_epoch(&t0).unwrap();
        assert_eq!(epoch, t0);
        assert_eq!(state[0], 1000.0);
        assert_eq!(state[1], 100.0);

        // Test at t0+30s (should return second state)
        let t0_plus_30 = t0 + 30.0;
        let (epoch, state) = traj.state_after_epoch(&t0_plus_30).unwrap();
        assert_eq!(epoch, t0 + 60.0);
        assert_eq!(state[0], 2000.0);
        assert_eq!(state[1], 200.0);

        // Test at exact match t0+60s (should return second state)
        let t0_plus_60 = t0 + 60.0;
        let (epoch, state) = traj.state_after_epoch(&t0_plus_60).unwrap();
        assert_eq!(epoch, t0 + 60.0);
        assert_eq!(state[0], 2000.0);
        assert_eq!(state[1], 200.0);

        // Test at t0+90s (should return third state)
        let t0_plus_90 = t0 + 90.0;
        let (epoch, state) = traj.state_after_epoch(&t0_plus_90).unwrap();
        assert_eq!(epoch, t0 + 120.0);
        assert_eq!(state[0], 3000.0);
        assert_eq!(state[1], 300.0);

        // Verify it uses the default trait implementation correctly by checking
        // that it produces the same result as calling index_after_epoch + get
        let t0_plus_45 = t0 + 45.0;
        let idx = traj.index_after_epoch(&t0_plus_45).unwrap();
        let (expected_epoch, expected_state) = traj.get(idx).unwrap();
        let (actual_epoch, actual_state) = traj.state_after_epoch(&t0_plus_45).unwrap();
        assert_eq!(actual_epoch, expected_epoch);
        assert_eq!(actual_state, expected_state);
    }

    #[test]
    fn test_strajectory_trajectory_get_eviction_policy() {
        let mut traj = STrajectory6::new();

        // Default is None
        assert_eq!(traj.get_eviction_policy(), TrajectoryEvictionPolicy::None);

        // Set to KeepCount
        traj.set_eviction_policy_max_size(10).unwrap();
        assert_eq!(traj.get_eviction_policy(), TrajectoryEvictionPolicy::KeepCount);

        // Set to KeepWithinDuration
        traj.set_eviction_policy_max_age(100.0).unwrap();
        assert_eq!(traj.get_eviction_policy(), TrajectoryEvictionPolicy::KeepWithinDuration);
    }

    #[test]
    fn test_strajectory_set_eviction_policy_max_size() {
        let mut traj = STrajectory6::new();

        // Add 5 states
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        for i in 0..5 {
            let epoch = t0 + (i as f64 * 60.0);
            let state = Vector6::new(7000e3 + i as f64 * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0);
            traj.add(epoch, state).unwrap();
        }

        assert_eq!(traj.len(), 5);

        // Set max size to 3
        traj.set_eviction_policy_max_size(3).unwrap();

        // Should only have 3 most recent states
        assert_eq!(traj.len(), 3);

        // First state should be the 3rd original state (oldest 2 evicted)
        let first_state = traj.state(0).unwrap();
        assert_abs_diff_eq!(first_state[0], 7000e3 + 2000.0, epsilon = 1.0);

        // Add another state - should still maintain max size
        let new_epoch = t0 + 5.0 * 60.0;
        let new_state = Vector6::new(7000e3 + 5000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add(new_epoch, new_state).unwrap();

        assert_eq!(traj.len(), 3);
        assert_eq!(traj.state(0).unwrap()[0], 7000e3 + 3000.0);

        // Test error case
        assert!(traj.set_eviction_policy_max_size(0).is_err());
    }

    #[test]
    fn test_strajectory_set_eviction_policy_max_age() {
        let mut traj = STrajectory6::new();

        // Add states spanning 5 minutes
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        for i in 0..6 {
            let epoch = t0 + (i as f64 * 60.0); // 0, 60, 120, 180, 240, 300 seconds
            let state = Vector6::new(7000e3 + i as f64 * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0);
            traj.add(epoch, state).unwrap();
        }

        assert_eq!(traj.len(), 6);

        // Set max age to 240 seconds
        traj.set_eviction_policy_max_age(240.0).unwrap();
        assert_eq!(traj.len(), 5);
        assert_eq!(traj.epoch(0).unwrap(), t0 + 60.0);
        assert_eq!(traj.state(0).unwrap()[0], 7000e3 + 1000.0);

        
        // Set max age to 239 seconds
        traj.set_eviction_policy_max_age(239.0).unwrap();
        assert_eq!(traj.len(), 4);
        assert_eq!(traj.epoch(0).unwrap(), t0 + 120.0);
        assert_eq!(traj.state(0).unwrap()[0], 7000e3 + 2000.0);

        // Test error case
        assert!(traj.set_eviction_policy_max_age(0.0).is_err());
        assert!(traj.set_eviction_policy_max_age(-10.0).is_err());
    }

    // Interpolatable Trait Tests

    #[test]
    fn test_strajectory_interpolatable_get_interpolation_method() {
        let mut traj = STrajectory6::new();

        // Test default interpolation method is Linear
        assert_eq!(traj.get_interpolation_method(), InterpolationMethod::Linear);

        // Test setting to Linear explicitly
        traj.set_interpolation_method(InterpolationMethod::Linear);
        assert_eq!(traj.get_interpolation_method(), InterpolationMethod::Linear);
    }

    #[test]
    fn test_strajectory_interpolatable_interpolate_linear() {
        // Setup EOP for any frame conversions if needed
        setup_global_test_eop();

        let t0 = Epoch::from_jd(2451545.0, TimeSystem::UTC);

        // Create a trajectory with 3 states at t0, t0+60s, t0+120s with distinct position values
        let epochs = vec![
            t0,
            t0 + 60.0,
            t0 + 120.0,
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7060e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7120e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
        ];
        let traj = STrajectory6::from_data(epochs, states).unwrap();

        // Test interpolate_linear at t0+30s (midpoint between first two states)
        // Should be halfway between [7000e3, ...] and [7060e3, ...]
        let t_mid = t0 + 30.0;
        let state_mid = traj.interpolate_linear(&t_mid).unwrap();
        assert_abs_diff_eq!(state_mid[0], 7030e3, epsilon = 1e-6);
        assert_abs_diff_eq!(state_mid[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(state_mid[2], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(state_mid[3], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(state_mid[4], 7.5e3, epsilon = 1e-6);
        assert_abs_diff_eq!(state_mid[5], 0.0, epsilon = 1e-6);

        // Test interpolate_linear at exact epochs - should return exact states
        let state_t0 = traj.interpolate_linear(&t0).unwrap();
        assert_abs_diff_eq!(state_t0[0], 7000e3, epsilon = 1e-6);

        let state_t60 = traj.interpolate_linear(&(t0 + 60.0)).unwrap();
        assert_abs_diff_eq!(state_t60[0], 7060e3, epsilon = 1e-6);

        let state_t120 = traj.interpolate_linear(&(t0 + 120.0)).unwrap();
        assert_abs_diff_eq!(state_t120[0], 7120e3, epsilon = 1e-6);

        // Test interpolate_linear at t0+90s - should be between second and third states
        // Should be halfway between [7060e3, ...] and [7120e3, ...]
        let t_90 = t0 + 90.0;
        let state_90 = traj.interpolate_linear(&t_90).unwrap();
        assert_abs_diff_eq!(state_90[0], 7090e3, epsilon = 1e-6);
        assert_abs_diff_eq!(state_90[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(state_90[2], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(state_90[3], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(state_90[4], 7.5e3, epsilon = 1e-6);
        assert_abs_diff_eq!(state_90[5], 0.0, epsilon = 1e-6);

        // Test error case: single state trajectory (should just return that state)
        let single_epoch = vec![t0];
        let single_state = vec![Vector6::new(8000e3, 100.0, 200.0, 1.0, 2.0, 3.0)];
        let single_traj = STrajectory6::from_data(single_epoch, single_state).unwrap();

        let result = single_traj.interpolate_linear(&t0).unwrap();
        assert_abs_diff_eq!(result[0], 8000e3, epsilon = 1e-6);
        assert_abs_diff_eq!(result[1], 100.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[2], 200.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[3], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[4], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[5], 3.0, epsilon = 1e-6);

        // Test error case: interpolation on single state trajectory at different epoch
        let different_epoch = t0 + 10.0;
        assert!(single_traj.interpolate_linear(&different_epoch).is_err());

        // Test error case: interpolation on empty trajectory
        let empty_traj = STrajectory6::new();
        assert!(empty_traj.interpolate_linear(&t0).is_err());
    }

    #[test]
    fn test_strajectory_interpolatable_interpolate() {
        // Setup EOP for any frame conversions if needed
        setup_global_test_eop();

        let t0 = Epoch::from_jd(2451545.0, TimeSystem::UTC);

        // Create a trajectory with 3 states at different epochs
        let epochs = vec![
            t0,
            t0 + 60.0,
            t0 + 120.0,
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7060e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7120e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
        ];
        let traj = STrajectory6::from_data(epochs, states).unwrap();

        // Test that interpolate() returns same result as interpolate_linear() for the same epoch
        let t_test = t0 + 30.0;
        let result_interpolate = traj.interpolate(&t_test).unwrap();
        let result_linear = traj.interpolate_linear(&t_test).unwrap();

        assert_abs_diff_eq!(result_interpolate[0], result_linear[0], epsilon = 1e-6);
        assert_abs_diff_eq!(result_interpolate[1], result_linear[1], epsilon = 1e-6);
        assert_abs_diff_eq!(result_interpolate[2], result_linear[2], epsilon = 1e-6);
        assert_abs_diff_eq!(result_interpolate[3], result_linear[3], epsilon = 1e-6);
        assert_abs_diff_eq!(result_interpolate[4], result_linear[4], epsilon = 1e-6);
        assert_abs_diff_eq!(result_interpolate[5], result_linear[5], epsilon = 1e-6);

        // Verify the actual values for completeness
        assert_abs_diff_eq!(result_interpolate[0], 7030e3, epsilon = 1e-6);
    }
}
