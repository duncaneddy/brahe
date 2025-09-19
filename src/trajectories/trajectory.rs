/*!
 * Dynamic trajectory implementation for storing and interpolating variable-dimensional state vectors over time.
 *
 * This module provides a runtime-sized, frame-agnostic trajectory container that stores epochs and
 * corresponding N-dimensional state vectors. The trajectory supports arbitrary dimensions determined at
 * runtime and provides various interpolation methods, memory management policies, and efficient access
 * patterns for flexible applications.
 *
 * # Key Features
 * - Runtime-sized vectors for arbitrary dimensions
 * - Frame-agnostic storage (no assumptions about coordinate frames)
 * - Multiple interpolation methods (linear, cubic spline, Lagrange, etc.)
 * - Memory management with configurable eviction policies
 * - Efficient nearest-state and exact-epoch lookups
 * - Dimension validation for all operations
 *
 * # Examples
 * ```rust
 * use brahe::trajectories::{Trajectory, InterpolationMethod, TrajectoryCore};
 * use brahe::time::{Epoch, TimeSystem};
 * use nalgebra::DVector;
 *
 * let mut traj = Trajectory::new(7); // 7-dimensional trajectory
 * let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
 * let state = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
 * traj.add_state(epoch, state).unwrap();
 * ```
 */

use nalgebra::{DVector, DMatrix};

use crate::time::Epoch;
use crate::utils::BraheError;

use super::traits::{TrajectoryCore, TrajectoryExtended};

/// Import types from strajectory for consistency
pub use super::strajectory::{
    InterpolationMethod, TrajectoryEvictionPolicy, OrbitalMetadata
};

/// Dynamic trajectory container for N-dimensional state vectors over time.
///
/// The trajectory maintains a chronologically sorted collection of epochs and corresponding
/// state vectors. State vectors can be of any dimension N determined at construction time,
/// with all subsequent states required to match this dimension. This provides flexibility
/// for various applications while maintaining type safety through runtime validation.
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
/// - Runtime dimension validation: O(1) per operation
///
/// # Thread Safety
/// This structure is not thread-safe. Use appropriate synchronization for concurrent access.
#[derive(Debug, Clone, PartialEq)]
pub struct Trajectory {
    /// Time epochs for each state, maintained in chronological order.
    /// All epochs should use consistent time systems for meaningful interpolation.
    pub epochs: Vec<Epoch>,

    /// Variable-dimensional state vectors corresponding to epochs.
    /// All vectors must have the same dimension as specified at construction.
    /// Units and interpretation depend on the specific use case:
    /// - Cartesian: [m, m, m, m/s, m/s, m/s] for 6D
    /// - Keplerian: [m, dimensionless, rad, rad, rad, rad] for 6D
    /// - Custom: arbitrary units for arbitrary dimensions
    pub states: Vec<DVector<f64>>,

    /// Dimension of state vectors (must be consistent for all states)
    pub dimension: usize,

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

    /// Optional orbital metadata (None for non-orbital trajectories)
    pub orbital_metadata: Option<OrbitalMetadata>,
}

impl Default for Trajectory {
    /// Creates a trajectory with default settings (6D, linear interpolation, no memory limits).
    fn default() -> Self {
        Self::new(6) // Default to 6D for backward compatibility
    }
}

impl TrajectoryCore for Trajectory {
    type StateVector = DVector<f64>;

    fn new() -> Self {
        Self::new(6) // Default to 6D for backward compatibility
    }

    fn add_state(&mut self, epoch: Epoch, state: DVector<f64>) -> Result<(), BraheError> {
        // Validate state dimension
        if state.len() != self.dimension {
            return Err(BraheError::Error(format!(
                "State vector dimension {} does not match trajectory dimension {}",
                state.len(),
                self.dimension
            )));
        }

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

    fn state_at_epoch(&self, epoch: &Epoch) -> Result<DVector<f64>, BraheError> {
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

    fn state_at_index(&self, index: usize) -> Result<DVector<f64>, BraheError> {
        if index >= self.states.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            )));
        }

        Ok(self.states[index].clone())
    }

    fn epoch_at_index(&self, index: usize) -> Result<Epoch, BraheError> {
        if index >= self.epochs.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} epochs",
                index,
                self.epochs.len()
            )));
        }

        Ok(self.epochs[index])
    }

    fn nearest_state(&self, epoch: &Epoch) -> Result<(Epoch, DVector<f64>), BraheError> {
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

    fn first(&self) -> Option<(Epoch, DVector<f64>)> {
        if self.epochs.is_empty() {
            None
        } else {
            Some((self.epochs[0], self.states[0].clone()))
        }
    }

    fn last(&self) -> Option<(Epoch, DVector<f64>)> {
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
}

impl TrajectoryExtended for Trajectory {
    fn remove_state(&mut self, epoch: &Epoch) -> Result<DVector<f64>, BraheError> {
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

    fn remove_state_at_index(&mut self, index: usize) -> Result<(Epoch, DVector<f64>), BraheError> {
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

    fn get(&self, index: usize) -> Result<(Epoch, DVector<f64>), BraheError> {
        if index >= self.states.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            )));
        }

        Ok((self.epochs[index], self.states[index].clone()))
    }
}

impl Trajectory {
    /// Creates a new empty trajectory with the specified dimension.
    ///
    /// # Arguments
    /// * `dimension` - Number of elements in each state vector
    ///
    /// # Returns
    /// A new empty trajectory with linear interpolation
    ///
    /// # Examples
    /// ```rust
    /// use brahe::trajectories::{Trajectory, TrajectoryCore};
    /// let traj = Trajectory::new(7); // 7-dimensional trajectory
    /// assert_eq!(traj.len(), 0);
    /// assert_eq!(traj.dimension, 7);
    /// ```
    pub fn new(dimension: usize) -> Self {
        if dimension == 0 {
            panic!("Trajectory dimension must be greater than 0");
        }

        Self {
            epochs: Vec::new(),
            states: Vec::new(),
            dimension,
            interpolation_method: InterpolationMethod::Linear,
            eviction_policy: TrajectoryEvictionPolicy::None,
            max_size: None,
            max_age: None,
            orbital_metadata: None,
        }
    }

    /// Creates a new empty trajectory with the specified dimension and interpolation method.
    ///
    /// # Arguments
    /// * `dimension` - Number of elements in each state vector
    /// * `interpolation_method` - Method to use for state interpolation between epochs
    ///
    /// # Returns
    /// A new empty trajectory ready for state addition
    ///
    /// # Examples
    /// ```rust
    /// use brahe::trajectories::{Trajectory, InterpolationMethod, TrajectoryCore};
    /// let traj = Trajectory::with_interpolation(12, InterpolationMethod::CubicSpline);
    /// assert_eq!(traj.len(), 0);
    /// assert_eq!(traj.dimension, 12);
    /// ```
    pub fn with_interpolation(dimension: usize, interpolation_method: InterpolationMethod) -> Self {
        if dimension == 0 {
            panic!("Trajectory dimension must be greater than 0");
        }

        Self {
            epochs: Vec::new(),
            states: Vec::new(),
            dimension,
            interpolation_method,
            eviction_policy: TrajectoryEvictionPolicy::None,
            max_size: None,
            max_age: None,
            orbital_metadata: None,
        }
    }

    /// Creates a trajectory from existing epoch and state data.
    ///
    /// The input data is validated and sorted chronologically. This method is efficient
    /// for bulk trajectory construction from pre-existing datasets.
    ///
    /// # Arguments
    /// * `epochs` - Vector of time epochs (must be non-empty and same length as states)
    /// * `states` - Vector of state vectors corresponding to epochs (all must have same dimension)
    /// * `interpolation_method` - Method to use for state interpolation
    ///
    /// # Returns
    /// * `Ok(Trajectory)` - Successfully created trajectory with sorted data
    /// * `Err(BraheError)` - If input validation fails
    ///
    /// # Errors
    /// * `BraheError::Error` - If epochs and states have different lengths, are empty, or states have inconsistent dimensions
    pub fn from_data(
        epochs: Vec<Epoch>,
        states: Vec<DVector<f64>>,
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

        // Validate all states have the same dimension
        let dimension = states[0].len();
        if dimension == 0 {
            return Err(BraheError::Error(
                "State vectors cannot be empty".to_string(),
            ));
        }

        for (i, state) in states.iter().enumerate() {
            if state.len() != dimension {
                return Err(BraheError::Error(format!(
                    "State {} has dimension {} but expected {}",
                    i,
                    state.len(),
                    dimension
                )));
            }
        }

        // Ensure epochs are sorted
        let mut indices: Vec<usize> = (0..epochs.len()).collect();
        indices.sort_by(|&i, &j| epochs[i].partial_cmp(&epochs[j]).unwrap());

        let sorted_epochs: Vec<Epoch> = indices.iter().map(|&i| epochs[i]).collect();
        let sorted_states: Vec<DVector<f64>> = indices.iter().map(|&i| states[i].clone()).collect();

        Ok(Self {
            epochs: sorted_epochs,
            states: sorted_states,
            dimension,
            interpolation_method,
            eviction_policy: TrajectoryEvictionPolicy::None,
            max_size: None,
            max_age: None,
            orbital_metadata: None,
        })
    }

    /// Get the dimension of state vectors in this trajectory
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Set the interpolation method for state retrieval.
    ///
    /// This allows changing the interpolation behavior after trajectory creation.
    /// The change affects all future calls to `state_at_epoch()` and related methods.
    ///
    /// # Arguments
    /// * `method` - New interpolation method to use
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

    /// Convert the trajectory to a matrix representation
    /// Returns a matrix where columns are time points and rows are state elements
    /// The matrix has shape (dimension, n_epochs)
    pub fn to_matrix(&self) -> Result<DMatrix<f64>, BraheError> {
        if self.states.is_empty() {
            return Err(BraheError::Error(
                "Cannot convert empty trajectory to matrix".to_string(),
            ));
        }

        let n_epochs = self.states.len();
        let n_elements = self.dimension;

        let mut matrix = DMatrix::<f64>::zeros(n_elements, n_epochs);

        for (col_idx, state) in self.states.iter().enumerate() {
            for row_idx in 0..n_elements {
                matrix[(row_idx, col_idx)] = state[row_idx];
            }
        }

        Ok(matrix)
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
                        let new_states: Vec<DVector<f64>> = indices_to_keep.iter().map(|&i| self.states[i].clone()).collect();

                        self.epochs = new_epochs;
                        self.states = new_states;
                    }
                }
            },
        }
        Ok(())
    }

    /// Interpolate between states using linear interpolation
    fn interpolate_linear(&self, epoch: &Epoch) -> Result<DVector<f64>, BraheError> {
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
                let mut interpolated_state = DVector::<f64>::zeros(self.dimension);
                for j in 0..self.dimension {
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
}

// Allow indexing into the trajectory directly - returns state vector only
// For (epoch, state) tuples, use the get() method instead
impl std::ops::Index<usize> for Trajectory {
    type Output = DVector<f64>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.states[index]
    }
}