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
 * use brahe::trajectories::DTrajectory;
 * use brahe::traits::{Trajectory, InterpolationMethod};
 * use brahe::time::{Epoch, TimeSystem};
 * use nalgebra::DVector;
 *
 * let mut traj = DTrajectory::new(7); // 7-dimensional trajectory
 * let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
 * let state = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
 * traj.add(epoch, state);
 * ```
 */

use nalgebra::{DMatrix, DVector};
use serde_json::Value;
use std::collections::HashMap;

use crate::math::{
    CovarianceInterpolationConfig, interpolate_covariance_sqrt_dmatrix,
    interpolate_covariance_two_wasserstein_dmatrix, interpolate_lagrange_dvector,
};
use crate::time::Epoch;
use crate::utils::BraheError;

use super::traits::{
    CovarianceInterpolationMethod, InterpolatableTrajectory, InterpolationConfig,
    InterpolationMethod, STMStorage, SensitivityStorage, Trajectory, TrajectoryEvictionPolicy,
};

/// Dynamic (runtime-sized) trajectory container for N-dimensional state vectors over time.
///
/// The trajectory maintains a chronologically sorted collection of epochs and corresponding
/// state vectors. State vectors can be of any dimension N determined at construction time,
/// with all subsequent states required to match this dimension. This provides flexibility
/// for various applications while maintaining type safety through runtime validation.
///
/// # Memory Management
/// The trajectory supports automatic memory management through configurable policies:
/// - Maximum state count (oldest states evicted first)
/// - Maximum age (states older than value evicted)
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
pub struct DTrajectory {
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

    /// Optional covariance matrices corresponding to each state.
    /// If present, must have the same length as `states` and each matrix
    /// must be square with dimension matching the state dimension.
    pub covariances: Option<Vec<DMatrix<f64>>>,

    /// Optional state transition matrices (STM) corresponding to each state.
    /// Each STM is dimension×dimension relating state changes: Φ(t,t₀) = ∂x(t)/∂x(t₀).
    /// If present, must have the same length as `states` and each matrix must be dimension×dimension.
    pub stms: Option<Vec<DMatrix<f64>>>,

    /// Optional sensitivity matrices corresponding to each state.
    /// Each matrix is dimension×param_dim: ∂x/∂p where x is the state and p are parameters.
    /// If present, must have the same length as `states`.
    pub sensitivities: Option<Vec<DMatrix<f64>>>,

    /// Sensitivity dimensions as (rows, cols) = (dimension, param_dim).
    /// Set when sensitivity storage is enabled.
    sensitivity_dimension: Option<(usize, usize)>,

    /// Dimension of state vectors (must be consistent for all states)
    pub dimension: usize,

    /// Interpolation method for state retrieval at arbitrary epochs.
    /// Default is linear interpolation for optimal performance/accuracy balance.
    pub interpolation_method: InterpolationMethod,

    /// Interpolation method for covariance retrieval at arbitrary epochs.
    /// Default is TwoWasserstein for proper positive semi-definiteness preservation.
    pub covariance_interpolation_method: CovarianceInterpolationMethod,

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

impl DTrajectory {
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
    /// use brahe::trajectories::DTrajectory;
    /// use brahe::traits::Trajectory;
    /// let traj = DTrajectory::new(7); // 7-dimensional trajectory
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
            covariances: None,
            stms: None,
            sensitivities: None,
            sensitivity_dimension: None,
            dimension,
            interpolation_method: InterpolationMethod::Linear,
            covariance_interpolation_method: CovarianceInterpolationMethod::TwoWasserstein,
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
    /// use brahe::trajectories::DTrajectory;
    /// use brahe::traits::InterpolationMethod;
    /// let traj = DTrajectory::new(6)
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
    /// use brahe::trajectories::DTrajectory;
    /// let traj = DTrajectory::new(6)
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
    /// use brahe::trajectories::DTrajectory;
    /// let traj = DTrajectory::new(6)
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

    /// Get the dimension of state vectors in this trajectory
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Convert the trajectory to a matrix representation
    /// Returns a matrix where rows are time points (epochs) and columns are state elements
    /// The matrix has shape (n_epochs, dimension)
    pub fn to_matrix(&self) -> Result<DMatrix<f64>, BraheError> {
        if self.states.is_empty() {
            return Err(BraheError::Error(
                "Cannot convert empty trajectory to matrix".to_string(),
            ));
        }

        let n_epochs = self.states.len();
        let n_elements = self.dimension;

        let mut matrix = DMatrix::<f64>::zeros(n_epochs, n_elements);

        for (row_idx, state) in self.states.iter().enumerate() {
            for col_idx in 0..n_elements {
                matrix[(row_idx, col_idx)] = state[col_idx];
            }
        }

        Ok(matrix)
    }

    /// Enable covariance storage
    ///
    /// Initializes the covariance vector with zero matrices for all existing states.
    /// After calling this, covariances can be added using `add_with_covariance()` or
    /// `set_covariance_at()`.
    pub fn enable_covariance_storage(&mut self) {
        if self.covariances.is_none() {
            // Initialize with zero matrices for all existing states
            let zero_cov = DMatrix::zeros(self.dimension, self.dimension);
            self.covariances = Some(vec![zero_cov; self.states.len()]);
        }
    }

    /// Add a state with its corresponding covariance matrix
    ///
    /// This automatically enables covariance storage if not already enabled.
    ///
    /// # Arguments
    /// * `epoch` - Time epoch
    /// * `state` - State vector
    /// * `covariance` - Covariance matrix (must be square with dimension matching state)
    ///
    /// # Panics
    /// Panics if state or covariance dimensions don't match trajectory dimension
    pub fn add_with_covariance(
        &mut self,
        epoch: Epoch,
        state: DVector<f64>,
        covariance: DMatrix<f64>,
    ) {
        // Validate dimensions
        if state.len() != self.dimension {
            panic!("State vector dimension does not match trajectory dimension.");
        }
        if covariance.nrows() != self.dimension || covariance.ncols() != self.dimension {
            panic!(
                "Covariance matrix dimensions {}x{} do not match trajectory dimension {}",
                covariance.nrows(),
                covariance.ncols(),
                self.dimension
            );
        }

        // Enable covariance storage if not already enabled
        if self.covariances.is_none() {
            self.enable_covariance_storage();
        }

        // Find the correct position to insert based on epoch
        let mut insert_idx = self.epochs.len();
        for (i, existing_epoch) in self.epochs.iter().enumerate() {
            if epoch < *existing_epoch {
                insert_idx = i;
                break;
            }
        }

        // Insert at the correct position
        self.epochs.insert(insert_idx, epoch);
        self.states.insert(insert_idx, state.clone());
        if let Some(ref mut covs) = self.covariances {
            covs.insert(insert_idx, covariance);
        }

        // Apply eviction policy after adding state
        self.apply_eviction_policy();
    }

    /// Set covariance matrix at a specific index
    ///
    /// Enables covariance storage if not already enabled.
    ///
    /// # Arguments
    /// * `index` - Index in the trajectory
    /// * `covariance` - Covariance matrix
    ///
    /// # Panics
    /// Panics if index is out of bounds or covariance dimensions are incorrect
    pub fn set_covariance_at(&mut self, index: usize, covariance: DMatrix<f64>) {
        if index >= self.states.len() {
            panic!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            );
        }
        if covariance.nrows() != self.dimension || covariance.ncols() != self.dimension {
            panic!(
                "Covariance matrix dimensions {}x{} do not match trajectory dimension {}",
                covariance.nrows(),
                covariance.ncols(),
                self.dimension
            );
        }

        // Enable covariance storage if not already enabled
        if self.covariances.is_none() {
            self.enable_covariance_storage();
        }

        // Set the covariance at the specified index
        if let Some(ref mut covs) = self.covariances {
            covs[index] = covariance;
        }
    }

    /// Get covariance matrix at a specific epoch (with interpolation)
    ///
    /// Returns None if covariance storage is not enabled or epoch is out of range.
    ///
    /// # Arguments
    /// * `epoch` - Time epoch to query
    ///
    /// # Returns
    /// Covariance matrix at the requested epoch (interpolated if necessary)
    pub fn covariance_at(&self, epoch: Epoch) -> Option<DMatrix<f64>> {
        // Check if covariance storage is enabled
        let covs = self.covariances.as_ref()?;

        // Check if trajectory has data
        if self.epochs.is_empty() {
            return None;
        }

        // Handle exact match at endpoint
        if let Some((idx, _)) = self.epochs.iter().enumerate().find(|(_, e)| **e == epoch) {
            return Some(covs[idx].clone());
        }

        // Find surrounding indices for interpolation
        let (idx_before, idx_after) = self.find_surrounding_indices(epoch)?;

        // Handle exact matches
        if self.epochs[idx_before] == epoch {
            return Some(covs[idx_before].clone());
        }
        if self.epochs[idx_after] == epoch {
            return Some(covs[idx_after].clone());
        }

        // Interpolation parameter
        let t0 = self.epochs[idx_before] - self.epoch_initial()?;
        let t1 = self.epochs[idx_after] - self.epoch_initial()?;
        let t = epoch - self.epoch_initial()?;
        let alpha = (t - t0) / (t1 - t0);

        let cov0 = &covs[idx_before];
        let cov1 = &covs[idx_after];

        // Dispatch based on covariance interpolation method
        let cov = match self.covariance_interpolation_method {
            CovarianceInterpolationMethod::MatrixSquareRoot => {
                interpolate_covariance_sqrt_dmatrix(cov0, cov1, alpha)
            }
            CovarianceInterpolationMethod::TwoWasserstein => {
                interpolate_covariance_two_wasserstein_dmatrix(cov0, cov1, alpha)
            }
        };

        Some(cov)
    }

    /// Helper to find initial epoch for interpolation
    fn epoch_initial(&self) -> Option<Epoch> {
        self.epochs.first().copied()
    }

    /// Helper to find surrounding indices for interpolation
    fn find_surrounding_indices(&self, epoch: Epoch) -> Option<(usize, usize)> {
        if self.epochs.is_empty() {
            return None;
        }

        // Check bounds
        if epoch < self.epochs[0] || epoch > *self.epochs.last()? {
            return None;
        }

        // Binary search to find the interval
        for i in 0..self.epochs.len() - 1 {
            if self.epochs[i] <= epoch && epoch <= self.epochs[i + 1] {
                return Some((i, i + 1));
            }
        }

        None
    }

    /// Apply eviction policy to manage trajectory memory
    fn apply_eviction_policy(&mut self) {
        match self.eviction_policy {
            TrajectoryEvictionPolicy::None => {
                // No eviction
            }
            TrajectoryEvictionPolicy::KeepCount => {
                if let Some(max_size) = self.max_size
                    && self.epochs.len() > max_size
                {
                    let to_remove = self.epochs.len() - max_size;
                    self.epochs.drain(0..to_remove);
                    self.states.drain(0..to_remove);
                    if let Some(ref mut covs) = self.covariances {
                        covs.drain(0..to_remove);
                    }
                    if let Some(ref mut stms) = self.stms {
                        stms.drain(0..to_remove);
                    }
                    if let Some(ref mut sens) = self.sensitivities {
                        sens.drain(0..to_remove);
                    }
                }
            }
            TrajectoryEvictionPolicy::KeepWithinDuration => {
                if let Some(max_age) = self.max_age
                    && let Some(&last_epoch) = self.epochs.last()
                {
                    let mut indices_to_keep = Vec::new();
                    for (i, &epoch) in self.epochs.iter().enumerate() {
                        if (last_epoch - epoch).abs() <= max_age {
                            indices_to_keep.push(i);
                        }
                    }

                    let new_epochs: Vec<Epoch> =
                        indices_to_keep.iter().map(|&i| self.epochs[i]).collect();
                    let new_states: Vec<DVector<f64>> = indices_to_keep
                        .iter()
                        .map(|&i| self.states[i].clone())
                        .collect();

                    self.epochs = new_epochs;
                    self.states = new_states;

                    // Also evict covariances if enabled
                    if let Some(ref mut covs) = self.covariances {
                        let new_covs: Vec<DMatrix<f64>> =
                            indices_to_keep.iter().map(|&i| covs[i].clone()).collect();
                        *covs = new_covs;
                    }

                    // Also evict STMs if enabled
                    if let Some(ref mut stms) = self.stms {
                        let new_stms: Vec<DMatrix<f64>> =
                            indices_to_keep.iter().map(|&i| stms[i].clone()).collect();
                        *stms = new_stms;
                    }

                    // Also evict sensitivities if enabled
                    if let Some(ref mut sens) = self.sensitivities {
                        let new_sens: Vec<DMatrix<f64>> =
                            indices_to_keep.iter().map(|&i| sens[i].clone()).collect();
                        *sens = new_sens;
                    }
                }
            }
        }
    }
}

impl Default for DTrajectory {
    fn default() -> Self {
        Self::new(6)
    }
}

impl DTrajectory {
    /// Add a state with optional covariance, STM, and sensitivity matrices.
    ///
    /// This method allows adding all trajectory data in a single operation,
    /// automatically enabling storage for any provided optional matrices.
    ///
    /// # Arguments
    /// * `epoch` - Time epoch
    /// * `state` - State vector (must match trajectory dimension)
    /// * `covariance` - Optional covariance matrix (dimension×dimension)
    /// * `stm` - Optional state transition matrix (dimension×dimension)
    /// * `sensitivity` - Optional sensitivity matrix (dimension×param_dim)
    ///
    /// # Panics
    /// Panics if:
    /// - State dimension doesn't match trajectory dimension
    /// - Matrix dimensions are incorrect
    /// - Sensitivity column count doesn't match previously enabled storage
    pub fn add_full(
        &mut self,
        epoch: Epoch,
        state: DVector<f64>,
        covariance: Option<DMatrix<f64>>,
        stm: Option<DMatrix<f64>>,
        sensitivity: Option<DMatrix<f64>>,
    ) {
        // Validate state dimension
        if state.len() != self.dimension {
            panic!(
                "State vector dimension {} does not match trajectory dimension {}",
                state.len(),
                self.dimension
            );
        }

        // Validate and auto-enable covariance storage
        if let Some(ref cov) = covariance {
            if cov.nrows() != self.dimension || cov.ncols() != self.dimension {
                panic!(
                    "Covariance dimensions {}×{} do not match expected {}×{}",
                    cov.nrows(),
                    cov.ncols(),
                    self.dimension,
                    self.dimension
                );
            }
            if self.covariances.is_none() {
                self.covariances = Some(vec![
                    DMatrix::zeros(self.dimension, self.dimension);
                    self.states.len()
                ]);
            }
        }

        // Validate and auto-enable STM storage
        if let Some(ref stm_val) = stm {
            if stm_val.nrows() != self.dimension || stm_val.ncols() != self.dimension {
                panic!(
                    "STM dimensions {}×{} do not match expected {}×{}",
                    stm_val.nrows(),
                    stm_val.ncols(),
                    self.dimension,
                    self.dimension
                );
            }
            if self.stms.is_none() {
                let identity = DMatrix::identity(self.dimension, self.dimension);
                self.stms = Some(vec![identity; self.states.len()]);
            }
        }

        // Validate and auto-enable sensitivity storage
        if let Some(ref sens) = sensitivity {
            if sens.nrows() != self.dimension {
                panic!(
                    "Sensitivity row count {} does not match state dimension {}",
                    sens.nrows(),
                    self.dimension
                );
            }

            // Check consistency with existing sensitivity dimension
            if let Some((_, existing_cols)) = self.sensitivity_dimension {
                if sens.ncols() != existing_cols {
                    panic!(
                        "Sensitivity column count {} does not match existing {}",
                        sens.ncols(),
                        existing_cols
                    );
                }
            } else if self.sensitivities.is_none() {
                let zero_sens = DMatrix::zeros(self.dimension, sens.ncols());
                self.sensitivities = Some(vec![zero_sens; self.states.len()]);
                self.sensitivity_dimension = Some((self.dimension, sens.ncols()));
            }
        }

        // Find insertion index (maintain sorted order)
        let mut insert_idx = self.epochs.len();
        for (i, existing_epoch) in self.epochs.iter().enumerate() {
            if epoch < *existing_epoch {
                insert_idx = i;
                break;
            }
        }

        // Insert into all vectors
        self.epochs.insert(insert_idx, epoch);
        self.states.insert(insert_idx, state);

        if let Some(ref mut covs) = self.covariances {
            if let Some(cov) = covariance {
                covs.insert(insert_idx, cov);
            } else {
                covs.insert(insert_idx, DMatrix::zeros(self.dimension, self.dimension));
            }
        }

        if let Some(ref mut stms) = self.stms {
            if let Some(stm_val) = stm {
                stms.insert(insert_idx, stm_val);
            } else {
                stms.insert(
                    insert_idx,
                    DMatrix::identity(self.dimension, self.dimension),
                );
            }
        }

        if let Some(ref mut sens) = self.sensitivities {
            if let Some(sens_val) = sensitivity {
                sens.insert(insert_idx, sens_val);
            } else if let Some((rows, cols)) = self.sensitivity_dimension {
                sens.insert(insert_idx, DMatrix::zeros(rows, cols));
            }
        }

        // Apply eviction policy
        self.apply_eviction_policy();
    }
}

// Allow indexing into the trajectory directly - returns state vector only
// For (epoch, state) tuples, use the get() method instead
/// Index implementation returns state vector at given index
///
/// # Panics
/// Panics if index is out of bounds
impl std::ops::Index<usize> for DTrajectory {
    type Output = DVector<f64>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.states[index]
    }
}

/// Iterator over trajectory (epoch, state) pairs
pub struct DTrajectoryIterator<'a> {
    trajectory: &'a DTrajectory,
    index: usize,
}

impl<'a> Iterator for DTrajectoryIterator<'a> {
    type Item = (Epoch, DVector<f64>);

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

impl<'a> ExactSizeIterator for DTrajectoryIterator<'a> {
    fn len(&self) -> usize {
        self.trajectory.len() - self.index
    }
}

/// IntoIterator implementation for iterating over (epoch, state) pairs
impl<'a> IntoIterator for &'a DTrajectory {
    type Item = (Epoch, DVector<f64>);
    type IntoIter = DTrajectoryIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        DTrajectoryIterator {
            trajectory: self,
            index: 0,
        }
    }
}

impl Trajectory for DTrajectory {
    type StateVector = DVector<f64>;

    fn from_data(epochs: Vec<Epoch>, states: Vec<Self::StateVector>) -> Result<Self, BraheError> {
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
            covariances: None,
            stms: None,
            sensitivities: None,
            sensitivity_dimension: None,
            dimension,
            interpolation_method: InterpolationMethod::Linear,
            covariance_interpolation_method: CovarianceInterpolationMethod::TwoWasserstein,
            eviction_policy: TrajectoryEvictionPolicy::None,
            max_size: None,
            max_age: None,
            metadata: HashMap::new(),
        })
    }

    fn add(&mut self, epoch: Epoch, state: DVector<f64>) {
        // Validate state dimension
        if state.len() != self.dimension {
            panic!("State vector dimension does not match trajectory dimension.");
        }

        // Find the correct position to insert based on epoch
        // Insert after any existing states at the same epoch to support
        // impulsive maneuvers where we want both pre- and post-maneuver states
        let mut insert_idx = self.epochs.len();
        for (i, existing_epoch) in self.epochs.iter().enumerate() {
            if epoch < *existing_epoch {
                insert_idx = i;
                break;
            }
            // If epochs are equal, continue to find the position after all equal epochs
        }

        // Insert at the correct position
        self.epochs.insert(insert_idx, epoch);
        self.states.insert(insert_idx, state.clone());

        // Maintain consistency for all optional arrays
        if let Some(ref mut covs) = self.covariances {
            covs.insert(insert_idx, DMatrix::zeros(self.dimension, self.dimension));
        }

        if let Some(ref mut stms) = self.stms {
            stms.insert(
                insert_idx,
                DMatrix::identity(self.dimension, self.dimension),
            );
        }

        if let Some(ref mut sens) = self.sensitivities
            && let Some((rows, cols)) = self.sensitivity_dimension
        {
            sens.insert(insert_idx, DMatrix::zeros(rows, cols));
        }

        // Apply eviction policy after adding state
        self.apply_eviction_policy();
    }

    fn epoch_at_idx(&self, index: usize) -> Result<Epoch, BraheError> {
        if index >= self.epochs.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} epochs",
                index,
                self.epochs.len()
            )));
        }

        Ok(self.epochs[index])
    }

    fn state_at_idx(&self, index: usize) -> Result<DVector<f64>, BraheError> {
        if index >= self.states.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            )));
        }

        Ok(self.states[index].clone())
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
        if let Some(ref mut covs) = self.covariances {
            covs.clear();
        }
        if let Some(ref mut stms) = self.stms {
            stms.clear();
        }
        if let Some(ref mut sens) = self.sensitivities {
            sens.clear();
        }
    }

    fn remove_epoch(&mut self, epoch: &Epoch) -> Result<DVector<f64>, BraheError> {
        if let Some(index) = self.epochs.iter().position(|e| e == epoch) {
            let removed_state = self.states.remove(index);
            self.epochs.remove(index);
            if let Some(ref mut covs) = self.covariances {
                covs.remove(index);
            }
            if let Some(ref mut stms) = self.stms {
                stms.remove(index);
            }
            if let Some(ref mut sens) = self.sensitivities {
                sens.remove(index);
            }
            Ok(removed_state)
        } else {
            Err(BraheError::Error(
                "Epoch not found in trajectory".to_string(),
            ))
        }
    }

    fn remove(&mut self, index: usize) -> Result<(Epoch, DVector<f64>), BraheError> {
        if index >= self.states.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            )));
        }

        let removed_epoch = self.epochs.remove(index);
        let removed_state = self.states.remove(index);
        if let Some(ref mut covs) = self.covariances {
            covs.remove(index);
        }
        if let Some(ref mut stms) = self.stms {
            stms.remove(index);
        }
        if let Some(ref mut sens) = self.sensitivities {
            sens.remove(index);
        }
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
            return Err(BraheError::Error("Maximum size must be >= 1".to_string()));
        }
        self.eviction_policy = TrajectoryEvictionPolicy::KeepCount;
        self.max_size = Some(max_size);
        self.max_age = None;
        self.apply_eviction_policy();
        Ok(())
    }

    fn set_eviction_policy_max_age(&mut self, max_age: f64) -> Result<(), BraheError> {
        if max_age <= 0.0 {
            return Err(BraheError::Error("Maximum age must be > 0.0".to_string()));
        }
        self.eviction_policy = TrajectoryEvictionPolicy::KeepWithinDuration;
        self.max_age = Some(max_age);
        self.max_size = None;
        self.apply_eviction_policy();
        Ok(())
    }

    fn get_eviction_policy(&self) -> TrajectoryEvictionPolicy {
        self.eviction_policy
    }
}

impl InterpolationConfig for DTrajectory {
    fn with_interpolation_method(mut self, method: InterpolationMethod) -> Self {
        self.interpolation_method = method;
        self
    }

    fn set_interpolation_method(&mut self, method: InterpolationMethod) {
        self.interpolation_method = method;
    }

    fn get_interpolation_method(&self) -> InterpolationMethod {
        self.interpolation_method
    }
}

impl CovarianceInterpolationConfig for DTrajectory {
    fn with_covariance_interpolation_method(
        mut self,
        method: CovarianceInterpolationMethod,
    ) -> Self {
        self.covariance_interpolation_method = method;
        self
    }

    fn set_covariance_interpolation_method(&mut self, method: CovarianceInterpolationMethod) {
        self.covariance_interpolation_method = method;
    }

    fn get_covariance_interpolation_method(&self) -> CovarianceInterpolationMethod {
        self.covariance_interpolation_method
    }
}

impl STMStorage for DTrajectory {
    fn enable_stm_storage(&mut self) {
        if self.stms.is_none() {
            let identity = DMatrix::identity(self.dimension, self.dimension);
            self.stms = Some(vec![identity; self.states.len()]);
        }
    }

    fn stm_at_idx(&self, index: usize) -> Option<&DMatrix<f64>> {
        self.stms.as_ref()?.get(index)
    }

    fn set_stm_at(&mut self, index: usize, stm: DMatrix<f64>) {
        if index >= self.states.len() {
            panic!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            );
        }
        if stm.nrows() != self.dimension || stm.ncols() != self.dimension {
            panic!(
                "STM dimensions {}×{} do not match expected {}×{}",
                stm.nrows(),
                stm.ncols(),
                self.dimension,
                self.dimension
            );
        }

        if self.stms.is_none() {
            self.enable_stm_storage();
        }

        if let Some(ref mut stms) = self.stms {
            stms[index] = stm;
        }
    }

    fn stm_dimensions(&self) -> (usize, usize) {
        (self.dimension, self.dimension)
    }

    fn stm_storage(&self) -> Option<&Vec<DMatrix<f64>>> {
        self.stms.as_ref()
    }

    fn stm_storage_mut(&mut self) -> Option<&mut Vec<DMatrix<f64>>> {
        self.stms.as_mut()
    }

    // stm_at() uses default trait implementation
}

impl SensitivityStorage for DTrajectory {
    fn enable_sensitivity_storage(&mut self, param_dim: usize) {
        if param_dim == 0 {
            panic!("Parameter dimension must be > 0");
        }
        if self.sensitivities.is_none() {
            let zero_sens = DMatrix::zeros(self.dimension, param_dim);
            self.sensitivities = Some(vec![zero_sens; self.states.len()]);
            self.sensitivity_dimension = Some((self.dimension, param_dim));
        }
    }

    fn sensitivity_at_idx(&self, index: usize) -> Option<&DMatrix<f64>> {
        self.sensitivities.as_ref()?.get(index)
    }

    fn set_sensitivity_at(&mut self, index: usize, sensitivity: DMatrix<f64>) {
        if index >= self.states.len() {
            panic!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            );
        }
        if sensitivity.nrows() != self.dimension {
            panic!(
                "Sensitivity row count {} does not match state dimension {}",
                sensitivity.nrows(),
                self.dimension
            );
        }

        // Check consistency with existing dimension
        if let Some((_, existing_cols)) = self.sensitivity_dimension
            && sensitivity.ncols() != existing_cols
        {
            panic!(
                "Sensitivity column count {} does not match existing {}",
                sensitivity.ncols(),
                existing_cols
            );
        }

        if self.sensitivities.is_none() {
            self.enable_sensitivity_storage(sensitivity.ncols());
        }

        if let Some(ref mut sens) = self.sensitivities {
            sens[index] = sensitivity;
        }
    }

    fn sensitivity_dimensions(&self) -> Option<(usize, usize)> {
        self.sensitivity_dimension
    }

    fn sensitivity_storage(&self) -> Option<&Vec<DMatrix<f64>>> {
        self.sensitivities.as_ref()
    }

    fn sensitivity_storage_mut(&mut self) -> Option<&mut Vec<DMatrix<f64>>> {
        self.sensitivities.as_mut()
    }

    // sensitivity_at() uses default trait implementation
}

impl InterpolatableTrajectory for DTrajectory {
    /// Interpolate state at a given epoch using the configured interpolation method.
    ///
    /// Overrides the default trait implementation to provide proper support for
    /// Lagrange interpolation. Hermite methods are not supported for generic DTrajectory
    /// as they require 6D orbital states with position/velocity structure.
    ///
    /// # Arguments
    /// * `epoch` - Target epoch for interpolation
    ///
    /// # Returns
    /// * `Ok(state)` - Interpolated state vector
    /// * `Err(BraheError)` - If interpolation fails or epoch is out of range
    ///
    /// # Panics
    /// - HermiteCubic/HermiteQuintic panic as they require 6D orbital states
    fn interpolate(&self, epoch: &Epoch) -> Result<DVector<f64>, BraheError> {
        // Bounds checking
        if let Some(start) = self.start_epoch()
            && *epoch < start
        {
            return Err(BraheError::OutOfBoundsError(format!(
                "Cannot interpolate: epoch {} is before trajectory start {}",
                epoch, start
            )));
        }

        if let Some(end) = self.end_epoch()
            && *epoch > end
        {
            return Err(BraheError::OutOfBoundsError(format!(
                "Cannot interpolate: epoch {} is after trajectory end {}",
                epoch, end
            )));
        }

        // Get indices before and after the target epoch
        let idx1 = self.index_before_epoch(epoch)?;
        let idx2 = self.index_after_epoch(epoch)?;

        // If indices are the same, we have an exact match
        if idx1 == idx2 {
            return self.state_at_idx(idx1);
        }

        // Validate minimum point count
        let method = self.get_interpolation_method();
        let required = method.min_points_required();
        if self.len() < required {
            return Err(BraheError::Error(format!(
                "{:?} requires {} points, trajectory has {}",
                method,
                required,
                self.len()
            )));
        }

        // Get reference epoch for time calculations
        let ref_epoch = self.start_epoch().unwrap();

        match method {
            InterpolationMethod::Linear => self.interpolate_linear(epoch),

            InterpolationMethod::Lagrange { degree } => {
                // Collect degree+1 points centered around query epoch
                let n_points = degree + 1;
                let (start_idx, end_idx) =
                    compute_lagrange_window(self.len(), idx1, idx2, n_points)?;

                // Build time and value arrays
                let times: Vec<f64> = (start_idx..=end_idx)
                    .map(|i| self.epochs[i] - ref_epoch)
                    .collect();
                let values: Vec<DVector<f64>> = (start_idx..=end_idx)
                    .map(|i| self.states[i].clone())
                    .collect();

                let t = *epoch - ref_epoch;
                Ok(interpolate_lagrange_dvector(&times, &values, t))
            }

            InterpolationMethod::HermiteCubic | InterpolationMethod::HermiteQuintic => {
                Err(BraheError::Error(format!(
                    "{:?} interpolation requires 6D orbital states with position/velocity \
                     structure. Use DOrbitTrajectory for orbital states with Hermite methods, \
                     or use Linear/Lagrange interpolation for generic N-dimensional systems.",
                    self.interpolation_method
                )))
            }
        }
    }
}

/// Helper function to compute the window of indices for Lagrange interpolation.
fn compute_lagrange_window(
    len: usize,
    idx1: usize,
    idx2: usize,
    n_points: usize,
) -> Result<(usize, usize), BraheError> {
    if len < n_points {
        return Err(BraheError::Error(format!(
            "Need {} points for interpolation, trajectory has {}",
            n_points, len
        )));
    }

    let center = (idx1 + idx2) / 2;
    let half_window = n_points / 2;
    let mut start_idx = center.saturating_sub(half_window);
    let mut end_idx = start_idx + n_points - 1;

    if end_idx >= len {
        end_idx = len - 1;
        start_idx = end_idx.saturating_sub(n_points - 1);
    }

    Ok((start_idx, end_idx))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::time::{Epoch, TimeSystem};
    use approx::assert_abs_diff_eq;

    fn create_test_trajectory() -> DTrajectory {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
            Epoch::from_jd(2451545.2, TimeSystem::UTC),
        ];

        let states = vec![
            DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]),
            DVector::from_vec(vec![7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0]),
            DVector::from_vec(vec![7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0]),
        ];

        DTrajectory::from_data(epochs, states).unwrap()
    }

    // Trajectory Trait Tests

    #[test]
    fn test_dtrajectory_new_with_dimension() {
        // 3
        let traj = DTrajectory::new(3);
        assert_eq!(traj.dimension, 3);
        assert_eq!(traj.len(), 0);
        assert!(traj.is_empty());

        // 6
        let traj = DTrajectory::new(6);
        assert_eq!(traj.dimension, 6);
        assert_eq!(traj.len(), 0);
        assert!(traj.is_empty());

        // 12
        let traj = DTrajectory::new(12);
        assert_eq!(traj.dimension, 12);
        assert_eq!(traj.len(), 0);
        assert!(traj.is_empty());
    }

    // Test panic on zero dimension
    #[test]
    #[should_panic(expected = "Trajectory dimension must be greater than 0")]
    fn test_dtrajectory_new_with_zero_dimension() {
        let _traj = DTrajectory::new(0);
    }

    #[test]
    fn test_dtrajectory_with_interpolation_method() {
        let traj = DTrajectory::new(12).with_interpolation_method(InterpolationMethod::Linear);
        assert_eq!(traj.dimension, 12);
        assert_eq!(traj.interpolation_method, InterpolationMethod::Linear);
    }

    #[test]
    fn test_dtrajectory_with_eviction_policy_max_size_builder() {
        // Test builder pattern for max size eviction policy
        let traj = DTrajectory::new(6).with_eviction_policy_max_size(5);

        assert_eq!(
            traj.get_eviction_policy(),
            TrajectoryEvictionPolicy::KeepCount
        );
        assert_eq!(traj.len(), 0);
    }

    #[test]
    fn test_dtrajectory_with_eviction_policy_max_age_builder() {
        // Test builder pattern for max age eviction policy
        let traj = DTrajectory::new(6).with_eviction_policy_max_age(300.0);

        assert_eq!(
            traj.get_eviction_policy(),
            TrajectoryEvictionPolicy::KeepWithinDuration
        );
        assert_eq!(traj.len(), 0);
    }

    #[test]
    fn test_dtrajectory_builder_pattern_chaining() {
        // Test chaining multiple builder methods
        let mut traj = DTrajectory::new(6)
            .with_interpolation_method(InterpolationMethod::Linear)
            .with_eviction_policy_max_size(10);

        assert_eq!(traj.get_interpolation_method(), InterpolationMethod::Linear);
        assert_eq!(
            traj.get_eviction_policy(),
            TrajectoryEvictionPolicy::KeepCount
        );

        // Add states and verify eviction policy works
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        for i in 0..15 {
            let epoch = t0 + (i as f64 * 60.0);
            let state =
                DVector::from_vec(vec![7000e3 + i as f64 * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
            traj.add(epoch, state);
        }

        // Should only have 10 states due to eviction policy
        assert_eq!(traj.len(), 10);
    }

    #[test]
    fn test_dtrajectory_dimension() {
        let traj = DTrajectory::new(9);
        assert_eq!(traj.dimension(), 9);

        let traj = DTrajectory::new(4);
        assert_eq!(traj.dimension(), 4);
    }

    #[test]
    fn test_dtrajectory_interpolatable_set_interpolation_method() {
        let mut traj = DTrajectory::new(6);
        assert_eq!(traj.interpolation_method, InterpolationMethod::Linear);

        traj.set_interpolation_method(InterpolationMethod::Linear);
        assert_eq!(traj.interpolation_method, InterpolationMethod::Linear);
    }

    #[test]
    fn test_dtrajectory_to_matrix() {
        let traj = create_test_trajectory();
        let matrix = traj.to_matrix().unwrap();

        // Matrix should be 3 rows (time points) x 6 columns (state elements)
        assert_eq!(matrix.nrows(), 3);
        assert_eq!(matrix.ncols(), 6);

        // Test first row (first state at t0)
        assert_abs_diff_eq!(matrix[(0, 0)], 7000e3, epsilon = 1.0);
        assert_abs_diff_eq!(matrix[(0, 1)], 0.0, epsilon = 1.0);
        assert_abs_diff_eq!(matrix[(0, 2)], 0.0, epsilon = 1.0);
        assert_abs_diff_eq!(matrix[(0, 3)], 0.0, epsilon = 1.0);
        assert_abs_diff_eq!(matrix[(0, 4)], 7.5e3, epsilon = 1.0);
        assert_abs_diff_eq!(matrix[(0, 5)], 0.0, epsilon = 1.0);

        // Test second row (second state at t1)
        assert_abs_diff_eq!(matrix[(1, 0)], 7100e3, epsilon = 1.0);
        assert_abs_diff_eq!(matrix[(1, 1)], 1000e3, epsilon = 1.0);

        // Test third row (third state at t2)
        assert_abs_diff_eq!(matrix[(2, 0)], 7200e3, epsilon = 1.0);
        assert_abs_diff_eq!(matrix[(2, 1)], 2000e3, epsilon = 1.0);
        assert_abs_diff_eq!(matrix[(2, 2)], 1000e3, epsilon = 1.0);
        assert_abs_diff_eq!(matrix[(2, 3)], 200.0, epsilon = 1.0);
        assert_abs_diff_eq!(matrix[(2, 4)], 7.7e3, epsilon = 1.0);
        assert_abs_diff_eq!(matrix[(2, 5)], 100.0, epsilon = 1.0);

        // Test first column (first element of each state over time)
        assert_abs_diff_eq!(matrix[(0, 0)], 7000e3, epsilon = 1.0);
        assert_abs_diff_eq!(matrix[(1, 0)], 7100e3, epsilon = 1.0);
        assert_abs_diff_eq!(matrix[(2, 0)], 7200e3, epsilon = 1.0);
    }

    #[test]
    fn test_dtrajectory_trajectory_get_eviction_policy() {
        let mut traj = DTrajectory::new(6);

        // Default is None
        assert_eq!(traj.get_eviction_policy(), TrajectoryEvictionPolicy::None);

        // Set to KeepCount
        traj.set_eviction_policy_max_size(10).unwrap();
        assert_eq!(
            traj.get_eviction_policy(),
            TrajectoryEvictionPolicy::KeepCount
        );

        // Set to KeepWithinDuration
        traj.set_eviction_policy_max_age(100.0).unwrap();
        assert_eq!(
            traj.get_eviction_policy(),
            TrajectoryEvictionPolicy::KeepWithinDuration
        );
    }

    #[test]
    fn test_dtrajectory_apply_eviction_policy_keep_count() {
        let mut traj = DTrajectory::new(6).with_eviction_policy_max_size(3);

        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        for i in 0..5 {
            let epoch = t0 + (i as f64 * 60.0);
            let state =
                DVector::from_vec(vec![7000e3 + i as f64 * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
            traj.add(epoch, state);
        }

        // Should only have 3 states due to eviction policy
        assert_eq!(traj.len(), 3);
        assert_eq!(traj.epochs[0], t0 + 2.0 * 60.0); // First state should be the third added
    }

    #[test]
    fn test_dtrajectory_apply_eviction_policy_keep_within_duration() {
        let mut traj = DTrajectory::new(6).with_eviction_policy_max_age(86400.0 * 7.0 - 1.0); // 7 days

        let t0 = Epoch::from_datetime(2023, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        for i in 0..10 {
            let epoch = t0 + (i as f64 * 86400.0); // 1 day apart
            let state =
                DVector::from_vec(vec![7000e3 + i as f64 * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
            traj.add(epoch, state);
        }

        // Should only have 7 states due to eviction policy
        assert_eq!(traj.len(), 7);
        assert_eq!(traj.epochs[0], t0 + 3.0 * 86400.0); // First state should be the fourth added

        // Repeat with an exact 7 days limit
        let mut traj = DTrajectory::new(6).with_eviction_policy_max_age(86400.0 * 7.0); // 7 days
        for i in 0..10 {
            let epoch = t0 + (i as f64 * 86400.0);
            let state =
                DVector::from_vec(vec![7000e3 + i as f64 * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
            traj.add(epoch, state);
        }

        // Should still have 8 states due to exact 7 days limit
        assert_eq!(traj.len(), 8);
        assert_eq!(traj.epochs[0], t0 + 2.0 * 86400.0); // First state should be the third added
    }

    // Default Trait Tests

    #[test]
    fn test_dtrajectory_default() {
        let traj = DTrajectory::default();
        assert_eq!(traj.dimension, 6);
        assert_eq!(traj.len(), 0);
        assert!(traj.is_empty());
        assert_eq!(traj.interpolation_method, InterpolationMethod::Linear);
        assert_eq!(traj.eviction_policy, TrajectoryEvictionPolicy::None);
    }

    // Index Trait Tests
    #[test]
    fn test_dtrajectory_index() {
        let traj = create_test_trajectory();
        let state = &traj[0];

        assert_eq!(state.len(), 6);
        assert_eq!(state[0], 7000e3);
        assert_eq!(state[1], 0.0);
        assert_eq!(state[2], 0.0);
        assert_eq!(state[3], 0.0);
        assert_eq!(state[4], 7.5e3);
        assert_eq!(state[5], 0.0);

        let state = &traj[1];
        assert_eq!(state[0], 7100e3);
        assert_eq!(state[1], 1000e3);
        assert_eq!(state[2], 500e3);
        assert_eq!(state[3], 100.0);
        assert_eq!(state[4], 7.6e3);
        assert_eq!(state[5], 50.0);

        let state = &traj[2];
        assert_eq!(state[0], 7200e3);
        assert_eq!(state[1], 2000e3);
        assert_eq!(state[2], 1000e3);
        assert_eq!(state[3], 200.0);
        assert_eq!(state[4], 7.7e3);
        assert_eq!(state[5], 100.0);
    }

    #[test]
    #[should_panic]
    fn test_dtrajectory_index_index_out_of_bounds() {
        let traj = create_test_trajectory();
        let _ = &traj[10]; // Should panic
    }

    // Iterator Trait Tests

    #[test]
    fn test_dtrajectory_iterator_iterator_len() {
        let traj = create_test_trajectory();

        let iter = traj.into_iter();
        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn test_dtrajectory_iterator_iterator_size_hint() {
        let traj = create_test_trajectory();

        let iter = traj.into_iter();
        let (lower, upper) = iter.size_hint();
        assert_eq!(lower, 3);
        assert_eq!(upper, Some(3));
    }

    // ExactSizeIterator Trait Tests

    #[test]
    fn test_dtrajectory_exactsizeiterator_len() {
        let traj = create_test_trajectory();
        let iter = traj.into_iter();
        assert_eq!(iter.len(), 3);
    }

    // IntoIterator Trait Tests

    #[test]
    fn test_dtrajectory_intoiterator_into_iter() {
        let traj = create_test_trajectory();

        let mut count = 0;
        for (epoch, state) in &traj {
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
    fn test_dtrajectory_intoiterator_into_iter_empty() {
        let traj = DTrajectory::new(6);

        let mut count = 0;
        for _ in &traj {
            count += 1;
        }
        assert_eq!(count, 0);
    }

    // Trajectory Trait Tests

    #[test]
    fn test_dtrajectory_from_data() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
        ];
        let states = vec![
            DVector::from_vec(vec![1.0, 2.0, 3.0]),
            DVector::from_vec(vec![4.0, 5.0, 6.0]),
        ];

        let traj = DTrajectory::from_data(epochs, states).unwrap();
        assert_eq!(traj.dimension, 3);
        assert_eq!(traj.len(), 2);
    }

    #[test]
    fn test_dtrajectory_from_data_errors() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
        ];
        let states = vec![DVector::from_vec(vec![1.0, 2.0, 3.0])];

        let result = DTrajectory::from_data(epochs.clone(), states);
        assert!(result.is_err());

        let empty_epochs: Vec<Epoch> = vec![];
        let empty_states: Vec<DVector<f64>> = vec![];
        let result = DTrajectory::from_data(empty_epochs, empty_states);
        assert!(result.is_err());
    }

    #[test]
    fn test_dtrajectory_trajectory_add() {
        let mut trajectory = DTrajectory::new(6);

        let epoch1 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state1 = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);

        trajectory.add(epoch1, state1.clone());
        assert_eq!(trajectory.len(), 1);

        let epoch2 = Epoch::from_datetime(2023, 1, 1, 13, 0, 0.0, 0.0, TimeSystem::UTC);
        let state2 = DVector::from_vec(vec![7100e3, 100e3, 50e3, 10.0, 7.6e3, 5.0]);

        trajectory.add(epoch2, state2.clone());
        assert_eq!(trajectory.len(), 2);

        assert_eq!(trajectory.states[0], state1);
        assert_eq!(trajectory.states[1], state2);
    }

    #[test]
    fn test_dtrajectory_trajectory_add_out_of_order() {
        let mut trajectory = DTrajectory::new(6);
        let epoch1 = Epoch::from_datetime(2023, 1, 1, 13, 0, 0.0, 0.0, TimeSystem::UTC);
        let state1 = DVector::from_vec(vec![7100e3, 100e3, 60e3, 10.0, 7.6e3, 5.0]);

        trajectory.add(epoch1, state1.clone());
        assert_eq!(trajectory.len(), 1);
        assert_eq!(trajectory.epochs[0], epoch1);
        assert_eq!(trajectory.states[0], state1);

        let epoch2 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state2 = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        trajectory.add(epoch2, state2.clone());
        assert_eq!(trajectory.len(), 2);
        assert_eq!(trajectory.epochs[0], epoch2);
        assert_eq!(trajectory.states[0], state2);
        assert_eq!(trajectory.epochs[1], epoch1);
        assert_eq!(trajectory.states[1], state1);
    }

    #[test]
    #[should_panic(expected = "State vector dimension does not match trajectory dimension")]
    fn test_dtrajectory_trajectory_add_dimension_mismatch() {
        let mut trajectory = DTrajectory::new(6);
        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0]); // Dimension 3 instead of 6

        trajectory.add(epoch, state);
    }

    #[test]
    fn test_dtrajectory_trajectory_add_append() {
        let mut trajectory = DTrajectory::new(6);
        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state1 = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);

        trajectory.add(epoch, state1.clone());
        assert_eq!(trajectory.len(), 1);
        assert_eq!(trajectory.states[0], state1);

        let state2 = DVector::from_vec(vec![7100e3, 100e3, 50e3, 10.0, 7.6e3, 5.0]);
        trajectory.add(epoch, state2.clone());
        assert_eq!(trajectory.len(), 2); // Length should increase (append, not replace)
        assert_eq!(trajectory.states[0], state1); // First state unchanged
        assert_eq!(trajectory.states[1], state2); // Second state appended
    }

    #[test]
    fn test_dtrajectory_trajectory_epoch() {
        let traj = create_test_trajectory();

        let epoch = traj.epoch_at_idx(0).unwrap();
        assert_eq!(epoch, Epoch::from_jd(2451545.0, TimeSystem::UTC));

        let epoch = traj.epoch_at_idx(1).unwrap();
        assert_eq!(epoch, Epoch::from_jd(2451545.1, TimeSystem::UTC));
    }

    #[test]
    fn test_dtrajectory_trajectory_state() {
        let traj = create_test_trajectory();

        let state = traj.state_at_idx(0).unwrap();
        assert_abs_diff_eq!(state[0], 7000e3, epsilon = 1.0);

        let state = traj.state_at_idx(1).unwrap();
        assert_abs_diff_eq!(state[0], 7100e3, epsilon = 1.0);
    }

    #[test]
    fn test_dtrajectory_trajectory_nearest_state() {
        let traj = create_test_trajectory();

        // Halfway between first and second
        let epoch = Epoch::from_jd(2451545.05, TimeSystem::UTC);
        let (nearest_epoch, _) = traj.nearest_state(&epoch).unwrap();
        assert_eq!(nearest_epoch, Epoch::from_jd(2451545.0, TimeSystem::UTC));

        // Slightly before the second
        let epoch = Epoch::from_jd(2451545.09, TimeSystem::UTC);
        let (nearest_epoch, _) = traj.nearest_state(&epoch).unwrap();
        assert_eq!(nearest_epoch, Epoch::from_jd(2451545.1, TimeSystem::UTC));

        // Slightly after the second
        let epoch = Epoch::from_jd(2451545.11, TimeSystem::UTC);
        let (nearest_epoch, _) = traj.nearest_state(&epoch).unwrap();
        assert_eq!(nearest_epoch, Epoch::from_jd(2451545.1, TimeSystem::UTC));

        // Exactly at the third
        let epoch = Epoch::from_jd(2451545.2, TimeSystem::UTC);
        let (nearest_epoch, _) = traj.nearest_state(&epoch).unwrap();
        assert_eq!(nearest_epoch, Epoch::from_jd(2451545.2, TimeSystem::UTC));
    }

    #[test]
    fn test_dtrajectory_trajectory_len() {
        let traj = create_test_trajectory();
        assert_eq!(traj.len(), 3);

        let empty_traj = DTrajectory::new(6);
        assert_eq!(empty_traj.len(), 0);
    }

    #[test]
    fn test_dtrajectory_trajectory_is_empty() {
        let traj = create_test_trajectory();
        assert!(!traj.is_empty());

        let empty_traj = DTrajectory::new(6);
        assert!(empty_traj.is_empty());
    }

    #[test]
    fn test_dtrajectory_trajectory_start_epoch() {
        let traj = create_test_trajectory();
        let start = traj.start_epoch().unwrap();
        assert_eq!(start, Epoch::from_jd(2451545.0, TimeSystem::UTC));

        let empty_traj = DTrajectory::new(6);
        assert!(empty_traj.start_epoch().is_none());
    }

    #[test]
    fn test_dtrajectory_trajectory_end_epoch() {
        let traj = create_test_trajectory();
        let end = traj.end_epoch().unwrap();
        assert_eq!(end, Epoch::from_jd(2451545.2, TimeSystem::UTC));

        let empty_traj = DTrajectory::new(6);
        assert!(empty_traj.end_epoch().is_none());
    }

    #[test]
    fn test_dtrajectory_trajectory_timespan() {
        let traj = create_test_trajectory();
        let timespan = traj.timespan().unwrap();
        assert_abs_diff_eq!(timespan, 0.2 * 86400.0, epsilon = 1.0);

        let empty_traj = DTrajectory::new(6);
        assert!(empty_traj.timespan().is_none());
    }

    #[test]
    fn test_dtrajectory_trajectory_first() {
        let traj = create_test_trajectory();
        let (epoch, state) = traj.first().unwrap();
        assert_eq!(epoch, Epoch::from_jd(2451545.0, TimeSystem::UTC));
        assert_abs_diff_eq!(state[0], 7000e3, epsilon = 1.0);

        let empty_traj = DTrajectory::new(6);
        assert!(empty_traj.first().is_none());
    }

    #[test]
    fn test_dtrajectory_trajectory_last() {
        let traj = create_test_trajectory();
        let (epoch, state) = traj.last().unwrap();
        assert_eq!(epoch, Epoch::from_jd(2451545.2, TimeSystem::UTC));
        assert_abs_diff_eq!(state[0], 7200e3, epsilon = 1.0);

        let empty_traj = DTrajectory::new(6);
        assert!(empty_traj.last().is_none());
    }

    #[test]
    fn test_dtrajectory_trajectory_clear() {
        let mut traj = create_test_trajectory();
        assert_eq!(traj.len(), 3);

        traj.clear();
        assert_eq!(traj.len(), 0);
        assert!(traj.is_empty());
    }

    #[test]
    fn test_dtrajectory_trajectory_remove_epoch() {
        let mut traj = create_test_trajectory();
        let epoch = Epoch::from_jd(2451545.1, TimeSystem::UTC);

        let removed_state = traj.remove_epoch(&epoch).unwrap();
        assert_abs_diff_eq!(removed_state[0], 7100e3, epsilon = 1.0);
        assert_eq!(traj.len(), 2);
    }

    #[test]
    fn test_dtrajectory_trajectory_remove() {
        let mut traj = create_test_trajectory();

        let (removed_epoch, removed_state) = traj.remove(1).unwrap();
        assert_eq!(removed_epoch, Epoch::from_jd(2451545.1, TimeSystem::UTC));
        assert_abs_diff_eq!(removed_state[0], 7100e3, epsilon = 1.0);
        assert_eq!(traj.len(), 2);
    }

    #[test]
    fn test_dtrajectory_trajectory_remove_out_of_bounds() {
        let mut traj = create_test_trajectory();

        let result = traj.remove(10);
        assert!(result.is_err());
    }

    #[test]
    fn test_dtrajectory_trajectory_get() {
        let traj = create_test_trajectory();

        let (epoch, state) = traj.get(1).unwrap();
        assert_eq!(epoch, Epoch::from_jd(2451545.1, TimeSystem::UTC));
        assert_abs_diff_eq!(state[0], 7100e3, epsilon = 1.0);
    }

    #[test]
    fn test_dtrajectory_trajectory_index_before_epoch() {
        // Create a 6-dimensional DTrajectory with states at epochs: t0, t0+60s, t0+120s
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            DVector::from_vec(vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
            DVector::from_vec(vec![21.0, 22.0, 23.0, 24.0, 25.0, 26.0]),
        ];

        let traj = DTrajectory::from_data(epochs, states).unwrap();

        // Test finding index before t0 (should error - before all states)
        let before_t0 = t0 - 10.0;
        assert!(traj.index_before_epoch(&before_t0).is_err());

        // Test finding index before t0+30s (should return index 0)
        let t0_plus_30 = t0 + 30.0;
        assert_eq!(traj.index_before_epoch(&t0_plus_30).unwrap(), 0);

        // Test finding index before t0+60s (should return index 1 - exact match)
        assert_eq!(traj.index_before_epoch(&t1).unwrap(), 1);

        // Test finding index before t0+90s (should return index 1)
        let t0_plus_90 = t0 + 90.0;
        assert_eq!(traj.index_before_epoch(&t0_plus_90).unwrap(), 1);

        // Test finding index before t0+120s (should return index 2 - exact match)
        assert_eq!(traj.index_before_epoch(&t2).unwrap(), 2);

        // Test finding index before t0+150s (should return index 2)
        let t0_plus_150 = t0 + 150.0;
        assert_eq!(traj.index_before_epoch(&t0_plus_150).unwrap(), 2);
    }

    #[test]
    fn test_dtrajectory_trajectory_index_after_epoch() {
        // Create a 6-dimensional DTrajectory with states at epochs: t0, t0+60s, t0+120s
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            DVector::from_vec(vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
            DVector::from_vec(vec![21.0, 22.0, 23.0, 24.0, 25.0, 26.0]),
        ];

        let traj = DTrajectory::from_data(epochs, states).unwrap();

        // Test finding index after t0-30s (should return index 0)
        let t0_minus_30 = t0 - 30.0;
        assert_eq!(traj.index_after_epoch(&t0_minus_30).unwrap(), 0);

        // Test finding index after t0 (should return index 0 - exact match)
        assert_eq!(traj.index_after_epoch(&t0).unwrap(), 0);

        // Test finding index after t0+30s (should return index 1)
        let t0_plus_30 = t0 + 30.0;
        assert_eq!(traj.index_after_epoch(&t0_plus_30).unwrap(), 1);

        // Test finding index after t0+60s (should return index 1 - exact match)
        assert_eq!(traj.index_after_epoch(&t1).unwrap(), 1);

        // Test finding index after t0+90s (should return index 2)
        let t0_plus_90 = t0 + 90.0;
        assert_eq!(traj.index_after_epoch(&t0_plus_90).unwrap(), 2);

        // Test finding index after t0+120s (should return index 2 - exact match)
        assert_eq!(traj.index_after_epoch(&t2).unwrap(), 2);

        // Test finding index after t0+150s (should error - after all states)
        let t0_plus_150 = t0 + 150.0;
        assert!(traj.index_after_epoch(&t0_plus_150).is_err());
    }

    #[test]
    fn test_dtrajectory_trajectory_state_before_epoch() {
        // Create a DTrajectory with distinguishable states at 3 epochs
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            DVector::from_vec(vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
            DVector::from_vec(vec![21.0, 22.0, 23.0, 24.0, 25.0, 26.0]),
        ];

        let traj = DTrajectory::from_data(epochs, states).unwrap();

        // Test that state_before_epoch returns correct (epoch, state) tuples
        let t0_plus_30 = t0 + 30.0;
        let (epoch, state) = traj.state_before_epoch(&t0_plus_30).unwrap();
        assert_eq!(epoch, t0);
        assert_eq!(state[0], 1.0);

        let t0_plus_90 = t0 + 90.0;
        let (epoch, state) = traj.state_before_epoch(&t0_plus_90).unwrap();
        assert_eq!(epoch, t1);
        assert_eq!(state[0], 11.0);

        // Test error case for epoch before all states
        let before_t0 = t0 - 10.0;
        assert!(traj.state_before_epoch(&before_t0).is_err());

        // Test that exact matches return the correct state
        let (epoch, state) = traj.state_before_epoch(&t1).unwrap();
        assert_eq!(epoch, t1);
        assert_eq!(state[0], 11.0);
    }

    #[test]
    fn test_dtrajectory_trajectory_state_after_epoch() {
        // Create a DTrajectory with distinguishable states at 3 epochs
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            DVector::from_vec(vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
            DVector::from_vec(vec![21.0, 22.0, 23.0, 24.0, 25.0, 26.0]),
        ];

        let traj = DTrajectory::from_data(epochs, states).unwrap();

        // Test that state_after_epoch returns correct (epoch, state) tuples
        let t0_plus_30 = t0 + 30.0;
        let (epoch, state) = traj.state_after_epoch(&t0_plus_30).unwrap();
        assert_eq!(epoch, t1);
        assert_eq!(state[0], 11.0);

        let t0_plus_90 = t0 + 90.0;
        let (epoch, state) = traj.state_after_epoch(&t0_plus_90).unwrap();
        assert_eq!(epoch, t2);
        assert_eq!(state[0], 21.0);

        // Test error case for epoch after all states
        let after_t2 = t2 + 10.0;
        assert!(traj.state_after_epoch(&after_t2).is_err());

        // Verify that exact matches return the correct state
        let (epoch, state) = traj.state_after_epoch(&t1).unwrap();
        assert_eq!(epoch, t1);
        assert_eq!(state[0], 11.0);
    }

    #[test]
    fn test_dtrajectory_set_eviction_policy_max_size() {
        let mut traj = create_test_trajectory();
        assert_eq!(traj.len(), 3);

        let _ = traj.set_eviction_policy_max_size(2);
        assert_eq!(traj.len(), 2);
        assert_eq!(traj.eviction_policy, TrajectoryEvictionPolicy::KeepCount);
    }

    #[test]
    fn test_dtrajectory_set_eviction_policy_max_age() {
        let mut traj = create_test_trajectory();

        // Max age slightly larger than 0.1 days
        let _ = traj.set_eviction_policy_max_age(0.11 * 86400.0);
        assert_eq!(traj.len(), 2);
        assert_eq!(
            traj.eviction_policy,
            TrajectoryEvictionPolicy::KeepWithinDuration
        );
    }

    // Interpolatable Trait Tests

    #[test]
    fn test_dtrajectory_interpolatable_get_interpolation_method() {
        // Create a trajectory with default Linear interpolation
        let mut traj = DTrajectory::new(6);

        // Test that get_interpolation_method returns Linear
        assert_eq!(traj.get_interpolation_method(), InterpolationMethod::Linear);

        // Set it to different methods and verify get_interpolation_method returns the correct value

        traj.set_interpolation_method(InterpolationMethod::Linear);
        assert_eq!(traj.get_interpolation_method(), InterpolationMethod::Linear);
    }

    #[test]
    fn test_dtrajectory_interpolatable_interpolate_linear() {
        // Create a 6-dimensional trajectory with 3 states at t0, t0+60s, t0+120s
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            DVector::from_vec(vec![60.0, 120.0, 180.0, 240.0, 300.0, 360.0]),
            DVector::from_vec(vec![120.0, 240.0, 360.0, 480.0, 600.0, 720.0]),
        ];

        let traj = DTrajectory::from_data(epochs, states).unwrap();

        // Test interpolate_linear at midpoints and exact epochs
        let state_at_t0 = traj.interpolate_linear(&t0).unwrap();
        assert_abs_diff_eq!(state_at_t0[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_t0[1], 0.0, epsilon = 1e-10);

        let state_at_t1 = traj.interpolate_linear(&t1).unwrap();
        assert_abs_diff_eq!(state_at_t1[0], 60.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_t1[1], 120.0, epsilon = 1e-10);

        let state_at_t2 = traj.interpolate_linear(&t2).unwrap();
        assert_abs_diff_eq!(state_at_t2[0], 120.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_t2[1], 240.0, epsilon = 1e-10);

        // Test interpolation at midpoint between t0 and t1
        let t0_plus_30 = t0 + 30.0;
        let state_at_midpoint = traj.interpolate_linear(&t0_plus_30).unwrap();
        assert_abs_diff_eq!(state_at_midpoint[0], 30.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_midpoint[1], 60.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_midpoint[2], 90.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_midpoint[3], 120.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_midpoint[4], 150.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_midpoint[5], 180.0, epsilon = 1e-10);

        // Test interpolation at midpoint between t1 and t2
        let t1_plus_30 = t1 + 30.0;
        let state_at_midpoint2 = traj.interpolate_linear(&t1_plus_30).unwrap();
        assert_abs_diff_eq!(state_at_midpoint2[0], 90.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_midpoint2[1], 180.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_midpoint2[2], 270.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_midpoint2[3], 360.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_midpoint2[4], 450.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_midpoint2[5], 540.0, epsilon = 1e-10);

        // Test error case: interpolation outside bounds
        let before_t0 = t0 - 10.0;
        assert!(traj.interpolate_linear(&before_t0).is_err());
        let after_t2 = t2 + 10.0;
        assert!(traj.interpolate_linear(&after_t2).is_err());

        // Test edge case: single state trajectory
        let single_epoch = vec![t0];
        let single_state = vec![DVector::from_vec(vec![
            100.0, 200.0, 300.0, 400.0, 500.0, 600.0,
        ])];
        let single_traj = DTrajectory::from_data(single_epoch, single_state).unwrap();

        let state_single = single_traj.interpolate_linear(&t0).unwrap();
        assert_abs_diff_eq!(state_single[0], 100.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_single[1], 200.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_single[2], 300.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_single[3], 400.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_single[4], 500.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_single[5], 600.0, epsilon = 1e-10);

        // Test error case: interpolation on single state trajectory at different epoch
        let different_epoch = t0 + 10.0;
        assert!(single_traj.interpolate_linear(&different_epoch).is_err());

        // Test error case: interpolation on empty trajectory
        let empty_traj = DTrajectory::new(6);
        assert!(empty_traj.interpolate_linear(&t0).is_err());
    }

    #[test]
    fn test_dtrajectory_interpolatable_interpolate() {
        // Create a trajectory for testing
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            DVector::from_vec(vec![60.0, 120.0, 180.0, 240.0, 300.0, 360.0]),
            DVector::from_vec(vec![120.0, 240.0, 360.0, 480.0, 600.0, 720.0]),
        ];

        let traj = DTrajectory::from_data(epochs, states).unwrap();

        // Test that interpolate() with Linear method returns same result as interpolate_linear()
        let t0_plus_30 = t0 + 30.0;
        let state_interpolate = traj.interpolate(&t0_plus_30).unwrap();
        let state_interpolate_linear = traj.interpolate_linear(&t0_plus_30).unwrap();

        for i in 0..6 {
            assert_abs_diff_eq!(
                state_interpolate[i],
                state_interpolate_linear[i],
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_dtrajectory_interpolate_before_start() {
        // Create a trajectory for testing
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            DVector::from_vec(vec![60.0, 120.0, 180.0, 240.0, 300.0, 360.0]),
            DVector::from_vec(vec![120.0, 240.0, 360.0, 480.0, 600.0, 720.0]),
        ];

        let traj = DTrajectory::from_data(epochs, states).unwrap();

        // Test interpolation before trajectory start
        let before_start = t0 - 10.0;
        let result = traj.interpolate_linear(&before_start);
        assert!(result.is_err());
        match result {
            Err(BraheError::OutOfBoundsError(_)) => {} // Expected error type
            _ => panic!("Expected OutOfBoundsError for interpolation before start"),
        }

        // Also test with interpolate() method
        let result = traj.interpolate(&before_start);
        assert!(result.is_err());
        match result {
            Err(BraheError::OutOfBoundsError(_)) => {} // Expected error type
            _ => panic!("Expected OutOfBoundsError for interpolation before start"),
        }
    }

    #[test]
    fn test_dtrajectory_interpolate_after_end() {
        // Create a trajectory for testing
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            DVector::from_vec(vec![60.0, 120.0, 180.0, 240.0, 300.0, 360.0]),
            DVector::from_vec(vec![120.0, 240.0, 360.0, 480.0, 600.0, 720.0]),
        ];

        let traj = DTrajectory::from_data(epochs, states).unwrap();

        // Test interpolation after trajectory end
        let after_end = t0 + 130.0;
        let result = traj.interpolate_linear(&after_end);
        assert!(result.is_err());
        match result {
            Err(BraheError::OutOfBoundsError(_)) => {} // Expected error type
            _ => panic!("Expected OutOfBoundsError for interpolation after end"),
        }

        // Also test with interpolate() method
        let result = traj.interpolate(&after_end);
        assert!(result.is_err());
        match result {
            Err(BraheError::OutOfBoundsError(_)) => {} // Expected error type
            _ => panic!("Expected OutOfBoundsError for interpolation after end"),
        }
    }

    #[test]
    fn test_dtrajectory_covariance_interpolation_config() {
        // Test the CovarianceInterpolationConfig trait implementation

        // Test default is TwoWasserstein
        let traj = DTrajectory::new(6);
        assert_eq!(
            traj.get_covariance_interpolation_method(),
            CovarianceInterpolationMethod::TwoWasserstein
        );

        // Test with_covariance_interpolation_method builder
        let traj = DTrajectory::new(6)
            .with_covariance_interpolation_method(CovarianceInterpolationMethod::MatrixSquareRoot);
        assert_eq!(
            traj.get_covariance_interpolation_method(),
            CovarianceInterpolationMethod::MatrixSquareRoot
        );

        // Test set_covariance_interpolation_method
        let mut traj = DTrajectory::new(6);
        traj.set_covariance_interpolation_method(CovarianceInterpolationMethod::MatrixSquareRoot);
        assert_eq!(
            traj.get_covariance_interpolation_method(),
            CovarianceInterpolationMethod::MatrixSquareRoot
        );
        traj.set_covariance_interpolation_method(CovarianceInterpolationMethod::TwoWasserstein);
        assert_eq!(
            traj.get_covariance_interpolation_method(),
            CovarianceInterpolationMethod::TwoWasserstein
        );
    }

    #[test]
    fn test_dtrajectory_covariance_interpolation_methods() {
        // Test that covariance interpolation produces correct results

        let t0 = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;

        let state1 = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let state2 = DVector::from_vec(vec![7100e3, 0.0, 0.0, 0.0, 7.6e3, 0.0]);

        // Create diagonal covariance matrices
        let cov1 =
            DMatrix::from_diagonal(&DVector::from_vec(vec![100.0, 100.0, 100.0, 1.0, 1.0, 1.0]));
        let cov2 =
            DMatrix::from_diagonal(&DVector::from_vec(vec![200.0, 200.0, 200.0, 2.0, 2.0, 2.0]));

        // Create trajectory with covariances
        let mut traj = DTrajectory::new(6);
        traj.enable_covariance_storage();
        traj.add(t0, state1);
        traj.add(t1, state2);
        traj.set_covariance_at(0, cov1);
        traj.set_covariance_at(1, cov2);

        // Test matrix square root interpolation at midpoint
        traj.set_covariance_interpolation_method(CovarianceInterpolationMethod::MatrixSquareRoot);
        let t_mid = t0 + 30.0;
        let cov_sqrt = traj.covariance_at(t_mid).unwrap();

        // Check symmetry and positive semi-definiteness
        for i in 0..6 {
            assert!(cov_sqrt[(i, i)] > 0.0);
            for j in 0..6 {
                assert_abs_diff_eq!(cov_sqrt[(i, j)], cov_sqrt[(j, i)], epsilon = 1e-10);
            }
        }

        // Check values are between endpoints
        assert!(cov_sqrt[(0, 0)] > 100.0 && cov_sqrt[(0, 0)] < 200.0);

        // Test two-Wasserstein interpolation at midpoint
        traj.set_covariance_interpolation_method(CovarianceInterpolationMethod::TwoWasserstein);
        let cov_wasserstein = traj.covariance_at(t_mid).unwrap();

        // Check symmetry and positive semi-definiteness
        for i in 0..6 {
            assert!(cov_wasserstein[(i, i)] > 0.0);
            for j in 0..6 {
                assert_abs_diff_eq!(
                    cov_wasserstein[(i, j)],
                    cov_wasserstein[(j, i)],
                    epsilon = 1e-10
                );
            }
        }

        // Check values are between endpoints
        assert!(cov_wasserstein[(0, 0)] > 100.0 && cov_wasserstein[(0, 0)] < 200.0);

        // For diagonal matrices, both methods should give similar results
        assert_abs_diff_eq!(cov_sqrt[(0, 0)], cov_wasserstein[(0, 0)], epsilon = 1e-6);
    }

    #[test]
    fn test_dtrajectory_covariance_at_exact_epochs() {
        // Test that covariance_at returns exact values at data points

        let t0 = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;

        let state1 = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let state2 = DVector::from_vec(vec![7100e3, 0.0, 0.0, 0.0, 7.6e3, 0.0]);

        let cov1 = DMatrix::identity(6, 6) * 100.0;
        let cov2 = DMatrix::identity(6, 6) * 200.0;

        let mut traj = DTrajectory::new(6);
        traj.enable_covariance_storage();
        traj.add(t0, state1);
        traj.add(t1, state2);
        traj.set_covariance_at(0, cov1);
        traj.set_covariance_at(1, cov2);

        // At exact t0, should return cov1
        let result = traj.covariance_at(t0).unwrap();
        assert_abs_diff_eq!(result[(0, 0)], 100.0, epsilon = 1e-10);

        // At exact t1, should return cov2
        let result = traj.covariance_at(t1).unwrap();
        assert_abs_diff_eq!(result[(0, 0)], 200.0, epsilon = 1e-10);
    }

    // ============================================================================
    // STM Storage Trait Tests
    // ============================================================================

    #[test]
    fn test_dtrajectory_enable_stm_storage() {
        let mut traj = create_test_trajectory();
        assert!(traj.stms.is_none());

        traj.enable_stm_storage();

        // Should now have STM storage with identity matrices
        assert!(traj.stms.is_some());
        let stms = traj.stms.as_ref().unwrap();
        assert_eq!(stms.len(), 3);

        // Each STM should be identity
        for stm in stms {
            assert_eq!(stm.nrows(), 6);
            assert_eq!(stm.ncols(), 6);
            for i in 0..6 {
                for j in 0..6 {
                    if i == j {
                        assert_abs_diff_eq!(stm[(i, j)], 1.0, epsilon = 1e-10);
                    } else {
                        assert_abs_diff_eq!(stm[(i, j)], 0.0, epsilon = 1e-10);
                    }
                }
            }
        }
    }

    #[test]
    fn test_dtrajectory_enable_stm_storage_idempotent() {
        let mut traj = create_test_trajectory();

        traj.enable_stm_storage();

        // Modify one STM
        traj.set_stm_at(0, DMatrix::from_element(6, 6, 2.0));

        // Enable again should be idempotent (no change)
        traj.enable_stm_storage();

        // The modified STM should still be there
        let stm = traj.stm_at_idx(0).unwrap();
        assert_abs_diff_eq!(stm[(0, 0)], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dtrajectory_set_stm_at() {
        let mut traj = create_test_trajectory();
        traj.enable_stm_storage();

        let custom_stm = DMatrix::from_element(6, 6, 5.0);
        traj.set_stm_at(1, custom_stm.clone());

        let result = traj.stm_at_idx(1).unwrap();
        assert_abs_diff_eq!(result[(0, 0)], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[(3, 3)], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dtrajectory_set_stm_at_auto_enables() {
        let mut traj = create_test_trajectory();
        assert!(traj.stms.is_none());

        // Setting STM without enabling first should auto-enable
        let custom_stm = DMatrix::from_element(6, 6, 3.0);
        traj.set_stm_at(0, custom_stm);

        assert!(traj.stms.is_some());
        let stm = traj.stm_at_idx(0).unwrap();
        assert_abs_diff_eq!(stm[(0, 0)], 3.0, epsilon = 1e-10);

        // Other indices should be identity (auto-enabled)
        let stm1 = traj.stm_at_idx(1).unwrap();
        assert_abs_diff_eq!(stm1[(0, 0)], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(stm1[(0, 1)], 0.0, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "STM dimensions")]
    fn test_dtrajectory_set_stm_at_dimension_mismatch() {
        let mut traj = create_test_trajectory();
        traj.enable_stm_storage();

        // Wrong dimension STM (3x3 instead of 6x6)
        let wrong_stm = DMatrix::identity(3, 3);
        traj.set_stm_at(0, wrong_stm);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_dtrajectory_set_stm_at_out_of_bounds() {
        let mut traj = create_test_trajectory();
        traj.enable_stm_storage();

        let stm = DMatrix::identity(6, 6);
        traj.set_stm_at(10, stm); // Only 3 states, index 10 is invalid
    }

    #[test]
    fn test_dtrajectory_stm_at_idx() {
        let mut traj = create_test_trajectory();
        traj.enable_stm_storage();

        let custom_stm = DMatrix::from_fn(6, 6, |i, j| (i * 6 + j) as f64);
        traj.set_stm_at(2, custom_stm);

        let result = traj.stm_at_idx(2).unwrap();
        assert_abs_diff_eq!(result[(0, 0)], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[(0, 1)], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[(1, 0)], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[(5, 5)], 35.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dtrajectory_stm_at_idx_no_storage() {
        let traj = create_test_trajectory();
        // No STM storage enabled
        assert!(traj.stm_at_idx(0).is_none());
        assert!(traj.stm_at_idx(1).is_none());
    }

    #[test]
    fn test_dtrajectory_stm_at_interpolation() {
        let mut traj = create_test_trajectory();
        traj.enable_stm_storage();

        // Set STMs at indices 0 and 1
        let stm0 = DMatrix::from_element(6, 6, 10.0);
        let stm1 = DMatrix::from_element(6, 6, 20.0);
        traj.set_stm_at(0, stm0);
        traj.set_stm_at(1, stm1);

        // Interpolate at midpoint
        let t0 = traj.epochs[0];
        let t1 = traj.epochs[1];
        let mid = t0 + (t1 - t0) / 2.0;

        let result = traj.stm_at(mid).unwrap();
        // Linear interpolation should give 15.0 at midpoint
        assert_abs_diff_eq!(result[(0, 0)], 15.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[(3, 3)], 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dtrajectory_stm_dimensions() {
        let traj = DTrajectory::new(6);
        let dims = traj.stm_dimensions();
        assert_eq!(dims, (6, 6));

        let traj = DTrajectory::new(9);
        let dims = traj.stm_dimensions();
        assert_eq!(dims, (9, 9));
    }

    // ============================================================================
    // Sensitivity Storage Trait Tests
    // ============================================================================

    #[test]
    fn test_dtrajectory_enable_sensitivity_storage() {
        let mut traj = create_test_trajectory();
        assert!(traj.sensitivities.is_none());
        assert!(traj.sensitivity_dimension.is_none());

        traj.enable_sensitivity_storage(3); // 3 parameters

        // Should now have sensitivity storage with zero matrices
        assert!(traj.sensitivities.is_some());
        assert_eq!(traj.sensitivity_dimension, Some((6, 3)));

        let sensitivities = traj.sensitivities.as_ref().unwrap();
        assert_eq!(sensitivities.len(), 3);

        // Each sensitivity should be zero
        for sens in sensitivities {
            assert_eq!(sens.nrows(), 6);
            assert_eq!(sens.ncols(), 3);
            for i in 0..6 {
                for j in 0..3 {
                    assert_abs_diff_eq!(sens[(i, j)], 0.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "Parameter dimension must be > 0")]
    fn test_dtrajectory_enable_sensitivity_storage_zero_param() {
        let mut traj = create_test_trajectory();
        traj.enable_sensitivity_storage(0); // Should panic
    }

    #[test]
    fn test_dtrajectory_set_sensitivity_at() {
        let mut traj = create_test_trajectory();
        traj.enable_sensitivity_storage(2);

        let custom_sens = DMatrix::from_element(6, 2, 7.0);
        traj.set_sensitivity_at(1, custom_sens);

        let result = traj.sensitivity_at_idx(1).unwrap();
        assert_abs_diff_eq!(result[(0, 0)], 7.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[(5, 1)], 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dtrajectory_set_sensitivity_at_auto_enables() {
        let mut traj = create_test_trajectory();
        assert!(traj.sensitivities.is_none());

        // Setting sensitivity without enabling first should auto-enable
        let custom_sens = DMatrix::from_element(6, 4, 9.0);
        traj.set_sensitivity_at(0, custom_sens);

        assert!(traj.sensitivities.is_some());
        assert_eq!(traj.sensitivity_dimensions(), Some((6, 4)));

        let sens = traj.sensitivity_at_idx(0).unwrap();
        assert_abs_diff_eq!(sens[(0, 0)], 9.0, epsilon = 1e-10);

        // Other indices should be zero (auto-enabled)
        let sens1 = traj.sensitivity_at_idx(1).unwrap();
        assert_abs_diff_eq!(sens1[(0, 0)], 0.0, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "row count")]
    fn test_dtrajectory_set_sensitivity_at_row_mismatch() {
        let mut traj = create_test_trajectory();
        traj.enable_sensitivity_storage(2);

        // Wrong row count (3 instead of 6)
        let wrong_sens = DMatrix::from_element(3, 2, 1.0);
        traj.set_sensitivity_at(0, wrong_sens);
    }

    #[test]
    #[should_panic(expected = "column count")]
    fn test_dtrajectory_set_sensitivity_at_col_mismatch() {
        let mut traj = create_test_trajectory();
        traj.enable_sensitivity_storage(2);

        // Wrong column count (5 instead of 2)
        let wrong_sens = DMatrix::from_element(6, 5, 1.0);
        traj.set_sensitivity_at(0, wrong_sens);
    }

    #[test]
    fn test_dtrajectory_sensitivity_at_idx() {
        let mut traj = create_test_trajectory();
        traj.enable_sensitivity_storage(2);

        let custom_sens = DMatrix::from_fn(6, 2, |i, j| (i * 2 + j) as f64);
        traj.set_sensitivity_at(2, custom_sens);

        let result = traj.sensitivity_at_idx(2).unwrap();
        assert_abs_diff_eq!(result[(0, 0)], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[(0, 1)], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[(1, 0)], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[(5, 1)], 11.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dtrajectory_sensitivity_at_idx_no_storage() {
        let traj = create_test_trajectory();
        // No sensitivity storage enabled
        assert!(traj.sensitivity_at_idx(0).is_none());
        assert!(traj.sensitivity_at_idx(1).is_none());
    }

    #[test]
    fn test_dtrajectory_sensitivity_at_interpolation() {
        let mut traj = create_test_trajectory();
        traj.enable_sensitivity_storage(2);

        // Set sensitivities at indices 0 and 1
        let sens0 = DMatrix::from_element(6, 2, 100.0);
        let sens1 = DMatrix::from_element(6, 2, 200.0);
        traj.set_sensitivity_at(0, sens0);
        traj.set_sensitivity_at(1, sens1);

        // Interpolate at midpoint
        let t0 = traj.epochs[0];
        let t1 = traj.epochs[1];
        let mid = t0 + (t1 - t0) / 2.0;

        let result = traj.sensitivity_at(mid).unwrap();
        // Linear interpolation should give 150.0 at midpoint
        assert_abs_diff_eq!(result[(0, 0)], 150.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[(5, 1)], 150.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dtrajectory_sensitivity_dimensions() {
        let traj = DTrajectory::new(6);
        assert_eq!(traj.sensitivity_dimensions(), None);

        let mut traj = DTrajectory::new(6);
        traj.enable_sensitivity_storage(4);
        assert_eq!(traj.sensitivity_dimensions(), Some((6, 4)));

        let mut traj = DTrajectory::new(9);
        traj.enable_sensitivity_storage(2);
        assert_eq!(traj.sensitivity_dimensions(), Some((9, 2)));
    }

    // ============================================================================
    // add_full Method Tests
    // ============================================================================

    #[test]
    fn test_dtrajectory_add_full_state_only() {
        let mut traj = DTrajectory::new(6);
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);

        traj.add_full(epoch, state.clone(), None, None, None);

        assert_eq!(traj.len(), 1);
        assert!(traj.covariances.is_none());
        assert!(traj.stms.is_none());
        assert!(traj.sensitivities.is_none());

        let (e, s) = traj.get(0).unwrap();
        assert_eq!(e, epoch);
        assert_abs_diff_eq!(s[0], 7000e3, epsilon = 1.0);
    }

    #[test]
    fn test_dtrajectory_add_full_with_covariance() {
        let mut traj = DTrajectory::new(6);
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let cov = DMatrix::identity(6, 6) * 100.0;

        traj.add_full(epoch, state, Some(cov), None, None);

        assert_eq!(traj.len(), 1);
        assert!(traj.covariances.is_some());
        assert!(traj.stms.is_none());
        assert!(traj.sensitivities.is_none());

        let result_cov = traj.covariances.as_ref().unwrap()[0].clone();
        assert_abs_diff_eq!(result_cov[(0, 0)], 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dtrajectory_add_full_with_stm() {
        let mut traj = DTrajectory::new(6);
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let stm = DMatrix::from_element(6, 6, 2.0);

        traj.add_full(epoch, state, None, Some(stm), None);

        assert_eq!(traj.len(), 1);
        assert!(traj.covariances.is_none());
        assert!(traj.stms.is_some());
        assert!(traj.sensitivities.is_none());

        let result_stm = traj.stms.as_ref().unwrap()[0].clone();
        assert_abs_diff_eq!(result_stm[(0, 0)], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dtrajectory_add_full_with_sensitivity() {
        let mut traj = DTrajectory::new(6);
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let sens = DMatrix::from_element(6, 3, 5.0);

        traj.add_full(epoch, state, None, None, Some(sens));

        assert_eq!(traj.len(), 1);
        assert!(traj.covariances.is_none());
        assert!(traj.stms.is_none());
        assert!(traj.sensitivities.is_some());

        assert_eq!(traj.sensitivity_dimensions(), Some((6, 3)));
        let result_sens = traj.sensitivities.as_ref().unwrap()[0].clone();
        assert_abs_diff_eq!(result_sens[(0, 0)], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dtrajectory_add_full_all_matrices() {
        let mut traj = DTrajectory::new(6);
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let cov = DMatrix::identity(6, 6) * 100.0;
        let stm = DMatrix::from_element(6, 6, 2.0);
        let sens = DMatrix::from_element(6, 3, 5.0);

        traj.add_full(epoch, state, Some(cov), Some(stm), Some(sens));

        assert_eq!(traj.len(), 1);
        assert!(traj.covariances.is_some());
        assert!(traj.stms.is_some());
        assert!(traj.sensitivities.is_some());
    }

    #[test]
    fn test_dtrajectory_add_full_maintains_order() {
        let mut traj = DTrajectory::new(6);
        let t0 = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let state1 = DVector::from_vec(vec![7100e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let state0 = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let state2 = DVector::from_vec(vec![7200e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);

        // Add out of order
        traj.add_full(t1, state1, None, None, None);
        traj.add_full(t0, state0, None, None, None);
        traj.add_full(t2, state2, None, None, None);

        // Should be in chronological order
        assert_eq!(traj.len(), 3);
        assert_eq!(traj.epochs[0], t0);
        assert_eq!(traj.epochs[1], t1);
        assert_eq!(traj.epochs[2], t2);

        assert_abs_diff_eq!(traj.states[0][0], 7000e3, epsilon = 1.0);
        assert_abs_diff_eq!(traj.states[1][0], 7100e3, epsilon = 1.0);
        assert_abs_diff_eq!(traj.states[2][0], 7200e3, epsilon = 1.0);
    }

    #[test]
    #[should_panic(expected = "State vector dimension")]
    fn test_dtrajectory_add_full_state_dimension_mismatch() {
        let mut traj = DTrajectory::new(6);
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let wrong_state = DVector::from_vec(vec![7000e3, 0.0, 0.0]); // Only 3 elements

        traj.add_full(epoch, wrong_state, None, None, None);
    }

    #[test]
    #[should_panic(expected = "STM dimensions")]
    fn test_dtrajectory_add_full_stm_dimension_mismatch() {
        let mut traj = DTrajectory::new(6);
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let wrong_stm = DMatrix::identity(3, 3); // 3x3 instead of 6x6

        traj.add_full(epoch, state, None, Some(wrong_stm), None);
    }

    #[test]
    #[should_panic(expected = "Sensitivity row count")]
    fn test_dtrajectory_add_full_sensitivity_row_mismatch() {
        let mut traj = DTrajectory::new(6);
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
        let wrong_sens = DMatrix::from_element(3, 2, 1.0); // 3 rows instead of 6

        traj.add_full(epoch, state, None, None, Some(wrong_sens));
    }

    #[test]
    #[should_panic(expected = "Sensitivity column count")]
    fn test_dtrajectory_add_full_sensitivity_col_mismatch() {
        let mut traj = DTrajectory::new(6);
        let t0 = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let state = DVector::from_vec(vec![7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0]);

        // First add with 2 columns
        let sens1 = DMatrix::from_element(6, 2, 1.0);
        traj.add_full(t0, state.clone(), None, None, Some(sens1));

        // Second add with 5 columns (inconsistent)
        let sens2 = DMatrix::from_element(6, 5, 1.0);
        traj.add_full(t1, state, None, None, Some(sens2));
    }

    // ============================================================================
    // epoch_initial and find_surrounding_indices Tests
    // ============================================================================

    #[test]
    fn test_dtrajectory_epoch_initial() {
        let traj = create_test_trajectory();
        let initial = traj.epoch_initial();
        assert!(initial.is_some());
        assert_eq!(initial.unwrap(), traj.epochs[0]);
    }

    #[test]
    fn test_dtrajectory_epoch_initial_empty() {
        let traj = DTrajectory::new(6);
        assert!(traj.epoch_initial().is_none());
    }

    #[test]
    fn test_dtrajectory_find_surrounding_indices() {
        let traj = create_test_trajectory();
        let t0 = traj.epochs[0];
        let t1 = traj.epochs[1];
        let mid = t0 + (t1 - t0) / 2.0;

        let result = traj.find_surrounding_indices(mid);
        assert!(result.is_some());
        let (idx0, idx1) = result.unwrap();
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
    }

    #[test]
    fn test_dtrajectory_find_surrounding_indices_empty() {
        let traj = DTrajectory::new(6);
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        assert!(traj.find_surrounding_indices(epoch).is_none());
    }

    #[test]
    fn test_dtrajectory_find_surrounding_indices_before_start() {
        let traj = create_test_trajectory();
        let before = traj.epochs[0] - 100.0;
        assert!(traj.find_surrounding_indices(before).is_none());
    }

    #[test]
    fn test_dtrajectory_find_surrounding_indices_after_end() {
        let traj = create_test_trajectory();
        let after = traj.epochs[2] + 100.0;
        assert!(traj.find_surrounding_indices(after).is_none());
    }

    // ============================================================================
    // Eviction Policy with Extended Data Tests
    // ============================================================================

    #[test]
    fn test_dtrajectory_eviction_keep_count_with_covariances() {
        let mut traj = DTrajectory::new(6).with_eviction_policy_max_size(3);
        traj.enable_covariance_storage();

        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        for i in 0..5 {
            let epoch = t0 + (i as f64 * 60.0);
            let state =
                DVector::from_vec(vec![7000e3 + i as f64 * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
            let cov = DMatrix::identity(6, 6) * (i as f64 * 10.0);
            traj.add_with_covariance(epoch, state, cov);
        }

        // Should only have 3 states and 3 covariances
        assert_eq!(traj.len(), 3);
        assert_eq!(traj.covariances.as_ref().unwrap().len(), 3);

        // First covariance should be from the third state (i=2)
        let cov = traj.covariances.as_ref().unwrap()[0].clone();
        assert_abs_diff_eq!(cov[(0, 0)], 20.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dtrajectory_eviction_keep_count_with_stms() {
        let mut traj = DTrajectory::new(6).with_eviction_policy_max_size(3);
        traj.enable_stm_storage();

        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        for i in 0..5 {
            let epoch = t0 + (i as f64 * 60.0);
            let state =
                DVector::from_vec(vec![7000e3 + i as f64 * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
            traj.add(epoch, state);
            let stm = DMatrix::from_element(6, 6, i as f64);
            traj.set_stm_at(traj.len() - 1, stm);
        }

        // Should only have 3 states and 3 STMs
        assert_eq!(traj.len(), 3);
        assert_eq!(traj.stms.as_ref().unwrap().len(), 3);

        // First STM should be from the third state (i=2)
        let stm = traj.stms.as_ref().unwrap()[0].clone();
        assert_abs_diff_eq!(stm[(0, 0)], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dtrajectory_eviction_keep_count_with_sensitivities() {
        let mut traj = DTrajectory::new(6).with_eviction_policy_max_size(3);
        traj.enable_sensitivity_storage(2);

        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        for i in 0..5 {
            let epoch = t0 + (i as f64 * 60.0);
            let state =
                DVector::from_vec(vec![7000e3 + i as f64 * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
            traj.add(epoch, state);
            let sens = DMatrix::from_element(6, 2, i as f64 * 10.0);
            traj.set_sensitivity_at(traj.len() - 1, sens);
        }

        // Should only have 3 states and 3 sensitivities
        assert_eq!(traj.len(), 3);
        assert_eq!(traj.sensitivities.as_ref().unwrap().len(), 3);

        // First sensitivity should be from the third state (i=2)
        let sens = traj.sensitivities.as_ref().unwrap()[0].clone();
        assert_abs_diff_eq!(sens[(0, 0)], 20.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dtrajectory_eviction_keep_count_all_data() {
        let mut traj = DTrajectory::new(6).with_eviction_policy_max_size(2);

        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        for i in 0..4 {
            let epoch = t0 + (i as f64 * 60.0);
            let state =
                DVector::from_vec(vec![7000e3 + i as f64 * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
            let cov = DMatrix::identity(6, 6) * (i as f64);
            let stm = DMatrix::from_element(6, 6, i as f64 * 2.0);
            let sens = DMatrix::from_element(6, 3, i as f64 * 3.0);
            traj.add_full(epoch, state, Some(cov), Some(stm), Some(sens));
        }

        // Should only have 2 of each
        assert_eq!(traj.len(), 2);
        assert_eq!(traj.covariances.as_ref().unwrap().len(), 2);
        assert_eq!(traj.stms.as_ref().unwrap().len(), 2);
        assert_eq!(traj.sensitivities.as_ref().unwrap().len(), 2);

        // First values should be from i=2
        assert_abs_diff_eq!(
            traj.covariances.as_ref().unwrap()[0][(0, 0)],
            2.0,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(traj.stms.as_ref().unwrap()[0][(0, 0)], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(
            traj.sensitivities.as_ref().unwrap()[0][(0, 0)],
            6.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_dtrajectory_eviction_keep_within_duration_with_covariances() {
        let mut traj = DTrajectory::new(6).with_eviction_policy_max_age(150.0); // 150 seconds

        traj.enable_covariance_storage();
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        for i in 0..5 {
            let epoch = t0 + (i as f64 * 60.0); // 60 seconds apart
            let state =
                DVector::from_vec(vec![7000e3 + i as f64 * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
            let cov = DMatrix::identity(6, 6) * (i as f64 * 10.0);
            traj.add_with_covariance(epoch, state, cov);
        }

        // With 150s max age and 60s intervals, should keep 3 states (t4, t3, t2)
        // t4-t2 = 120s <= 150s, t4-t1 = 180s > 150s
        assert_eq!(traj.len(), 3);
        assert_eq!(traj.covariances.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_dtrajectory_eviction_keep_within_duration_with_stms() {
        let mut traj = DTrajectory::new(6).with_eviction_policy_max_age(150.0); // 150 seconds

        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        for i in 0..5 {
            let epoch = t0 + (i as f64 * 60.0); // 60 seconds apart
            let state =
                DVector::from_vec(vec![7000e3 + i as f64 * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
            let stm = DMatrix::from_element(6, 6, i as f64);
            traj.add_full(epoch, state, None, Some(stm), None);
        }

        // With 150s max age and 60s intervals, should keep 3 states
        assert_eq!(traj.len(), 3);
        assert_eq!(traj.stms.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_dtrajectory_eviction_keep_within_duration_with_sensitivities() {
        let mut traj = DTrajectory::new(6).with_eviction_policy_max_age(150.0); // 150 seconds

        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        for i in 0..5 {
            let epoch = t0 + (i as f64 * 60.0); // 60 seconds apart
            let state =
                DVector::from_vec(vec![7000e3 + i as f64 * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
            let sens = DMatrix::from_element(6, 2, i as f64);
            traj.add_full(epoch, state, None, None, Some(sens));
        }

        // With 150s max age and 60s intervals, should keep 3 states
        assert_eq!(traj.len(), 3);
        assert_eq!(traj.sensitivities.as_ref().unwrap().len(), 3);
    }

    // ============================================================================
    // with_interpolation_method Tests
    // ============================================================================

    #[test]
    fn test_dtrajectory_with_interpolation_method_builder_pattern() {
        let traj = DTrajectory::new(6).with_interpolation_method(InterpolationMethod::Linear);
        assert_eq!(traj.get_interpolation_method(), InterpolationMethod::Linear);
    }

    #[test]
    fn test_dtrajectory_with_interpolation_method_lagrange() {
        let traj = DTrajectory::new(6)
            .with_interpolation_method(InterpolationMethod::Lagrange { degree: 5 });
        match traj.get_interpolation_method() {
            InterpolationMethod::Lagrange { degree } => assert_eq!(degree, 5),
            _ => panic!("Expected Lagrange interpolation method"),
        }
    }
}
